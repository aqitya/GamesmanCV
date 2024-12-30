import numpy as np
import cv2
import json
import boto3

grid = np.zeros((6, 7))

YELLOW_LOWER_HSV = np.array([20, 100, 100])
YELLOW_UPPER_HSV = np.array([30, 255, 255])

RED_LOWER_HSV = np.array([140,  75,  45])
RED_UPPER_HSV = np.array([180, 255, 255])

convert_moves = {
  1: 'M_42_49_x',
  2: 'M_43_50_x',
  3: 'M_44_51_x',
  4: 'M_45_52_x',
  5: 'M_46_53_x',
  6: 'M_47_54_x',
  7: 'M_48_55_x'
}

def order_corners(corners):
    """ Orders the corners in the following order: top-left, top-right, bottom-right, bottom-left """
    corners = np.array(corners)
    sum_points = corners.sum(axis=1)
    top_left = corners[np.argmin(sum_points)]
    bottom_right = corners[np.argmax(sum_points)]
    diff_points = np.diff(corners, axis=1)
    top_right = corners[np.argmin(diff_points)]
    bottom_left = corners[np.argmax(diff_points)]
    return np.array([top_left, top_right, bottom_right, bottom_left])

def grid_to_position_string(grid):
  rows, cols = grid.shape
  position_string = ''

  for col in range(cols):
    for row in range(rows):
      if grid[row][col] == -1:
        position_string += 'X'
      elif grid[row][col] == 1:
        position_string += 'O'
      else:
        position_string += '-'

  return position_string

def is_motion(previous_frame, current_frame, threshold=10):
  frame_delta = cv2.absdiff(previous_frame, current_frame)
  thresholded = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
  motion_level = np.sum(thresholded)
  return motion_level > threshold

def check_winner(grid):
    rows, cols = len(grid), len(grid[0])

    grid_array = np.array(grid)

    for row in grid_array:
        for i in range(cols - 3):
            if np.array_equal(row[i:i + 4], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal(row[i:i + 4], np.array([-1, -1, -1, -1])):
                return -1

    for col in range(cols):
        for i in range(rows - 3):
            if np.array_equal(grid_array[i:i + 4, col], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal(grid_array[i:i + 4, col], np.array([-1, -1, -1, -1])):
                return -1

    for row in range(rows - 3):
        for col in range(cols - 3):
            if np.array_equal([grid_array[row + i][col + i] for i in range(4)], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal([grid_array[row + i][col + i] for i in range(4)], np.array([-1, -1, -1, -1])):
                return -1

    for row in range(3, rows):
        for col in range(cols - 3):
            if np.array_equal([grid_array[row - i][col + i] for i in range(4)], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal([grid_array[row - i][col + i] for i in range(4)], np.array([-1, -1, -1, -1])):
                return -1
    return 0

def is_color_dominant(mask):
    white_pixels = cv2.countNonZero(mask)
    white_area_ratio = white_pixels / mask.size
    return white_area_ratio > 0.2

def corners(frame, low_range, high_range):
  hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv_image, low_range, high_range)
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  ret = []
  for i in contours:
    if cv2.contourArea(i) > 100:
      M = cv2.moments(i)
      if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        ret.append((cX, cY))
  return ret

def detect_column_change(previous_position_string, current_position_string):
    for col in range(7):
        prev_column = previous_position_string[col*6:(col+1)*6]
        curr_column = current_position_string[col*6:(col+1)*6]
        
        if prev_column != curr_column:
            return col + 1
    
    return -1

def process_cell(img, x_start, y_start, width, height):
    cell_img = img[y_start:y_start + height, x_start:x_start + width]
    hsv_cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv_cell_img, RED_LOWER_HSV, RED_UPPER_HSV)
    yellow_mask = cv2.inRange(hsv_cell_img, YELLOW_LOWER_HSV, YELLOW_UPPER_HSV)
    if is_color_dominant(red_mask):
        return 1
    elif is_color_dominant(yellow_mask):
        return -1
    return 0

def process_frame(frame, bounding_box, corners):
    min_x, min_y, max_x, max_y = bounding_box
    pts1 = np.float32(corners)
    width = max_x - min_x
    height = max_y - min_y
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    rectified_frame = cv2.warpPerspective(frame, M, (width, height))

    new_width = 500  
    img_h, img_w, _ = rectified_frame.shape
    scale = new_width / img_w
    img_w = int(img_w * scale)
    img_h = int(img_h * scale)
    img = cv2.resize(rectified_frame, (img_w, img_h), interpolation=cv2.INTER_AREA)

    bilateral_filtered_image = cv2.bilateralFilter(img, 15, 190, 190)

    cell_width = img_w // 7
    cell_height = img_h // 6
    grid = np.zeros((6, 7))
    for row in range(6):
        for col in range(7):
            x_start = col * cell_width
            y_start = row * cell_height
            grid[row, col] = process_cell(bilateral_filtered_image, x_start, y_start, cell_width, cell_height)
    return grid


def extract_frames(video_path, skip_frames=20):
  cap = cv2.VideoCapture(video_path)
  previous_position_string = None
  output_strings = []
  moves = []
  auto_gui_moves = []

  ret, previous_frame = cap.read()
  if not ret:
      output_strings.append("Error: Cannot read frame from video.")
      cap.release()
      return output_strings
  
  previous_position_string = '------------------------------------------'
  p, frame_count = 0, 0
  
  green_lower_hsv = np.array([30, 30, 30])
  green_upper_hsv = np.array([85, 230, 170])
  C4 = corners(previous_frame, green_lower_hsv, green_upper_hsv)
  C4 = order_corners(C4)

  if len(C4) != 4:
    cap.release()
    return "Error: Did not find exactly 4 green regions."

  x_coords = [coord[0] for coord in C4]
  y_coords = [coord[1] for coord in C4]
  min_x, max_x = min(x_coords), max(x_coords)
  min_y, max_y = min(y_coords), max(y_coords)
  bounding_box = (min_x, min_y, max_x, max_y)

  previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
  previous_frame = cv2.GaussianBlur(previous_frame, (21, 21), 0)
  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      gray = cv2.GaussianBlur(gray, (21, 21), 0)
      if frame_count % skip_frames == 0:
          if not is_motion(previous_frame, gray):
              board_array = process_frame(frame, bounding_box, C4)
              current_position_string = grid_to_position_string(board_array)
              
              if current_position_string != previous_position_string:
                  output_strings.append(f'p={str((p % 2) + 1)}_{current_position_string}')
                  p += 1
                  column_played = detect_column_change(previous_position_string, current_position_string)
                  moves.append(column_played)
                  auto_gui_moves.append(convert_moves[column_played])

              result = check_winner(board_array)
              if result == -1:
                  output_strings.append("Player 2 RED wins!")
                  break
              elif result == 1:
                  output_strings.append("Player 1 BLUE wins!")
                  break

              previous_position_string = current_position_string
          previous_frame = gray
      frame_count += 1

  cap.release()
  return auto_gui_moves
  # return output_strings, auto_gui_moves, moves


def lambda_handler(event, context):
    # Extract bucket name and object key from the event
    bucket_name = event.get('bucket')
    object_key = event.get('key')
    
    if not bucket_name or not object_key:
        return {
            "statusCode": 400,
            "body": json.dumps({"message": "Bucket name or object key is missing from the event"})
        }
    
    download_path = '/tmp/' + object_key
    
    # Download the video file from S3
    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(bucket_name, object_key, download_path)
        print(f"Successfully downloaded {object_key} to {download_path}")
    except Exception as e:
        print(f"Error downloading object: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"message": "Error downloading object", "error": str(e)})
        }
    
    # Process the video to extract moves
    print("Attempting to extract frames from the downloaded file...")
    try:
        move_objects = extract_frames(download_path)

        if move_objects is None or not isinstance(move_objects, list):
            print("Error processing the video file.")
            return {
                "statusCode": 500,
                "body": json.dumps({"message": "Error processing the video file."})
            }

        # Return the list of move objects as a newline-separated string
        return {
            "statusCode": 200,
            "body": json.dumps({"moves": "\n".join(move_objects)})
        }
    except Exception as e:
        print(f"Error processing the video: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"message": "Error processing the video", "error": str(e)})
        }
