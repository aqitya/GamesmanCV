import json
import boto3
from urllib.parse import unquote_plus
import botocore
from botocore.config import Config

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    myconfig = Config(connect_timeout=900, read_timeout=900, retries={'max_attempts': 0})
    lambda_client = boto3.client('lambda', config=myconfig)

    try:
        print('Received event:', json.dumps(event))

        if 'Records' in event:
            s3_event = event
        elif 'body' in event and event['body']:
            try:
                s3_event = json.loads(event['body'])
            except json.JSONDecodeError as e:
                print(f"Error parsing event body: {str(e)}")
                return {
                    "statusCode": 400,
                    "body": json.dumps({"message": "Invalid JSON in request body."})
                }
        else:
            return {
                "statusCode": 400,
                "body": json.dumps({"message": "No Records found in event."})
            }

        if 'Records' not in s3_event:
            return {
                "statusCode": 400,
                "body": json.dumps({"message": "No Records found in event after parsing."})
            }

        # Extract bucket name and object key from the S3 event
        record = s3_event['Records'][0]
        bucket_name = record['s3']['bucket']['name']
        object_key = unquote_plus(record['s3']['object']['key'])

        # Check file size using head_object
        response = s3.head_object(Bucket=bucket_name, Key=object_key)
        file_size = response['ContentLength']

        # Check if the file exceeds the size limit (10 MB)
        if file_size > 10 * 1024 * 1024:  # 10 MB
            s3.delete_object(Bucket=bucket_name, Key=object_key)
            print(f"Deleted {object_key} due to size restriction (size: {file_size} bytes)")
            return {
                "statusCode": 413,
                "body": json.dumps({"message": "File size exceeds limit and was deleted."})
            }

        print(f"Processing video from bucket: {bucket_name}, key: {object_key}, size: {file_size}")

        response = lambda_client.invoke(
            FunctionName='perms',
            InvocationType='RequestResponse',
            Payload=json.dumps({
                'bucket': bucket_name,
                'key': object_key
            })
        )

        response_payload = json.loads(response['Payload'].read())

        if response.get('StatusCode') == 200:
            result = response_payload.get('body', '')
            print("Result from 'perms' Lambda function:", result)
            return {
                "statusCode": 200,
                "body": result
            }
        else:
            print(f"Error from 'perms' Lambda function: {response_payload}")
            return {
                "statusCode": response.get('StatusCode', 500),
                "body": json.dumps({"message": "Error in perms function", "error": response_payload})
            }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"message": "Internal server error", "error": str(e)})
        }
