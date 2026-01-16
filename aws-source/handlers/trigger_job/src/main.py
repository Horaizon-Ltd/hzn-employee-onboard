"""Lambda for triggering employee data processing jobs."""
import json
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

import boto3
from botocore.exceptions import ClientError

# Constants
DEFAULT_AWS_REGION = "eu-west-1"
JOB_TTL_DAYS = 7  # Auto-delete jobs after 7 days

# Initialize logging
logger = logging.getLogger()
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))

# Initialize AWS clients
_aws_region = os.getenv("AWS_REGION", DEFAULT_AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=_aws_region)
lambda_client = boto3.client('lambda', region_name=_aws_region)

# Environment variables
JOBS_TABLE = os.getenv("JOBS_TABLE")
WORKER_LAMBDA_NAME = os.getenv("WORKER_LAMBDA_NAME")


def _create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a standardized HTTP response."""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
        },
        'body': json.dumps(body)
    }


def _create_job_record(job_id: str, s3_files: list) -> None:
    """
    Create a job record in DynamoDB with PENDING status.

    Args:
        job_id: Unique job identifier
        s3_files: List of S3 file info to process
    """
    if not JOBS_TABLE:
        raise ValueError("JOBS_TABLE environment variable not set")

    jobs_table = dynamodb.Table(JOBS_TABLE)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=JOB_TTL_DAYS)

    jobs_table.put_item(Item={
        'job_id': job_id,
        'status': 'PENDING',
        's3_files': s3_files,
        'created_at': now.isoformat(),
        'updated_at': now.isoformat(),
        'expires_at': int(expires_at.timestamp())
    })

    logger.info(f"Created job record: {job_id} with status PENDING")


def _invoke_worker_async(job_id: str, s3_files: list) -> None:
    """
    Invoke the worker Lambda asynchronously.

    Args:
        job_id: Job ID to process
        s3_files: List of S3 file info to process
    """
    if not WORKER_LAMBDA_NAME:
        raise ValueError("WORKER_LAMBDA_NAME environment variable not set")

    payload = {
        'job_id': job_id,
        's3_files': s3_files
    }

    lambda_client.invoke(
        FunctionName=WORKER_LAMBDA_NAME,
        InvocationType='Event',  # Async invocation
        Payload=json.dumps(payload)
    )

    logger.info(f"Invoked worker Lambda async for job: {job_id}")


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handle requests to trigger employee data processing.

    Creates a job record in DynamoDB with status=PENDING,
    then invokes the worker Lambda asynchronously.

    Args:
        event: API Gateway event or direct invocation
        context: Lambda context

    Returns:
        HTTP response with job_id
    """
    # Validate configuration
    if not JOBS_TABLE:
        logger.error("JOBS_TABLE environment variable not set")
        return _create_response(500, {'error': 'Configuration error: JOBS_TABLE not set'})

    if not WORKER_LAMBDA_NAME:
        logger.error("WORKER_LAMBDA_NAME environment variable not set")
        return _create_response(500, {'error': 'Configuration error: WORKER_LAMBDA_NAME not set'})

    try:
        # Parse request body
        if 'body' in event:
            if event.get('isBase64Encoded', False):
                import base64
                body = base64.b64decode(event['body']).decode('utf-8')
            else:
                body = event['body']
            request_data = json.loads(body) if isinstance(body, str) else body
        else:
            request_data = event

        # Extract s3_files from request
        s3_files = request_data.get('s3_files', [])

        if not s3_files:
            return _create_response(400, {'error': 'Missing s3_files in request'})

        # Validate s3_files structure
        for f in s3_files:
            if not f.get('s3_key') or not f.get('file_type'):
                return _create_response(400, {
                    'error': 'Each file must have s3_key and file_type'
                })

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Create job record in DynamoDB
        try:
            _create_job_record(job_id, s3_files)
        except ClientError as e:
            logger.error(f"DynamoDB error creating job record: {e}", exc_info=True)
            return _create_response(500, {'error': 'Failed to create job'})

        # Invoke worker Lambda asynchronously
        try:
            _invoke_worker_async(job_id, s3_files)
        except ClientError as e:
            logger.error(f"Error invoking worker Lambda: {e}", exc_info=True)
            # Update job status to ERROR
            try:
                jobs_table = dynamodb.Table(JOBS_TABLE)
                jobs_table.update_item(
                    Key={'job_id': job_id},
                    UpdateExpression='SET #status = :status, #error = :error, updated_at = :updated_at',
                    ExpressionAttributeNames={
                        '#status': 'status',
                        '#error': 'error'
                    },
                    ExpressionAttributeValues={
                        ':status': 'ERROR',
                        ':error': f'Failed to start processing: {str(e)}',
                        ':updated_at': datetime.now(timezone.utc).isoformat()
                    }
                )
            except Exception:
                pass  # Best effort to update status
            return _create_response(500, {'error': 'Failed to start processing'})

        # Return 202 Accepted with job_id
        logger.info(f"Job {job_id} created and worker invoked successfully")
        return _create_response(202, {
            'message': 'Processing started',
            'job_id': job_id,
            'status': 'PENDING'
        })

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request body: {e}")
        return _create_response(400, {'error': 'Invalid JSON in request body'})
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return _create_response(500, {'error': 'Internal server error'})
