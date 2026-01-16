"""Lambda for checking job status."""
import json
import logging
import os
from decimal import Decimal
from typing import Dict, Any, Optional
from urllib.parse import unquote

import boto3
from botocore.exceptions import ClientError

# Constants
DEFAULT_AWS_REGION = "eu-west-1"

# Initialize logging
logger = logging.getLogger()
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))

# Initialize DynamoDB
_aws_region = os.getenv("AWS_REGION", DEFAULT_AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=_aws_region)

# Environment variables
JOBS_TABLE = os.getenv("JOBS_TABLE")


def _create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a standardized HTTP response."""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET, OPTIONS'
        },
        'body': json.dumps(body)
    }


def _convert_decimals(obj):
    """Convert Decimal types to int/float for JSON serialization."""
    if isinstance(obj, Decimal):
        if obj % 1 == 0:
            return int(obj)
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_decimals(i) for i in obj]
    return obj


def _format_job_response(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format job data for API response.

    Args:
        job: Raw job data from DynamoDB

    Returns:
        Formatted job response
    """
    # Convert all Decimals first
    job = _convert_decimals(job)

    # Core fields
    response = {
        'job_id': job.get('job_id'),
        'status': job.get('status'),
        'created_at': job.get('created_at'),
        'updated_at': job.get('updated_at'),
    }

    # Optional fields - only include if present
    optional_fields = [
        'completed_at',
        'error',
        'download_url',
        'file_name',
        'records_processed',
        'files_processed',
        'processing_summary'
    ]

    for field in optional_fields:
        if field in job and job[field] is not None:
            response[field] = job[field]

    return response


def _extract_job_id(event: Dict[str, Any]) -> Optional[str]:
    """
    Extract job_id from event (path parameters or query string).

    Args:
        event: API Gateway event

    Returns:
        job_id or None
    """
    # Try path parameters first
    path_params = event.get('pathParameters') or {}
    job_id = path_params.get('job_id')
    if job_id:
        job_id = unquote(job_id)

    if not job_id:
        # Try query string parameters
        query_params = event.get('queryStringParameters') or {}
        job_id = query_params.get('job_id')
        if job_id:
            job_id = unquote(job_id)

    return job_id


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handle requests to check job status.

    Args:
        event: API Gateway event or Lambda Function URL event
        context: Lambda context

    Returns:
        HTTP response with job status
    """
    # Validate configuration
    if not JOBS_TABLE:
        logger.error("JOBS_TABLE environment variable not set")
        return _create_response(500, {'error': 'Configuration error: JOBS_TABLE not set'})

    try:
        # Extract job_id
        job_id = _extract_job_id(event)

        if not job_id:
            return _create_response(400, {'error': 'Missing job_id parameter'})

        logger.info(f"Checking status for job: {job_id}")

        # Get job from DynamoDB
        jobs_table = dynamodb.Table(JOBS_TABLE)

        try:
            response = jobs_table.get_item(Key={'job_id': job_id})
        except ClientError as e:
            logger.error(f"DynamoDB error getting job: {e}", exc_info=True)
            return _create_response(500, {'error': 'Failed to get job status'})

        if 'Item' not in response:
            logger.warning(f"Job not found: {job_id}")
            return _create_response(404, {'error': 'Job not found'})

        job = response['Item']
        job_response = _format_job_response(job)

        logger.info(f"Job {job_id} status: {job_response.get('status')}")

        return _create_response(200, job_response)

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return _create_response(500, {'error': 'Internal server error'})

