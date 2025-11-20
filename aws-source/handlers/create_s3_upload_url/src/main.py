from logging import getLogger
import os
import json
import boto3
from botocore import config
import uuid
import re
import unicodedata

S3_UPLOAD_BUCKET = os.getenv("S3_UPLOAD_BUCKET", "")
PRESIGNED_URL_EXPIRATION_SECONDS = 60 * 10  # 10 minutes
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB 

s3_client = boto3.client(
    "s3", config=config.Config(signature_version="v4")
)

logger = getLogger()
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))

# Document types for employee onboarding
DOCUMENT_TYPES = {
    "employee_payslip": {"extension": ".pdf", "description": "Payslip (Lønseddel)"},
    "employee_active": {"extension": ".xlsx", "description": "Active Employee List"},
    "employee_holiday": {"extension": ".xlsx", "description": "Holiday Records"},
    "employee_general": {"extension": ".pdf", "description": "Employee General Info"},
    "employee_list": {"extension": ".pdf", "description": "Employee List"}
}


def handler(event, context):
    try:
        # Parse request body
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event

        files = body.get('files', [])

        if not files:
            logger.error("Missing files array")
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                },
                "body": json.dumps({"error": "Missing files array in request body"}),
            }

        # Generate presigned URLs for each file
        upload_urls = []
        session_id = str(uuid.uuid4())

        for file_info in files:
            file_type = file_info.get("file_type", "")
            filename = file_info.get("file_name", "")

            if not file_type or not filename:
                logger.error(f"Missing file_type or file_name: {file_info}")
                continue

            if file_type not in DOCUMENT_TYPES:
                logger.error(f"Invalid file type: {file_type}")
                continue

            # Validate file extension
            file_extension = os.path.splitext(filename)[1].lower()
            expected_extension = DOCUMENT_TYPES[file_type]["extension"]

            if file_extension != expected_extension:
                logger.error(f"Invalid extension {file_extension} for type {file_type}")
                continue

            safe_filename = sanitize_filename(filename)
            s3_key = f"uploads/{session_id}/{file_type}/{safe_filename}"

            logger.info(f"Generating presigned POST for: {s3_key}")

            presigned_post = s3_client.generate_presigned_post(
                Bucket=S3_UPLOAD_BUCKET,
                Key=s3_key,
                ExpiresIn=PRESIGNED_URL_EXPIRATION_SECONDS,
                Conditions=[
                    ["content-length-range", 1, MAX_FILE_SIZE_BYTES],
                ],
            )

            upload_urls.append({
                "file_type": file_type,
                "upload_url": presigned_post["url"],
                "fields": presigned_post["fields"],
                "s3_key": s3_key,
                "file_name": safe_filename
            })

        if not upload_urls:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                },
                "body": json.dumps({"error": "No valid files to upload"}),
            }

        response_data = {
            "session_id": session_id,
            "upload_urls": upload_urls,
            "max_file_size_mb": MAX_FILE_SIZE_BYTES / (1024 * 1024),
            "expires_in_seconds": PRESIGNED_URL_EXPIRATION_SECONDS
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
            },
            "body": json.dumps(response_data),
        }

    except Exception as e:
        logger.error(f"Error generating presigned POST URL: {e}")
        import traceback
        logger.error(traceback.format_exc())

        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
            },
            "body": json.dumps({"error": "Failed to generate upload URL"}),
        }


def sanitize_filename(filename: str) -> str:
    """
    Turn arbitrary filename into an S3-safe slug:
    - Normalize Unicode to NFKD (e.g. "é" → "e")
    - Strip out any remaining non-ASCII
    - Replace spaces with underscores
    - Replace any character not in [A-Za-z0-9._-] with underscore
    - Collapse multiple underscores into one
    """
    # 1) Normalize Unicode & drop non-ASCII
    nfkd = unicodedata.normalize("NFKD", filename)
    ascii_only = nfkd.encode("ascii", "ignore").decode("ascii")

    # 2) Replace spaces with underscore
    no_spaces = ascii_only.strip().replace(" ", "_")

    # 3) Replace disallowed chars
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", no_spaces)

    # 4) Collapse repeats
    safe = re.sub(r"_+", "_", safe).strip("_")

    # 5) Ensure we didn't end up empty
    return safe or "video"
