# Employee Onboarding Document Processor

A document processing system that extracts data from Danish employee documents and generates standardized CSV output for Danlon payroll system integration.

## Overview

This application processes multiple employee document types using OCR (Optical Character Recognition) and merges the extracted data into a single CSV file formatted for Danlon import.

### Supported Document Types

- **Payslip (Lønseddel)** - PDF format
- **Active Employee List (Medarbejderoversigt)** - Excel format
- **Holiday Records (Feriepengeforpligtelse)** - Excel format
- **Employee General Information (Medarbejder Stamkort)** - PDF format
- **Employee List (Medarbejderliste)** - PDF format

## Architecture

### Components

- **Frontend**: React application with AWS Amplify authentication
- **Backend**: AWS Lambda function with Docker container (Python 3.11)
- **Storage**: Amazon S3 for processed CSV results
- **Infrastructure**: Terraform for AWS resource management
- **OCR Engine**: Tesseract with Danish language support

### Processing Flow

1. User uploads documents through web interface
2. Frontend sends files to Lambda via Function URL
3. Lambda processes documents (6-8 minutes for OCR)
4. Processed CSV uploaded to S3 with KMS encryption
5. Frontend downloads CSV via presigned URL

## Prerequisites

- AWS Account with appropriate permissions
- Terraform >= 1.0
- Docker
- Node.js >= 16
- Python 3.11

## Project Structure

```
.
├── frontend/                 # React application
│   ├── src/
│   └── public/
├── aws-source/
│   └── handlers_docker/
│       └── generate_output/  # Lambda function
│           ├── src/
│           ├── mapping/      # Danlon mapping configuration
│           └── Dockerfile
├── infra/                    # Terraform infrastructure
│   ├── modules/
│   │   ├── aws-s3/
│   │   ├── aws-lambda-docker/
│   │   └── aws-ecr/
│   └── main.tf
└── README.md
```

## Setup

### 1. Infrastructure Deployment

```bash
cd infra

# Initialize Terraform
terraform init

# Review planned changes
terraform plan

# Deploy infrastructure
terraform apply
```

This creates:
- ECR repository for Lambda Docker images
- Lambda function with 15-minute timeout and 3GB memory
- S3 bucket for storing results (KMS encrypted)
- IAM roles and policies
- Lambda Function URL with CORS configuration

### 2. Lambda Deployment

```bash
cd aws-source/handlers_docker/generate_output

# Build Docker image
DOCKER_BUILDKIT=0 docker build --platform linux/arm64 -t employee-onboarding-generate-output-ecr-repo:latest .

# Login to ECR
aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com

# Tag for ECR
docker tag employee-onboarding-generate-output-ecr-repo:latest <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/employee-onboarding-generate-output-ecr-repo:latest

# Push to ECR
docker push <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/employee-onboarding-generate-output-ecr-repo:latest

# Update Lambda
cd ../../../infra
terraform apply
```

## Usage

### Processing Documents

1. Navigate to the web application
2. Sign in with AWS Amplify credentials
3. Upload required documents:
   - At least one of: Active Employee List, Employee General, or Payslip
   - Optional: Holiday Records, Employee List
4. Click "Process Documents"
5. Wait 6-8 minutes for processing
6. CSV file downloads automatically when complete

### Output Format

The generated CSV follows Danlon's import schema with 133 columns including:
- Employee identification (name, employee number)
- Address and contact information
- Employment details (salary, position, hours)
- Tax information
- Pension and benefits
- Holiday balances

## Technical Details

### OCR Processing

- **DPI**: 200 (default quality for accuracy)
- **Language**: Danish (dan)
- **Engine**: Tesseract with LSTM
- **Processing Mode**: Sequential (1 PDF at a time) to manage memory

