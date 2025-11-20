module "ecr_repo" {
  source = "./modules/aws-ecr"
  name = "${var.project_name}-generate-output"
}

module "results_bucket" {
  source      = "./modules/aws-s3"
  project     = var.project_name
  environment = var.environment
  name        = "${var.project_name}-results-${var.environment}"
  acl         = "private"
}

module "file_bucket" {
  source      = "./modules/aws-s3"
  project     = var.project_name
  environment = var.environment
  name        = "${var.project_name}-${var.environment}"
  acl         = "private"
}

resource "aws_iam_policy" "lambda_s3_policy" {
  name        = "${var.project_name}-lambda-s3-access-${var.environment}"
  description = "Allow Lambda to read/write S3 buckets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:PutObjectAcl",
          "s3:GetObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${module.results_bucket.bucket_arn}/*",
          "${module.file_bucket.bucket_arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey"
        ]
        Resource = [
          module.results_bucket.kms_key_arn,
          module.file_bucket.kms_key_arn
        ]
      }
    ]
  })
}

data "archive_file" "create_s3_upload_url_zip" {
  type        = "zip"
  source_dir  = "../aws-source/handlers/create_s3_upload_url/src"
  output_path = "${path.module}/.terraform/create_s3_upload_url.zip"
}

resource "aws_iam_role" "create_s3_upload_url_role" {
  name = "${var.project_name}-create-s3-upload-url-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "create_s3_upload_url_s3_policy" {
  role       = aws_iam_role.create_s3_upload_url_role.name
  policy_arn = aws_iam_policy.lambda_s3_policy.arn
}

resource "aws_iam_role_policy_attachment" "create_s3_upload_url_basic_execution" {
  role       = aws_iam_role.create_s3_upload_url_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_lambda_function" "create_s3_upload_url" {
  filename         = data.archive_file.create_s3_upload_url_zip.output_path
  function_name    = "${var.project_name}-create-s3-upload-url-${var.environment}"
  role             = aws_iam_role.create_s3_upload_url_role.arn
  handler          = "main.handler"
  source_code_hash = data.archive_file.create_s3_upload_url_zip.output_base64sha256
  runtime          = "python3.11"
  timeout          = 30

  environment {
    variables = {
      S3_UPLOAD_BUCKET  = module.file_bucket.bucket
      LOGGING_LEVEL     = "INFO"
    }
  }
}

resource "aws_lambda_function_url" "create_s3_upload_url_url" {
  count = var.enable_function_url ? 1 : 0

  function_name      = aws_lambda_function.create_s3_upload_url.function_name
  authorization_type = "NONE"

  cors {
    allow_origins = var.cors_allow_origins
    allow_methods = ["*"]
    allow_headers = var.cors_allow_headers
    max_age       = 86400
  }
}

resource "aws_lambda_permission" "allow_function_url_create_s3_upload_url" {
  count = var.enable_function_url ? 1 : 0

  statement_id           = "AllowFunctionURLInvoke"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.create_s3_upload_url.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

module "docker_lambda" {
  source = "./modules/aws-lambda-docker"

  function_name           = "generate-output"
  environment             = var.environment
  project                 = var.project_name
  aws_ecr_repository_name = module.ecr_repo.repository_name
  region                  = var.region
  timeout                 = tostring(var.lambda_timeout)

  source_code_path = "../aws-source/handlers_docker/generate_output"

  environment_variables = {
    ENVIRONMENT      = var.environment
    LOG_LEVEL        = "INFO"
    RESULTS_BUCKET   = module.results_bucket.bucket
    UPLOADS_BUCKET   = module.file_bucket.bucket
  }

  additional_policies = [aws_iam_policy.lambda_s3_policy.arn]

  layers = []
}

resource "aws_lambda_function_url" "generate_output_url" {
  count = var.enable_function_url ? 1 : 0

  function_name      = module.docker_lambda.function_name
  authorization_type = "NONE"

  cors {
    allow_origins = var.cors_allow_origins
    allow_methods = var.cors_allow_methods
    allow_headers = var.cors_allow_headers
    max_age       = 86400
  }
}

resource "aws_lambda_permission" "allow_function_url" {
  count = var.enable_function_url ? 1 : 0

  statement_id           = "AllowFunctionURLInvoke"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = module.docker_lambda.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}
