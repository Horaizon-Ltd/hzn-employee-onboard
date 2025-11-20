output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = module.ecr_repo.repository_url
}

output "ecr_repository_name" {
  description = "ECR repository name"
  value       = module.ecr_repo.repository_name
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = module.docker_lambda.function_name
}

output "lambda_function_arn" {
  description = "Lambda function ARN"
  value       = module.docker_lambda.arn
}

output "lambda_function_url" {
  description = "Lambda Function URL for HTTP access"
  value       = var.enable_function_url ? aws_lambda_function_url.generate_output_url[0].function_url : "Function URL not enabled"
}

output "aws_region" {
  description = "AWS region where resources are deployed"
  value       = var.region
}

output "results_bucket_name" {
  value       = module.results_bucket.bucket
  description = "S3 bucket name for storing processed CSV results"
  sensitive   = true
}

output "uploads_bucket_name" {
  value       = module.file_bucket.bucket
  description = "S3 bucket name for uploading input documents"
  sensitive   = true
}

output "upload_url_function_url" {
  value       = var.enable_function_url ? aws_lambda_function_url.create_s3_upload_url_url[0].function_url : "Function URL not enabled"
  description = "Lambda Function URL for generating presigned upload URLs"
}
