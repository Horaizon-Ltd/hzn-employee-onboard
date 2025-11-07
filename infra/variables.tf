locals {
  project = "employee_onborading"
}

variable "region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "eu-west-1"
}

variable "fallback_region" {
  description = "The region of the resources"
  type        = string
  default     = "eu-north-1"
}

variable "environment" {
  type        = string
  description = "Environment (dev / test / prod)"
}


variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "employee-onboarding"
}


variable "enable_function_url" {
  description = "Enable Lambda Function URL for direct HTTP access"
  type        = bool
  default     = true
}

variable "cors_allow_origins" {
  description = "Allowed origins for CORS"
  type        = list(string)
  default     = ["*"]
}

variable "cors_allow_methods" {
  description = "Allowed methods for CORS"
  type        = list(string)
  default     = ["*"]
}

variable "cors_allow_headers" {
  description = "Allowed headers for CORS"
  type        = list(string)
  default     = ["Content-Type", "Authorization"]
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 7
}

variable "ecr_image_tag" {
  description = "ECR image tag to deploy"
  type        = string
  default     = "latest"
}

variable "lambda_timeout" {
    description = "lambda timeout"
    type = string
    default = "600"  # 10 minutes - increased for PDF OCR processing
}
