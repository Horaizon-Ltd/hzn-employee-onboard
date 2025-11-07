locals {
  function_dir = replace(var.function_name, "-", "_")
  project      = "hzn-rag-chatbot"
}

variable "source_code_path" {
  type = string
}

variable "function_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "environment_variables" {
  description = "The environment variables for the function"
  type        = map(string)
  sensitive   = true
}

variable "region" {
  type = string
}


variable "aws_ecr_repository_name" {
  type = string
}

variable "additional_policies" {
  description = "List of additional IAM policy ARNs to attach to the Lambda role"
  type        = list(string)
  default     = [] # Default to an empty list if not provided
}

variable "layers" {
  description = "List of Lambda Layer ARNs to attach to the function"
  type        = list(string)
  default     = []
}

variable "timeout" {
  type = string
}

variable "project" {
  type = string
}