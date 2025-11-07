variable "log_group_retention_days" {
  description = "Retention period for the CloudWatch log group in days"
  type        = number
  default     = 30
}

variable "iam_role_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "log_group_name" {
  type = string
}

variable "resource_name" {
  type = string
}

variable "project" {
  type = string
}