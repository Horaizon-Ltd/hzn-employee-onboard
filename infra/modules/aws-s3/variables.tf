variable "project" {
  type = string
}

variable "environment" {
  type = string
}

variable "name" {
  type = string
}

variable "acl" {
  type    = string
  default = "private"
}

variable "bucket_block" {
  type = map(bool)
  default = {
    block_public_acls       = true
    block_public_policy     = true
    ignore_public_acls      = true
    restrict_public_buckets = true
  }
}