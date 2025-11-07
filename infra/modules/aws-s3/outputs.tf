output "id" {
  sensitive = true
  value     = aws_s3_bucket.bucket.id
}

output "bucket_name" {
  value = aws_s3_bucket.bucket.bucket_domain_name
}

output "bucket_arn" {
  value = aws_s3_bucket.bucket.arn
}

output "bucket_acl" {
  value = aws_s3_bucket_acl.bucket_acl.acl
}

output "bucket" {
  value = aws_s3_bucket.bucket.bucket
  sensitive = true
}

output "kms_key_arn" {
  value       = aws_kms_key.mykey.arn
  description = "The ARN of the KMS key used for bucket encryption."
  sensitive = true
}