output "invoke_arn" {
  sensitive = true
  value     = aws_lambda_function.lambda.invoke_arn
}

output "arn" {
  sensitive = true
  value     = aws_lambda_function.lambda.arn
}

output "function_name" {
  value = aws_lambda_function.lambda.function_name
}