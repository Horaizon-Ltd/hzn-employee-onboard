data "aws_iam_policy_document" "assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}

data "aws_ecr_repository" "lambda_repo" {
  name = var.aws_ecr_repository_name
}

data "aws_ecr_image" "latest_image" {
  repository_name = var.aws_ecr_repository_name
  most_recent     = true
}

resource "aws_iam_role" "lambda_role" {
  name               = "lambda-${var.function_name}-role-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json

}

resource "aws_iam_role_policy_attachment" "additional_policy" {
  count      = length(var.additional_policies)
  policy_arn = var.additional_policies[count.index]
  role       = aws_iam_role.lambda_role.name
}

resource "aws_lambda_function" "lambda" {
  function_name = "${var.function_name}-${var.environment}"
  role          = aws_iam_role.lambda_role.arn
  image_uri     = data.aws_ecr_image.latest_image.image_uri
  package_type  = "Image"
  timeout       = var.timeout
  memory_size   = 3008
  publish       = true
  architectures = ["x86_64"]
  layers        = var.layers

  # lifecycle {
  #   # Prevents unnecessary redeployments when using latest tag
  #   ignore_changes = [image_uri]
  # }
  environment {
    variables = var.environment_variables
  }
}

# resource "aws_lambda_provisioned_concurrency_config" "provisioned_concurrent_executions" {
#   function_name                     = aws_lambda_function.lambda.function_name
#   qualifier                         = aws_lambda_function.lambda.version
#   provisioned_concurrent_executions = 5
# }

resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "LambdaHighErrorRate"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 60
  statistic           = "Sum"
  threshold           = 5
}


module "aws_cloudwatch" {
  source                 = "../aws-log"
  iam_role_name          = aws_iam_role.lambda_role.name
  log_group_name         = "/aws/lambda/${aws_lambda_function.lambda.function_name}"
  environment            = var.environment
  resource_name          = aws_lambda_function.lambda.function_name
  project                = var.project
}