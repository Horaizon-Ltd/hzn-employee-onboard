resource "aws_cloudwatch_log_group" "log_group" {
  name              = var.log_group_name
  retention_in_days = var.log_group_retention_days
}

data "aws_iam_policy_document" "cloudwatch" {
  statement {
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:DescribeLogGroups",
      "logs:DescribeLogStreams",
      "logs:PutLogEvents",
      "logs:GetLogEvents",
      "logs:FilterLogEvents"
    ]
    resources = ["arn:aws:logs:*:*:*"]
  }
}

resource "aws_iam_policy" "cloudwatch_policy" {
  name        = "cloudwatch-logging-policy-${var.resource_name}-${var.environment}"
  description = "IAM policy for logging to CloudWatch in ${var.environment}"
  policy      = data.aws_iam_policy_document.cloudwatch.json
}


resource "aws_iam_role_policy_attachment" "lambda_logs_attachment" {
  role       = var.iam_role_name
  policy_arn = aws_iam_policy.cloudwatch_policy.arn
}
