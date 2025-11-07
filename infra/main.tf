data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

module "ecr_repo" {
  source = "./modules/aws-ecr"
  name = "${var.project_name}-generate-output"
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
    ENVIRONMENT = var.environment
    LOG_LEVEL   = "INFO"
  }

  additional_policies = []

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
