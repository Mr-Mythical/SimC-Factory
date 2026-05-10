# ──────────────────────────────────────────────────────────────────────────────
# SageMaker resources: IAM role for training jobs and hyperparameter tuning.
#
# Training jobs and tuning jobs are ephemeral — launched via the Python SDK
# (sagemaker/launch_training.py, sagemaker/launch_tuning.py). Only the
# persistent IAM execution role is managed here.
#
# S3 storage reuses the existing aws_s3_bucket.temp bucket under the
# "sagemaker/" key prefix.
# ──────────────────────────────────────────────────────────────────────────────

resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${local.name_prefix}-sagemaker-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "sagemaker.amazonaws.com" },
      Action    = [
        "sts:AssumeRole",
        "sts:SetSourceIdentity"
      ]
    }]
  })

  tags = local.tags
}

# S3 access: read training data, write model artifacts and checkpoints
resource "aws_iam_role_policy" "sagemaker_s3_access" {
  name = "${local.name_prefix}-sagemaker-s3-access"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ],
        Resource = [
          aws_s3_bucket.temp.arn,
          "${aws_s3_bucket.temp.arn}/sagemaker/*"
        ]
      }
    ]
  })
}

# CloudWatch Logs: SageMaker writes training logs here
resource "aws_iam_role_policy" "sagemaker_cloudwatch" {
  name = "${local.name_prefix}-sagemaker-cloudwatch"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ],
        Resource = "arn:aws:logs:${var.aws_region}:*:log-group:/aws/sagemaker/*"
      }
    ]
  })
}

# ECR access: SageMaker pulls the PyTorch DLC (Deep Learning Container) image
resource "aws_iam_role_policy" "sagemaker_ecr" {
  name = "${local.name_prefix}-sagemaker-ecr"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetAuthorizationToken"
        ],
        Resource = "*"
      }
    ]
  })
}

# SageMaker managed spot training: permission to create spot requests
resource "aws_iam_role_policy" "sagemaker_spot" {
  name = "${local.name_prefix}-sagemaker-spot"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "ec2:CreateNetworkInterface",
          "ec2:DeleteNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DescribeVpcs",
          "ec2:DescribeSubnets",
          "ec2:DescribeSecurityGroups"
        ],
        Resource = "*"
      }
    ]
  })
}
