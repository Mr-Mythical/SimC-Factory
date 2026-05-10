# ──────────────────────────────────────────────────────────────────────────────
# Dashboard service user: a dedicated IAM user for unattended automation.
#
# This user has the minimum permissions needed by the local dashboard to:
# - Submit and monitor Batch jobs (sim_orchestrator_batch.py)
# - Upload training data and download models from S3
# - Launch and describe SageMaker training/tuning jobs
#
# After `terraform apply`, create an access key in the AWS console (or CLI)
# for this user and put it in the project's .env file.
# ──────────────────────────────────────────────────────────────────────────────

resource "aws_iam_user" "dashboard" {
  name = "${local.name_prefix}-dashboard"
  tags = local.tags
}

resource "aws_iam_user_policy" "dashboard" {
  name = "${local.name_prefix}-dashboard-policy"
  user = aws_iam_user.dashboard.name

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      # S3: upload training data, download models, list bucket
      {
        Sid    = "S3Access",
        Effect = "Allow",
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ],
        Resource = [
          aws_s3_bucket.temp.arn,
          "${aws_s3_bucket.temp.arn}/*"
        ]
      },
      # Batch: submit jobs, describe jobs/queues/definitions
      {
        Sid    = "BatchAccess",
        Effect = "Allow",
        Action = [
          "batch:SubmitJob",
          "batch:DescribeJobs",
          "batch:ListJobs",
          "batch:TerminateJob",
          "batch:CancelJob",
          "batch:DescribeJobQueues",
          "batch:DescribeJobDefinitions",
          "batch:DescribeComputeEnvironments"
        ],
        Resource = "*"
      },
      # SageMaker: launch and monitor training/tuning jobs
      {
        Sid    = "SageMakerAccess",
        Effect = "Allow",
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:ListTrainingJobs",
          "sagemaker:StopTrainingJob",
          "sagemaker:CreateHyperParameterTuningJob",
          "sagemaker:DescribeHyperParameterTuningJob",
          "sagemaker:ListHyperParameterTuningJobs",
          "sagemaker:StopHyperParameterTuningJob",
          "sagemaker:ListTrainingJobsForHyperParameterTuningJob"
        ],
        Resource = "*"
      },
      # IAM: SageMaker needs to pass the execution role
      {
        Sid    = "PassSageMakerRole",
        Effect = "Allow",
        Action = "iam:PassRole",
        Resource = aws_iam_role.sagemaker_execution_role.arn,
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "sagemaker.amazonaws.com"
          }
        }
      },
      # STS: identity check from dashboard
      {
        Sid      = "IdentityCheck",
        Effect   = "Allow",
        Action   = "sts:GetCallerIdentity",
        Resource = "*"
      }
    ]
  })
}
