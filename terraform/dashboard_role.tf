# ──────────────────────────────────────────────────────────────────────────────
# IAM Roles Anywhere: certificate-based auth for the local dashboard.
#
# This is AWS's recommended approach for servers outside AWS. Instead of
# static access keys, the dashboard uses an X.509 certificate to obtain
# short-lived temporary credentials that auto-rotate.
#
# Setup:
#   1. Run `python scripts/generate_certs.py` to create a self-signed CA
#      and client certificate.
#   2. `terraform apply` — registers the CA as a trust anchor.
#   3. Install aws_signing_helper:
#      https://docs.aws.amazon.com/rolesanywhere/latest/userguide/credential-helper.html
#   4. Configure .env with paths to cert, key, and signing helper.
# ──────────────────────────────────────────────────────────────────────────────

# Trust anchor: tells AWS to trust certificates signed by our CA
resource "aws_rolesanywhere_trust_anchor" "dashboard" {
  name    = "${local.name_prefix}-dashboard-ca"
  enabled = true

  source {
    source_type = "CERTIFICATE_BUNDLE"
    source_data {
      x509_certificate_data = local.dashboard_ca_cert
    }
  }

  tags = local.tags
}

# IAM role the dashboard assumes via Roles Anywhere
resource "aws_iam_role" "dashboard" {
  name = "${local.name_prefix}-dashboard"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "rolesanywhere.amazonaws.com" },
      Action = [
        "sts:AssumeRole",
        "sts:TagSession",
        "sts:SetSourceIdentity"
      ],
      Condition = {
        ArnEquals = {
          "aws:SourceArn" = aws_rolesanywhere_trust_anchor.dashboard.arn
        }
      }
    }]
  })

  tags = local.tags
}

# Roles Anywhere profile: links the trust anchor to the IAM role
resource "aws_rolesanywhere_profile" "dashboard" {
  name      = "${local.name_prefix}-dashboard"
  enabled   = true
  role_arns = [aws_iam_role.dashboard.arn]

  # Session duration: 1 hour (signing helper auto-refreshes before expiry)
  duration_seconds = 3600

  tags = local.tags
}

# ── Permissions ──────────────────────────────────────────────────────────────

resource "aws_iam_role_policy" "dashboard" {
  name = "${local.name_prefix}-dashboard-policy"
  role = aws_iam_role.dashboard.id

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
          "sagemaker:ListTrainingJobsForHyperParameterTuningJob",
          "sagemaker:AddTags"
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
      # ECR: authenticate and push rebuilt worker images
      {
        Sid    = "ECRAuth",
        Effect = "Allow",
        Action = "ecr:GetAuthorizationToken",
        Resource = "*"
      },
      {
        Sid    = "ECRPush",
        Effect = "Allow",
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ],
        Resource = "arn:aws:ecr:${var.aws_region}:${data.aws_caller_identity.current.account_id}:repository/simc-batch-worker"
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
