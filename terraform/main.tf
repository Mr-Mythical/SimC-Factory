terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 6.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  name_prefix = var.name_prefix
  tags = merge(var.tags, {
    Project   = local.name_prefix
    ManagedBy = "Terraform"
    Workload  = "simulationcraft-batch-array"
  })
}

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "temp" {
  bucket = var.s3_bucket_name
  tags   = local.tags
}

resource "aws_s3_bucket_public_access_block" "temp" {
  bucket                  = aws_s3_bucket.temp.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "temp" {
  bucket = aws_s3_bucket.temp.id

  versioning_configuration {
    status = "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "temp" {
  bucket = aws_s3_bucket.temp.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "temp" {
  bucket = aws_s3_bucket.temp.id

  rule {
    id     = "cleanup-orphaned-temp-objects"
    status = "Enabled"

    expiration {
      days = var.temp_object_expiration_days
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }

    filter {
      prefix = ""
    }
  }
}

resource "aws_cloudwatch_log_group" "batch" {
  name              = "/aws/batch/job/${local.name_prefix}"
  retention_in_days = var.log_retention_days
  tags              = local.tags
}

resource "aws_iam_service_linked_role" "batch" {
  aws_service_name = "batch.amazonaws.com"
  description      = "Service-linked role for AWS Batch"
}

resource "aws_iam_role" "ecs_instance_role" {
  name = "${local.name_prefix}-ecs-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "ec2.amazonaws.com" },
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "ecs_instance_ecs" {
  role       = aws_iam_role.ecs_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_role_policy_attachment" "ecs_instance_ecr_readonly" {
  role       = aws_iam_role.ecs_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "ecs_instance_ssm" {
  role       = aws_iam_role.ecs_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ecs_instance_profile" {
  name = "${local.name_prefix}-ecs-instance-profile"
  role = aws_iam_role.ecs_instance_role.name
  tags = local.tags
}

resource "aws_iam_role" "spot_fleet_role" {
  name = "${local.name_prefix}-spot-fleet-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "spotfleet.amazonaws.com" },
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "spot_fleet_tagging" {
  role       = aws_iam_role.spot_fleet_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
}

resource "aws_iam_role" "execution_role" {
  name = "${local.name_prefix}-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "ecs-tasks.amazonaws.com" },
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "execution_ecs_task" {
  role       = aws_iam_role.execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "job_role" {
  name = "${local.name_prefix}-job-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "ecs-tasks.amazonaws.com" },
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy" "job_s3_access" {
  name = "${local.name_prefix}-job-s3-access"
  role = aws_iam_role.job_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ],
        Resource = "${aws_s3_bucket.temp.arn}/*"
      }
    ]
  })
}

resource "aws_batch_compute_environment" "spot" {
  name         = "${local.name_prefix}-spot-ce"
  type         = "MANAGED"
  service_role = aws_iam_service_linked_role.batch.arn
  state        = "ENABLED"
  tags         = local.tags

  compute_resources {
    type                = var.compute_type
    allocation_strategy = var.compute_type == "SPOT" ? var.spot_allocation_strategy : var.on_demand_allocation_strategy
    min_vcpus           = var.min_vcpus
    max_vcpus           = var.max_vcpus
    # Leave desired_vcpus unset so AWS Batch owns scaling.
    subnets             = var.subnet_ids
    security_group_ids  = var.security_group_ids
    instance_role       = aws_iam_instance_profile.ecs_instance_profile.arn
    instance_type       = var.instance_types
    ec2_key_pair        = var.ec2_key_pair
    spot_iam_fleet_role = var.compute_type == "SPOT" ? aws_iam_role.spot_fleet_role.arn : null

    tags = merge(local.tags, {
      Name = "${local.name_prefix}-batch-worker"
    })
  }

  update_policy {
    terminate_jobs_on_update      = false
    job_execution_timeout_minutes = 30
  }

  lifecycle {
    ignore_changes = [compute_resources[0].desired_vcpus]
  }

  depends_on = [
    aws_iam_role_policy_attachment.ecs_instance_ecs,
    aws_iam_role_policy_attachment.ecs_instance_ecr_readonly,
    aws_iam_role_policy_attachment.ecs_instance_ssm,
    aws_iam_role_policy_attachment.execution_ecs_task,
    aws_iam_role_policy_attachment.spot_fleet_tagging,
  ]
}

resource "aws_batch_job_queue" "main" {
  name     = "${local.name_prefix}-queue"
  state    = "ENABLED"
  priority = 1
  tags     = local.tags

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.spot.arn
  }
}

resource "aws_batch_compute_environment" "staleness" {
  name         = "${local.name_prefix}-staleness-ce"
  type         = "MANAGED"
  service_role = aws_iam_service_linked_role.batch.arn
  state        = "ENABLED"
  tags         = local.tags

  compute_resources {
    type                = var.staleness_compute_type
    allocation_strategy = var.staleness_compute_type == "SPOT" ? var.spot_allocation_strategy : var.on_demand_allocation_strategy
    min_vcpus           = var.staleness_min_vcpus
    max_vcpus           = var.staleness_max_vcpus
    subnets             = var.subnet_ids
    security_group_ids  = var.security_group_ids
    instance_role       = aws_iam_instance_profile.ecs_instance_profile.arn
    instance_type       = var.staleness_instance_types
    ec2_key_pair        = var.ec2_key_pair
    spot_iam_fleet_role = var.staleness_compute_type == "SPOT" ? aws_iam_role.spot_fleet_role.arn : null

    tags = merge(local.tags, {
      Name = "${local.name_prefix}-batch-staleness-worker"
    })
  }

  update_policy {
    terminate_jobs_on_update      = false
    job_execution_timeout_minutes = 30
  }

  lifecycle {
    ignore_changes = [compute_resources[0].desired_vcpus]
  }

  depends_on = [
    aws_iam_role_policy_attachment.ecs_instance_ecs,
    aws_iam_role_policy_attachment.ecs_instance_ecr_readonly,
    aws_iam_role_policy_attachment.ecs_instance_ssm,
    aws_iam_role_policy_attachment.execution_ecs_task,
    aws_iam_role_policy_attachment.spot_fleet_tagging,
  ]
}

resource "aws_batch_job_queue" "staleness" {
  name     = "${local.name_prefix}-staleness-queue"
  state    = "ENABLED"
  priority = 10
  tags     = local.tags

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.staleness.arn
  }
}

resource "aws_batch_job_definition" "worker" {
  name                  = "${local.name_prefix}-worker"
  type                  = "container"
  platform_capabilities = ["EC2"]
  tags                  = local.tags

  retry_strategy {
    attempts = 1
  }

  timeout {
    attempt_duration_seconds = var.job_attempt_timeout_seconds
  }

  container_properties = jsonencode({
    image            = var.worker_image
    executionRoleArn = aws_iam_role.execution_role.arn
    jobRoleArn       = aws_iam_role.job_role.arn
    resourceRequirements = [
      { type = "VCPU", value = tostring(var.job_vcpus) },
      { type = "MEMORY", value = tostring(var.job_memory) }
    ]
    environment = [
      { name = "SIMC_WORKER_PARALLEL", value = tostring(var.worker_parallel_default) }
    ]
    linuxParameters = {
      initProcessEnabled = true
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.batch.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "simc-array-worker"
      }
    }
  })
}

resource "aws_batch_job_definition" "staleness_worker" {
  name                  = "${local.name_prefix}-staleness-worker"
  type                  = "container"
  platform_capabilities = ["EC2"]
  tags                  = local.tags

  retry_strategy {
    attempts = 1
  }

  timeout {
    attempt_duration_seconds = var.staleness_job_attempt_timeout_seconds
  }

  container_properties = jsonencode({
    image            = var.worker_image
    executionRoleArn = aws_iam_role.execution_role.arn
    jobRoleArn       = aws_iam_role.job_role.arn
    resourceRequirements = [
      { type = "VCPU", value = tostring(var.staleness_job_vcpus) },
      { type = "MEMORY", value = tostring(var.staleness_job_memory) }
    ]
    environment = [
      { name = "SIMC_WORKER_PARALLEL", value = tostring(var.staleness_worker_parallel_default) }
    ]
    linuxParameters = {
      initProcessEnabled = true
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.batch.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "simc-staleness-worker"
      }
    }
  })
}
