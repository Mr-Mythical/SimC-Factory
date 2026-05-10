variable "aws_region" {
  type        = string
  description = "AWS region for Batch, S3, and related resources."
}

variable "name_prefix" {
  type        = string
  description = "Name prefix for resources."
  default     = "simc-batch-array"
}

variable "tags" {
  type        = map(string)
  description = "Additional tags to apply to resources."
  default     = {}
}

variable "subnet_ids" {
  type        = list(string)
  description = "Subnet IDs for the Batch compute environment."
}

variable "security_group_ids" {
  type        = list(string)
  description = "Security group IDs for the Batch compute environment."
}

variable "ec2_key_pair" {
  type        = string
  description = "Optional EC2 key pair for debugging Batch worker instances."
  default     = null
}

variable "compute_type" {
  type        = string
  description = "Batch compute environment type: SPOT or EC2."
  default     = "SPOT"
  validation {
    condition     = contains(["SPOT", "EC2"], var.compute_type)
    error_message = "compute_type must be SPOT or EC2."
  }
}

variable "spot_allocation_strategy" {
  type        = string
  description = "Spot allocation strategy for the managed compute environment."
  default     = "SPOT_PRICE_CAPACITY_OPTIMIZED"
}

variable "on_demand_allocation_strategy" {
  type        = string
  description = "On-demand allocation strategy."
  default     = "BEST_FIT_PROGRESSIVE"
}

variable "instance_types" {
  type        = list(string)
  description = "Allowed EC2 instance types for the Batch compute environment."
  default     = ["c7i-flex.large", "m7i-flex.large"]
}

variable "min_vcpus" {
  type        = number
  description = "Minimum vCPUs for the Batch compute environment."
  default     = 0
}

variable "max_vcpus" {
  type        = number
  description = "Maximum vCPUs for the Batch compute environment."
  default     = 32
}

variable "worker_image" {
  type        = string
  description = "Container image for the lightweight Batch worker."
}

variable "job_vcpus" {
  type        = number
  description = "vCPU requirement per Batch worker job."
  default     = 1
}

variable "job_memory" {
  type        = number
  description = "Memory requirement (MiB) per Batch worker job."
  default     = 1024
}

variable "worker_parallel_default" {
  type        = number
  description = "Default parallel SimulationCraft processes inside one worker container."
  default     = 2
}

variable "job_attempt_timeout_seconds" {
  type        = number
  description = "Batch job attempt timeout in seconds."
  default     = 7200
}

variable "s3_bucket_name" {
  type        = string
  description = "Temporary S3 bucket for chunk input/output transport."
}

variable "temp_object_expiration_days" {
  type        = number
  description = "Safety cleanup for orphaned temporary S3 objects."
  default     = 2
}

variable "log_retention_days" {
  type        = number
  description = "CloudWatch log retention for Batch worker logs."
  default     = 14
}

variable "staleness_compute_type" {
  type        = string
  description = "Compute environment type for dedicated staleness checks: SPOT or EC2."
  default     = "EC2"
  validation {
    condition     = contains(["SPOT", "EC2"], var.staleness_compute_type)
    error_message = "staleness_compute_type must be SPOT or EC2."
  }
}

variable "staleness_instance_types" {
  type        = list(string)
  description = "Allowed EC2 instance types for dedicated staleness checks."
  default     = ["c7i-flex.large"]
}

variable "staleness_min_vcpus" {
  type        = number
  description = "Minimum vCPUs for staleness check compute environment."
  default     = 0
}

variable "staleness_max_vcpus" {
  type        = number
  description = "Maximum vCPUs for staleness check compute environment (set to a single instance capacity)."
  default     = 2
}

variable "staleness_job_vcpus" {
  type        = number
  description = "vCPU requirement per staleness check worker job."
  default     = 1
}

variable "staleness_job_memory" {
  type        = number
  description = "Memory requirement (MiB) per staleness check worker job."
  default     = 2048
}

variable "staleness_worker_parallel_default" {
  type        = number
  description = "Default parallel SimulationCraft processes inside staleness check worker container."
  default     = 1
}

variable "staleness_job_attempt_timeout_seconds" {
  type        = number
  description = "Batch staleness check job attempt timeout in seconds."
  default     = 7200
}

# ── SageMaker training variables ─────────────────────────────────────────────

variable "sagemaker_instance_type" {
  type        = string
  description = "SageMaker training instance type."
  default     = "ml.g4dn.xlarge"
}

variable "sagemaker_max_run_seconds" {
  type        = number
  description = "Maximum training time per SageMaker job in seconds."
  default     = 3600
}

variable "sagemaker_max_wait_seconds" {
  type        = number
  description = "Maximum total wait time for spot training (includes queue time)."
  default     = 7200
}

variable "sagemaker_tuning_max_jobs" {
  type        = number
  description = "Maximum number of AMT hyperparameter tuning trials."
  default     = 50
}

variable "sagemaker_tuning_max_parallel" {
  type        = number
  description = "Maximum parallel training jobs during AMT tuning."
  default     = 3
}

# ── Dashboard IAM Roles Anywhere ────────────────────────────────────────────

variable "dashboard_ca_certificate_pem" {
  type        = string
  description = "PEM-encoded CA certificate for IAM Roles Anywhere trust anchor. Generate with: python scripts/generate_certs.py"
  default     = ""
}

locals {
  # Auto-read from certs/ca.pem if the variable isn't set explicitly
  dashboard_ca_cert = var.dashboard_ca_certificate_pem != "" ? var.dashboard_ca_certificate_pem : (
    fileexists("${path.module}/../certs/ca.pem") ? file("${path.module}/../certs/ca.pem") : ""
  )
}
