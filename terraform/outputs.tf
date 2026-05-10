output "s3_bucket_name" {
  value = aws_s3_bucket.temp.bucket
}

output "batch_job_queue_name" {
  value = aws_batch_job_queue.main.name
}

output "batch_job_definition_name" {
  value = aws_batch_job_definition.worker.name
}

output "batch_compute_environment_name" {
  value = aws_batch_compute_environment.spot.name
}

output "cloudwatch_log_group" {
  value = aws_cloudwatch_log_group.batch.name
}

output "staleness_batch_job_queue_name" {
  value = aws_batch_job_queue.staleness.name
}

output "staleness_batch_job_definition_name" {
  value = aws_batch_job_definition.staleness_worker.name
}

output "staleness_batch_compute_environment_name" {
  value = aws_batch_compute_environment.staleness.name
}

# ── SageMaker outputs ────────────────────────────────────────────────────────

output "sagemaker_execution_role_arn" {
  description = "IAM role ARN for SageMaker training and tuning jobs."
  value       = aws_iam_role.sagemaker_execution_role.arn
}

# ── Dashboard Roles Anywhere ────────────────────────────────────────────────

output "dashboard_trust_anchor_arn" {
  description = "IAM Roles Anywhere trust anchor ARN."
  value       = aws_rolesanywhere_trust_anchor.dashboard.arn
}

output "dashboard_profile_arn" {
  description = "IAM Roles Anywhere profile ARN."
  value       = aws_rolesanywhere_profile.dashboard.arn
}

output "dashboard_role_arn" {
  description = "IAM role ARN the dashboard assumes via Roles Anywhere."
  value       = aws_iam_role.dashboard.arn
}
