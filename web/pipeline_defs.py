"""Pipeline definitions — daily refresh pipeline and registry."""

import re

from web.job_manager import Job
from web.pipeline_engine import PipelineDef, PipelineRun, PipelineStepDef

# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def _parse_staleness_output(job: Job, run: PipelineRun) -> None:
    """Scan staleness check output and set run.context['has_stale']."""
    for line in job.log_lines:
        if "stale profile" in line.lower() and "detected" in line.lower():
            # Line like: "Detected 3 stale profile(s):"
            match = re.search(r"Detected\s+(\d+)\s+stale", line)
            if match and int(match.group(1)) > 0:
                run.context["has_stale"] = True
                return
    run.context["has_stale"] = False


def _parse_model_artifact(job: Job, run: PipelineRun) -> None:
    """Extract model S3 URI from training job output."""
    for line in job.log_lines:
        if "Model artifact:" in line or "model artifact:" in line.lower():
            # Line like: "  Model artifact: s3://bucket/path/model.tar.gz"
            match = re.search(r"(s3://\S+)", line)
            if match:
                run.context["model_s3_uri"] = match.group(1)
                return


def _download_model_dynamic_args(run: PipelineRun) -> dict:
    """Provide --s3-uri from training step's output."""
    s3_uri = run.context.get("model_s3_uri", "")
    if s3_uri:
        return {"--s3-uri": s3_uri}
    return {}


# ---------------------------------------------------------------------------
# Daily Refresh Pipeline
# ---------------------------------------------------------------------------

DAILY_PIPELINE = PipelineDef(
    id="daily_refresh",
    name="Daily Refresh Pipeline",
    description=(
        "Rebuild the SimC worker image (keeps binary in sync with profiles), "
        "staleness check, regenerate stale profiles if needed, "
        "then boost worst specs, full retrain, and download the updated model."
    ),
    schedule_cron="0 3 * * *",  # 3:00 AM daily
    steps=[
        PipelineStepDef(
            id="rebuild_ecr",
            preset_id="rebuild_ecr_image",
            name="Rebuild ECR Image",
        ),
        PipelineStepDef(
            id="refresh_profiles",
            preset_id="refresh_profiles",
            name="Refresh Profiles",
        ),
        PipelineStepDef(
            id="staleness_check",
            preset_id="quick_staleness_check",
            name="Check Staleness",
            arg_overrides={"--skip-profile-refresh": True},
            output_parser=_parse_staleness_output,
        ),
        PipelineStepDef(
            id="regenerate_stale",
            preset_id="regenerate_stale",
            name="Regenerate Stale Profiles",
            arg_overrides={"--skip-profile-refresh": True},
            condition=lambda run: run.context.get("has_stale", False),
        ),
        PipelineStepDef(
            id="boost_worst",
            preset_id="boost_worst_specs",
            name="Boost Worst Specs",
            arg_overrides={"--trigger-training": False, "--skip-profile-refresh": True},
        ),
        PipelineStepDef(
            id="retrain",
            preset_id="sagemaker_train",
            name="SageMaker Retrain",
            output_parser=_parse_model_artifact,
        ),
        PipelineStepDef(
            id="download_model",
            preset_id="download_model",
            name="Download Model",
            dynamic_args=_download_model_dynamic_args,
        ),
    ],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PIPELINES: dict[str, PipelineDef] = {
    DAILY_PIPELINE.id: DAILY_PIPELINE,
}
