"""Project paths, AWS defaults, and preset configurations."""

import os
import sys
from dataclasses import dataclass, field

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WEB_DIR)
LOCAL_DIR = os.path.join(PROJECT_ROOT, "local")
SAGEMAKER_DIR = os.path.join(PROJECT_ROOT, "sagemaker")

# Load .env early so env vars are available for constants below
_ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(_ENV_FILE):
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _val = _line.partition("=")
            _key = _key.strip()
            _val = _val.strip().strip("\"'")
            if _key and _key not in os.environ:
                os.environ[_key] = _val

TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
SPEC_PROFILES_DIR = os.path.join(PROJECT_ROOT, "spec_profiles")
MODEL_DIR = os.path.join(LOCAL_DIR, "nn_website_model", "deep_nn")
SNAPSHOTS_DIR = os.path.join(LOCAL_DIR, "nn_website_model", "snapshots")
PLOTS_DIR = os.path.join(LOCAL_DIR, "nn_website_model", "evaluation_plots")
DRIFT_HISTORY_FILE = os.path.join(LOCAL_DIR, "nn_website_model", "drift_history.jsonl")
DRIFT_PLOT_FILE = os.path.join(PLOTS_DIR, "drift_over_time.png")
METADATA_FILE = os.path.join(MODEL_DIR, "nn_metadata.json")
PROFILE_METADATA_FILE = os.path.join(PROJECT_ROOT, "profile_metadata.json")
ALL_SPECS_CSV = os.path.join(PROJECT_ROOT, "all_specs_training_data.csv")
BEST_HYPERPARAMS_FILE = os.path.join(SAGEMAKER_DIR, "best_hyperparameters.json")
ENSEMBLE_REPORT_FILE = os.path.join(MODEL_DIR, "minimal_ensemble_report.json")

TERRAFORM_DIR = os.path.join(PROJECT_ROOT, "terraform")

# Default target samples per spec (matches DEFAULT_SAMPLES_COUNT in sim_orchestrator_batch.py)
TARGET_SAMPLES_PER_SPEC = 500

PYTHON_EXE = sys.executable

# AWS defaults
AWS_REGION = "eu-north-1"
S3_BUCKET = "simc-batch-bucket"
BATCH_JOB_QUEUE = "simc-batch-array-queue"
BATCH_JOB_DEFINITION = "simc-batch-array-worker"
SAGEMAKER_EXECUTION_ROLE_ARN = os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN", "")


@dataclass
class FormField:
    arg_name: str
    label: str
    field_type: str  # "int", "float", "str", "bool", "select"
    default: object = None
    help_text: str = ""
    options: list[str] = field(default_factory=list)
    required: bool = False


@dataclass
class PresetConfig:
    id: str
    name: str
    description: str
    category: str  # "simulation", "training", "sagemaker", "infrastructure"
    script: str  # relative to PROJECT_ROOT, or binary name for shell commands
    default_args: dict = field(default_factory=dict)
    form_fields: list[FormField] = field(default_factory=list)
    is_shell: bool = False  # True = run script as a raw shell command, not via PYTHON_EXE
    cwd: str | None = None  # override working directory (e.g. terraform/ dir)
    subcommand: str = ""  # explicit subcommand(s) for shell presets (e.g. "sso login")


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

PRESETS: dict[str, PresetConfig] = {}


def _register(p: PresetConfig) -> None:
    PRESETS[p.id] = p


# ── Simulation presets ─────────────────────────────────────────────────────

_register(PresetConfig(
    id="quick_staleness_check",
    name="Staleness Check",
    description="Re-sim a sample of points per spec to detect DPS drift.",
    category="simulation",
    script="local/sim_orchestrator_batch.py",
    default_args={"--check-staleness": True},
    form_fields=[
        FormField("--staleness-drift-threshold-pct", "Drift Threshold %", "float", 2.0,
                  "DPS drift percentage to flag as stale."),
        FormField("--staleness-sample-size", "Sample Size", "int", 10,
                  "How many existing points to re-sim per profile."),
        FormField("--specs", "Specs", "str", "existing",
                  "Which specs to check: all, existing, or comma-separated names."),
    ],
))

_register(PresetConfig(
    id="regenerate_stale",
    name="Regenerate Stale Profiles",
    description="Detect stale profiles via drift check, then re-simulate them.",
    category="simulation",
    script="local/sim_orchestrator_batch.py",
    default_args={"--regenerate-stale": True},
    form_fields=[
        FormField("--staleness-drift-threshold-pct", "Drift Threshold %", "float", 2.0,
                  "DPS drift percentage to flag as stale."),
        FormField("--max-active-profiles", "Max Active Profiles", "int", 4,
                  "Concurrent specs in-flight."),
        FormField("--trigger-training", "Trigger Training After", "bool", False,
                  "Launch SageMaker training after regeneration finishes."),
        FormField("--training-s3-bucket", "Training S3 Bucket", "str", S3_BUCKET,
                  "S3 bucket for SageMaker (required if trigger-training is on)."),
        FormField("--training-role-arn", "Training Role ARN", "str", SAGEMAKER_EXECUTION_ROLE_ARN,
                  "SageMaker execution role ARN (required if trigger-training is on)."),
    ],
))

_register(PresetConfig(
    id="regenerate_stale_skip_check",
    name="Regenerate Stale (Skip Check)",
    description="Re-simulate profiles already marked stale — skips the drift check.",
    category="simulation",
    script="local/sim_orchestrator_batch.py",
    default_args={"--regenerate-stale": True, "--skip-staleness-check": True},
    form_fields=[
        FormField("--max-active-profiles", "Max Active Profiles", "int", 4,
                  "Concurrent specs in-flight."),
        FormField("--trigger-training", "Trigger Training After", "bool", False,
                  "Launch SageMaker training after regeneration finishes."),
        FormField("--training-s3-bucket", "Training S3 Bucket", "str", S3_BUCKET,
                  "S3 bucket for SageMaker (required if trigger-training is on)."),
        FormField("--training-role-arn", "Training Role ARN", "str", SAGEMAKER_EXECUTION_ROLE_ARN,
                  "SageMaker execution role ARN (required if trigger-training is on)."),
    ],
))

_register(PresetConfig(
    id="boost_worst_specs",
    name="Boost Worst Specs",
    description="Add up to 200 extra samples for specs with highest prediction error, proportional to MAE.",
    category="simulation",
    script="local/sim_orchestrator_batch.py",
    default_args={
        "--boost-from-model": os.path.join(MODEL_DIR, "nn_metadata.json"),
        "--training-role-arn": SAGEMAKER_EXECUTION_ROLE_ARN,
    },
    form_fields=[
        FormField("--samples", "Baseline Samples", "int", 500,
                  "Baseline sample count for all specs."),
        FormField("--boost-extra-budget", "Extra Samples (worst spec)", "int", 200,
                  "Max extra samples the worst spec gets. Others get a proportional share."),
        FormField("--max-active-profiles", "Max Active Profiles", "int", 4,
                  "Concurrent specs in-flight."),
        FormField("--trigger-training", "Trigger Training After", "bool", False,
                  "Launch SageMaker training after simulations finish."),
        FormField("--training-s3-bucket", "Training S3 Bucket", "str", S3_BUCKET,
                  "S3 bucket for SageMaker (required if trigger-training is on)."),
        FormField("--exclude-specs", "Exclude Specs", "str", "",
                  "Comma-separated spec names to skip (e.g. Druid_Balance)."),
    ],
))

_register(PresetConfig(
    id="full_sim_500",
    name="Full Simulation (500 samples)",
    description="Run all specs to 500 samples. Long-running.",
    category="simulation",
    script="local/sim_orchestrator_batch.py",
    default_args={"--specs": "all"},
    form_fields=[
        FormField("--samples", "Samples Per Spec", "int", 500,
                  "Target saved samples per spec."),
        FormField("--iterations", "SimC Iterations", "int", 5000,
                  "SimulationCraft iterations per profile."),
        FormField("--chunk-size", "Chunk Size", "int", 25,
                  "Sims per Batch array child."),
        FormField("--max-active-profiles", "Max Active Profiles", "int", 4,
                  "Concurrent specs in-flight."),
        FormField("--worker-parallel", "Worker Parallelism", "int", 5,
                  "Parallel SimC processes per worker container."),
        FormField("--exclude-specs", "Exclude Specs", "str", "",
                  "Comma-separated spec names to skip (e.g. Druid_Balance)."),
    ],
))

_register(PresetConfig(
    id="small_test_run",
    name="Small Test Run",
    description="Quick test: 50 samples, 1000 iterations, 1 profile at a time.",
    category="simulation",
    script="local/sim_orchestrator_batch.py",
    default_args={
        "--specs": "existing",
        "--samples": "50",
        "--iterations": "1000",
        "--max-active-profiles": "1",
    },
    form_fields=[
        FormField("--specs", "Specs", "str", "existing",
                  "Which specs: all, existing, or comma-separated."),
        FormField("--samples", "Samples", "int", 50,
                  "Target samples per spec."),
        FormField("--iterations", "SimC Iterations", "int", 1000,
                  "Iterations per profile."),
        FormField("--profile-limit", "Profile Limit", "int", 5,
                  "Max profiles to queue (leave 0 for no limit)."),
        FormField("--exclude-specs", "Exclude Specs", "str", "",
                  "Comma-separated spec names to skip (e.g. Druid_Balance)."),
    ],
))

_register(PresetConfig(
    id="wipe_all_data",
    name="Wipe All Data",
    description="Delete ALL training data, shards, profile metadata, and cached profiles. Full reset.",
    category="simulation",
    script="local/sim_orchestrator_batch.py",
    default_args={"--wipe-all-data": True},
    form_fields=[],
))

_register(PresetConfig(
    id="refresh_profiles",
    name="Refresh Profiles",
    description="Download all spec profiles from GitHub. Used as a shared step before simulation runs.",
    category="simulation",
    script="local/sim_orchestrator_batch.py",
    default_args={"--download-profiles": True},
    form_fields=[],
))

# ── Training presets ───────────────────────────────────────────────────────

_register(PresetConfig(
    id="local_ensemble_optimize",
    name="Local HPO (Small Model - Phase 2)",
    description="Run Optuna hyperparameter optimization locally with GPU. Narrowed search: 2-layer [180-280]×[150-250], epochs [900-1500], LR [8e-5,1.5e-4], dropout [0,0.05].",
    category="training",
    script="local/optimize_ensemble.py",
    default_args={},
    form_fields=[],
))

# ── SageMaker presets ──────────────────────────────────────────────────────

_register(PresetConfig(
    id="sagemaker_train",
    name="SageMaker Training",
    description="Launch SageMaker training. Runs N models with different seeds and keeps the best.",
    category="sagemaker",
    script="sagemaker/launch_training.py",
    default_args={"--role-arn": SAGEMAKER_EXECUTION_ROLE_ARN},
    form_fields=[
        FormField("--s3-bucket", "S3 Bucket", "str", S3_BUCKET,
                  "S3 bucket for training data and model output.", required=True),
        FormField("--num-runs", "Training Runs", "int", 3,
                  "Parallel runs with different seeds. Best test MAE wins."),
        FormField("--max-parallel", "Max Parallel Jobs", "int", 3,
                  "Maximum concurrent SageMaker training jobs."),
        FormField("--instance-type", "Instance Type", "select", "ml.g4dn.xlarge",
                  "SageMaker training instance.", ["ml.g4dn.xlarge", "ml.g5.xlarge", "ml.p3.2xlarge"]),
        FormField("--no-spot", "Use On-Demand", "bool", False,
                  "Use on-demand instances instead of spot (for debugging)."),
    ],
))

_register(PresetConfig(
    id="sagemaker_experiment",
    name="SageMaker Test-Size Experiment",
    description="Grid search across test-size values with multiple seeds. Finds the best train/test split and updates best_hyperparameters.json.",
    category="sagemaker",
    script="sagemaker/launch_experiment.py",
    default_args={"--role-arn": SAGEMAKER_EXECUTION_ROLE_ARN},
    form_fields=[
        FormField("--s3-bucket", "S3 Bucket", "str", S3_BUCKET,
                  "S3 bucket for training data.", required=True),
        FormField("--test-sizes", "Test Size Grid", "str", "0.10,0.15,0.20",
                  "Comma-separated test-size values to test."),
        FormField("--cv-folds", "CV Folds", "int", None,
                  "CV folds for evaluation (default: from hyperparams). Does not affect model training."),
        FormField("--seeds-per-combo", "Seeds Per Combo", "int", 2,
                  "Runs per test_size to measure variance."),
        FormField("--max-parallel", "Max Parallel Jobs", "int", 2,
                  "Maximum concurrent SageMaker training jobs."),
        FormField("--resume", "Resume Experiment", "str", "",
                  "Resume a previous experiment by its job prefix (e.g. dps-exp-20260420-153000). Leave blank for new."),
        FormField("--instance-type", "Instance Type", "select", "ml.g4dn.xlarge",
                  "SageMaker training instance.", ["ml.g4dn.xlarge", "ml.g5.xlarge", "ml.p3.2xlarge"]),
        FormField("--no-spot", "Use On-Demand", "bool", False,
                  "Use on-demand instances instead of spot."),
    ],
))

_register(PresetConfig(
    id="sagemaker_tune",
    name="SageMaker HPO Tuning (Small Model - Phase 2)",
    description="SageMaker Bayesian HPO for small, fast models. Narrowed search: 2-layer [180-280]×[150-250], epochs [900-1500], LR [8e-5,1.5e-4], dropout [0,0.05].",
    category="sagemaker",
    script="sagemaker/launch_tuning.py",
    default_args={"--role-arn": SAGEMAKER_EXECUTION_ROLE_ARN},
    form_fields=[
        FormField("--s3-bucket", "S3 Bucket", "str", S3_BUCKET,
                  "S3 bucket for training data.", required=True),
        FormField("--max-jobs", "Max Tuning Jobs", "int", 100,
                  "Maximum number of AMT trials."),
        FormField("--max-parallel", "Max Parallel Jobs", "int", 5,
                  "Concurrent training jobs during tuning."),
        FormField("--instance-type", "Instance Type", "select", "ml.g4dn.xlarge",
                  "Training instance type.", ["ml.g4dn.xlarge", "ml.g5.xlarge"]),
        FormField("--no-spot", "Use On-Demand", "bool", False,
                  "Use on-demand instances instead of spot."),
    ],
))

_register(PresetConfig(
    id="download_model",
    name="Download Model from S3",
    description="Download trained model artifacts from a SageMaker job.",
    category="sagemaker",
    script="sagemaker/download_model.py",
    default_args={},
    form_fields=[
        FormField("--job-name", "Job Name", "str", "",
                  "SageMaker training job name."),
        FormField("--s3-uri", "S3 URI", "str", "",
                  "Direct S3 URI to model.tar.gz (alternative to job name)."),
        FormField("--region", "Region", "str", AWS_REGION,
                  "AWS region."),
    ],
))

# ── Infrastructure (Terraform) presets ─────────────────────────────────────

_register(PresetConfig(
    id="terraform_plan",
    name="Terraform Plan",
    description="Preview infrastructure changes without applying them.",
    category="infrastructure",
    script="terraform",
    subcommand="plan",
    default_args={},
    is_shell=True,
    cwd=TERRAFORM_DIR,
    form_fields=[
        FormField("-target", "Target Resource", "str", "",
                  "Optional: plan only a specific resource (e.g. aws_batch_job_definition.worker)."),
    ],
))

_register(PresetConfig(
    id="terraform_apply",
    name="Terraform Apply",
    description="Apply infrastructure changes. Will prompt for confirmation in the log.",
    category="infrastructure",
    script="terraform",
    subcommand="apply",
    default_args={},
    is_shell=True,
    cwd=TERRAFORM_DIR,
    form_fields=[
        FormField("-auto-approve", "Auto Approve", "bool", False,
                  "Skip interactive approval (use with caution)."),
        FormField("-target", "Target Resource", "str", "",
                  "Optional: apply only a specific resource."),
    ],
))

_register(PresetConfig(
    id="terraform_destroy",
    name="Terraform Destroy",
    description="Destroy all managed infrastructure. Dangerous!",
    category="infrastructure",
    script="terraform",
    subcommand="destroy",
    default_args={},
    is_shell=True,
    cwd=TERRAFORM_DIR,
    form_fields=[
        FormField("-auto-approve", "Auto Approve", "bool", False,
                  "Skip interactive approval (required for non-interactive destroy)."),
        FormField("-target", "Target Resource", "str", "",
                  "Optional: destroy only a specific resource."),
    ],
))

_register(PresetConfig(
    id="terraform_output",
    name="Terraform Output",
    description="Show current infrastructure output values (S3 bucket, queue names, role ARNs).",
    category="infrastructure",
    script="terraform",
    subcommand="output",
    default_args={},
    is_shell=True,
    cwd=TERRAFORM_DIR,
    form_fields=[],
))

_register(PresetConfig(
    id="terraform_init",
    name="Terraform Init",
    description="Initialize Terraform working directory and download providers.",
    category="infrastructure",
    script="terraform",
    subcommand="init",
    default_args={},
    is_shell=True,
    cwd=TERRAFORM_DIR,
    form_fields=[
        FormField("-upgrade", "Upgrade Providers", "bool", False,
                  "Upgrade provider plugins to the latest allowed versions."),
    ],
))

_register(PresetConfig(
    id="terraform_validate",
    name="Terraform Validate",
    description="Check Terraform configuration files for syntax errors.",
    category="infrastructure",
    script="terraform",
    subcommand="validate",
    default_args={},
    is_shell=True,
    cwd=TERRAFORM_DIR,
    form_fields=[],
))

# ── AWS Auth presets ──────────────────────────────────────────────────────

_register(PresetConfig(
    id="aws_sso_login",
    name="AWS SSO Login",
    description="Authenticate with AWS via SSO. Opens a browser link — click it to confirm.",
    category="aws",
    script="aws",
    subcommand="sso login",
    default_args={},
    is_shell=True,
    form_fields=[
        FormField("--profile", "AWS Profile", "str", "",
                  "Profile name from ~/.aws/config (leave empty for default)."),
    ],
))

_register(PresetConfig(
    id="aws_whoami",
    name="AWS Identity Check",
    description="Verify current AWS credentials (sts get-caller-identity).",
    category="aws",
    script="aws",
    subcommand="sts get-caller-identity",
    default_args={},
    is_shell=True,
    form_fields=[],
))

# ── Docker / ECR presets ─────────────────────────────────────────────────

if sys.platform == "win32":
    _rebuild_ecr_script = "powershell"
    _rebuild_ecr_subcommand = "-ExecutionPolicy Bypass -File scripts/rebuild_ecr_image.ps1"
else:
    _rebuild_ecr_script = "bash"
    _rebuild_ecr_subcommand = "scripts/rebuild_ecr_image.sh"

_register(PresetConfig(
    id="rebuild_ecr_image",
    name="Rebuild ECR Image",
    description="Pull latest SimC binary, rebuild worker image, and push to ECR. Keeps binary in sync with profiles.",
    category="docker",
    script=_rebuild_ecr_script,
    subcommand=_rebuild_ecr_subcommand,
    default_args={},
    is_shell=True,
    cwd=PROJECT_ROOT,
    form_fields=[],
))

_register(PresetConfig(
    id="pull_simc_latest",
    name="Pull SimC Latest (Local)",
    description="Pull the latest simulationcraftorg/simc Docker image locally. Use before local baseline generation.",
    category="docker",
    script="docker",
    subcommand="pull simulationcraftorg/simc:latest",
    default_args={},
    is_shell=True,
    form_fields=[],
))


def get_presets_by_category() -> dict[str, list[PresetConfig]]:
    """Return presets grouped by category."""
    groups: dict[str, list[PresetConfig]] = {}
    for p in PRESETS.values():
        groups.setdefault(p.category, []).append(p)
    return groups


def build_args_for_preset(preset: PresetConfig, overrides: dict | None = None) -> list[str]:
    """Build CLI arg list from preset defaults with optional overrides applied.

    Used by both the job launch route and the pipeline engine.
    """
    merged = dict(preset.default_args)
    if overrides:
        merged.update(overrides)
    args: list[str] = []
    for arg_name, arg_val in merged.items():
        if isinstance(arg_val, bool):
            if arg_val:
                args.append(arg_name)
        else:
            args.extend([arg_name, str(arg_val)])
    return args


# Shared form fields for pipeline launches (params needed across multiple steps)
PIPELINE_SHARED_FIELDS: list[FormField] = [
    FormField("--s3-bucket", "S3 Bucket", "str", S3_BUCKET,
              "S3 bucket for SageMaker training data.", required=True),
    FormField("--instance-type", "Instance Type", "select", "ml.g4dn.xlarge",
              "SageMaker training instance.", ["ml.g4dn.xlarge", "ml.g5.xlarge", "ml.p3.2xlarge"]),
]
