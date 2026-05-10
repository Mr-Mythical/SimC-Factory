"""
Launch SageMaker training job(s) for the DPS prediction model.

Supports best-of-N training: launches multiple jobs with different random
seeds and keeps only the model with the lowest test MAE. All completed
models are saved as snapshots for comparison.

Uses Managed Spot Training with S3 checkpointing for cost savings and
interruption recovery. Reads hyperparameters (including cv_folds and
test_size) from best_hyperparameters.json.

Uses boto3 directly (no sagemaker SDK dependency).

Usage:
    python launch_training.py --s3-bucket my-bucket --role-arn arn:...
    python launch_training.py --num-runs 5 --max-parallel 3
    python launch_training.py --num-runs 1 --no-spot
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
import time
from datetime import datetime

import glob as globmod

import boto3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LOCAL_DIR = os.path.join(PROJECT_ROOT, "local")
DATA_FILE = os.path.join(PROJECT_ROOT, "all_specs_training_data.csv")
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
HYPERPARAMS_FILE = os.path.join(SCRIPT_DIR, "best_hyperparameters.json")
LOCAL_MODEL_DIR = os.path.join(LOCAL_DIR, "nn_website_model", "deep_nn")

FRAMEWORK_VERSION = "2.5.1"
PY_VERSION = "py311"
IMAGE_URI_TEMPLATE = (
    "{account}.dkr.ecr.{region}.amazonaws.com/"
    "pytorch-training:{framework}-gpu-{py}-cu124-ubuntu22.04-sagemaker"
)
DLC_ACCOUNTS = {
    "eu-north-1": "763104351884",
    "eu-west-1": "763104351884",
    "us-east-1": "763104351884",
    "us-west-2": "763104351884",
}


def get_image_uri(region):
    account = DLC_ACCOUNTS.get(region)
    if not account:
        raise ValueError(f"No DLC account known for region {region}. "
                         f"Add it to DLC_ACCOUNTS or pass --image-uri.")
    return IMAGE_URI_TEMPLATE.format(
        account=account, region=region,
        framework=FRAMEWORK_VERSION, py=PY_VERSION,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Launch SageMaker training job")
    parser.add_argument("--s3-bucket", type=str, required=True,
                        help="S3 bucket name.")
    parser.add_argument("--role-arn", type=str, required=True,
                        help="SageMaker execution role ARN.")
    parser.add_argument("--region", type=str, default="eu-north-1")
    parser.add_argument("--instance-type", type=str, default="ml.g4dn.xlarge")
    parser.add_argument("--max-run", type=int, default=3600,
                        help="Maximum training time in seconds.")
    parser.add_argument("--max-wait", type=int, default=7200,
                        help="Maximum total wait time (including spot delays).")
    parser.add_argument("--no-spot", action="store_true",
                        help="Use on-demand instances instead of spot.")
    parser.add_argument("--no-wait", action="store_true",
                        help="Submit the job(s) and return immediately.")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of training runs with different seeds. Best wins.")
    parser.add_argument("--max-parallel", type=int, default=3,
                        help="Maximum concurrent SageMaker training jobs.")
    parser.add_argument("--hyperparams-file", type=str, default=HYPERPARAMS_FILE)
    parser.add_argument("--data-file", type=str, default=DATA_FILE)
    parser.add_argument("--image-uri", type=str, default="",
                        help="Override the training container image URI.")
    return parser.parse_args()


def upload_training_data(s3, s3_bucket, data_file):
    """Upload training data CSV to S3 and return the S3 URI."""
    s3_key = "sagemaker/training-data/all_specs_training_data.csv"
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    print(f"Uploading {data_file} -> {s3_uri}")
    s3.upload_file(data_file, s3_bucket, s3_key)
    print(f"  Upload complete ({os.path.getsize(data_file) / 1024:.1f} KB)")
    return f"s3://{s3_bucket}/sagemaker/training-data"


def package_source_dir(source_dir, s3, s3_bucket, job_name):
    """Tar the training source dir and upload to S3, return the S3 URI."""
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            for fname in os.listdir(source_dir):
                fpath = os.path.join(source_dir, fname)
                if os.path.isfile(fpath) and not fname.startswith("."):
                    tar.add(fpath, arcname=fname)
        s3_key = f"sagemaker/source/{job_name}/sourcedir.tar.gz"
        s3.upload_file(tmp_path, s3_bucket, s3_key)
        return f"s3://{s3_bucket}/{s3_key}"
    finally:
        os.unlink(tmp_path)


def rebuild_combined_csv(data_file, training_data_dir):
    """Build all_specs_training_data.csv from per-spec CSVs."""
    import pandas as pd
    spec_csvs = sorted(globmod.glob(os.path.join(training_data_dir, "*.csv")))
    if not spec_csvs:
        return False
    print(f"Building combined CSV from {len(spec_csvs)} per-spec files...")
    frames = [pd.read_csv(f, encoding="utf-8") for f in spec_csvs]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(data_file, index=False, encoding="utf-8")
    print(f"  Created {data_file} ({len(combined)} rows)")
    return True


def create_training_job(sm, job_name, image_uri, role_arn, training_input,
                        output_path, source_s3, sm_hyperparams,
                        instance_type, max_run, max_wait, use_spot, bucket):
    """Create a single SageMaker training job."""
    checkpoint_s3 = f"s3://{bucket}/sagemaker/checkpoints/{job_name}"
    training_params = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "MetricDefinitions": [
                {"Name": "train_mae", "Regex": r"train_mae=([0-9.]+);"},
                {"Name": "test_mae", "Regex": r"test_mae=([0-9.]+);"},
                {"Name": "train_rmse", "Regex": r"train_rmse=([0-9.]+);"},
                {"Name": "test_rmse", "Regex": r"test_rmse=([0-9.]+);"},
                {"Name": "train_r2", "Regex": r"train_r2=([0-9.]+);"},
                {"Name": "test_r2", "Regex": r"test_r2=([0-9.]+);"},
                {"Name": "cv_mae", "Regex": r"cv_mae=([0-9.]+);"},
            ],
        },
        "RoleArn": role_arn,
        "InputDataConfig": [{
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": training_input,
                "S3DataDistributionType": "FullyReplicated",
            }},
            "ChannelName": "training",
            "ContentType": "text/csv",
        }],
        "OutputDataConfig": {"S3OutputPath": output_path},
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": max_run},
        "HyperParameters": sm_hyperparams,
        "CheckpointConfig": {
            "S3Uri": checkpoint_s3,
            "LocalPath": "/opt/ml/checkpoints",
        },
        "Tags": [
            {"Key": "Project", "Value": "simc-batch-array"},
            {"Key": "Workflow", "Value": "daily-retrain"},
        ],
    }

    if use_spot:
        training_params["EnableManagedSpotTraining"] = True
        training_params["StoppingCondition"]["MaxWaitTimeInSeconds"] = max_wait

    sm.create_training_job(**training_params)


def extract_metrics(resp):
    """Extract metric values from a completed training job response."""
    return {m["MetricName"]: m["Value"] for m in resp.get("FinalMetricDataList", [])}


def wait_for_jobs(sm, job_names):
    """Poll until all training jobs complete or fail."""
    pending = set(job_names)
    results = {}

    print(f"\nWaiting for {len(pending)} training job(s) to complete...")
    while pending:
        for job_name in list(pending):
            resp = sm.describe_training_job(TrainingJobName=job_name)
            status = resp["TrainingJobStatus"]

            if status in ("Completed", "Failed", "Stopped"):
                results[job_name] = resp
                pending.discard(job_name)
                short = job_name.rsplit("-", 1)[-1]
                if status == "Completed":
                    mae = extract_metrics(resp).get("test_mae", "?")
                    print(f"  {short}: Completed (test_mae={mae})")
                else:
                    print(f"  {short}: {status} — {resp.get('FailureReason', '?')}")

        if pending:
            print(f"  ... {len(pending)} job(s) still running", flush=True)
            time.sleep(30)

    return results


def print_comparison_table(results):
    """Print a comparison table of all completed runs."""
    completed = []
    for job_name, resp in sorted(results.items()):
        if resp["TrainingJobStatus"] != "Completed":
            continue
        completed.append((job_name, extract_metrics(resp)))

    if not completed:
        return

    best_mae = min(m.get("test_mae", float("inf")) for _, m in completed)

    print(f"\n{'=' * 78}")
    print("TRAINING RUN COMPARISON")
    print(f"{'=' * 78}")
    print(f"  {'Run':<12s} {'test_mae':>10s} {'train_mae':>10s} "
          f"{'test_r2':>10s} {'cv_mae':>10s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for job_name, metrics in completed:
        short = job_name.rsplit("-", 1)[-1]
        test_mae = metrics.get("test_mae")
        train_mae = metrics.get("train_mae")
        test_r2 = metrics.get("test_r2")
        cv_mae = metrics.get("cv_mae")

        cols = [f"  {short:<12s}"]
        cols.append(f"{test_mae:10.1f}" if test_mae is not None else f"{'—':>10s}")
        cols.append(f"{train_mae:10.1f}" if train_mae is not None else f"{'—':>10s}")
        cols.append(f"{test_r2:10.4f}" if test_r2 is not None else f"{'—':>10s}")
        cols.append(f"{cv_mae:10.1f}" if cv_mae is not None else f"{'—':>10s}")

        marker = " <-- best" if test_mae == best_mae else ""
        print(" ".join(cols) + marker)

    print(f"{'=' * 78}")


def launch_jobs_with_limit(sm, job_specs, max_parallel, image_uri, role_arn,
                           training_input, output_path, source_s3,
                           instance_type, max_run, max_wait, use_spot, bucket):
    """Launch jobs respecting max_parallel concurrency limit."""
    launched = []
    queue = list(job_specs)

    while queue:
        # Count currently running jobs
        running = 0
        for jn in launched:
            try:
                resp = sm.describe_training_job(TrainingJobName=jn)
                if resp["TrainingJobStatus"] == "InProgress":
                    running += 1
            except Exception:
                pass

        slots = max(0, max_parallel - running)
        batch = queue[:slots] if slots > 0 else []
        queue = queue[len(batch):]

        for spec in batch:
            sm_hp = {k.replace("_", "-"): str(v) for k, v in spec["hyperparams"].items()}
            sm_hp["sagemaker_submit_directory"] = source_s3
            sm_hp["sagemaker_program"] = "train.py"

            print(f"  Launching: {spec['label']}")
            create_training_job(
                sm, spec["name"], image_uri, role_arn, training_input,
                output_path, source_s3, sm_hp,
                instance_type, max_run, max_wait, use_spot, bucket,
            )
            launched.append(spec["name"])

        if queue:
            print(f"  ... {len(queue)} job(s) queued, waiting for slots "
                  f"({running + len(batch)}/{max_parallel} active)")
            time.sleep(30)

    return launched


def main():
    args = parse_args()
    num_runs = max(1, args.num_runs)
    max_parallel = max(1, args.max_parallel)

    # Rebuild combined CSV
    if not rebuild_combined_csv(args.data_file, TRAINING_DATA_DIR):
        if not os.path.exists(args.data_file):
            print(f"ERROR: Training data not found at {args.data_file}")
            print(f"       and no per-spec CSVs in {TRAINING_DATA_DIR}")
            sys.exit(1)
    if not os.path.exists(args.hyperparams_file):
        print(f"ERROR: Hyperparameters file not found at {args.hyperparams_file}")
        sys.exit(1)

    with open(args.hyperparams_file) as f:
        hyperparams = json.load(f)

    print(f"Hyperparameters from {args.hyperparams_file}:")
    for k, v in sorted(hyperparams.items()):
        print(f"  {k}: {v}")

    s3 = boto3.client("s3", region_name=args.region)
    sm = boto3.client("sagemaker", region_name=args.region)

    # Upload data and source once (shared across all runs)
    training_input = upload_training_data(s3, args.s3_bucket, args.data_file)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    use_spot = not args.no_spot
    image_uri = args.image_uri or get_image_uri(args.region)
    output_path = f"s3://{args.s3_bucket}/sagemaker/models"
    base_job_name = f"dps-retrain-full-{timestamp}"
    source_s3 = package_source_dir(SCRIPT_DIR, s3, args.s3_bucket, base_job_name)
    base_seed = hyperparams.get("random_state", 42)

    # Build job specs
    job_specs = []
    for i in range(num_runs):
        seed = base_seed + i
        hp = dict(hyperparams)
        hp["random_state"] = seed

        name = base_job_name if num_runs == 1 else f"{base_job_name}-run{i + 1}"
        job_specs.append({
            "name": name,
            "hyperparams": hp,
            "label": f"{name} (seed={seed})",
        })

    print(f"\nLaunching {num_runs} training run(s) (max {max_parallel} parallel)")
    print(f"  Instance: {args.instance_type} ({'spot' if use_spot else 'on-demand'})")
    print(f"  Image: {image_uri}")
    print(f"  Max run: {args.max_run}s")

    job_names = launch_jobs_with_limit(
        sm, job_specs, max_parallel, image_uri, args.role_arn,
        training_input, output_path, source_s3,
        args.instance_type, args.max_run, args.max_wait, use_spot, args.s3_bucket,
    )

    if args.no_wait:
        print(f"\nTraining job(s) submitted:")
        for jn in job_names:
            print(f"  aws sagemaker describe-training-job "
                  f"--training-job-name {jn} --region {args.region}")
        return

    # Wait for all
    results = wait_for_jobs(sm, job_names)
    completed = {jn: r for jn, r in results.items()
                 if r["TrainingJobStatus"] == "Completed"}

    if not completed:
        print("\nERROR: All training runs failed.")
        for jn, resp in results.items():
            print(f"  {jn}: {resp.get('FailureReason', 'Unknown')}")
        sys.exit(1)

    failed_count = len(results) - len(completed)
    if failed_count:
        print(f"\nWARNING: {failed_count}/{len(results)} run(s) failed.")

    if len(completed) > 1:
        print_comparison_table(results)

    # Download ALL completed models as snapshots
    from download_model import download_to_snapshot, download_and_extract
    print(f"\nDownloading {len(completed)} model(s) to snapshots...")
    model_metadata = {}  # job_name -> nn_metadata.json contents
    for jn, resp in completed.items():
        s3_uri = resp["ModelArtifacts"]["S3ModelArtifacts"]
        seed = jn.rsplit("-run", 1)[-1] if "-run" in jn else "0"
        label = f"training seed={seed}"
        meta = download_to_snapshot(s3_uri, jn, args.region, label)
        model_metadata[jn] = meta

    # Select best by test_mae from downloaded metadata (more accurate than regex)
    best_jn = min(
        model_metadata,
        key=lambda jn: model_metadata[jn].get("test_mae", float("inf")),
    )
    best_meta = model_metadata[best_jn]

    if num_runs > 1:
        print(f"\nBest run: {best_jn} (test_mae={best_meta.get('test_mae', '?'):.1f})")

    # Print per-spec MAE for best model
    per_spec = best_meta.get("per_spec_mae", {})
    if per_spec:
        print(f"\nPer-spec MAE (best model):")
        for spec, mae in sorted(per_spec.items(), key=lambda x: -x[1]):
            print(f"  {spec:<45s} {mae:8.1f}")

    # Install best as production model
    print(f"\nInstalling best model as production...")
    from snapshot import snapshot_current_model
    snap = snapshot_current_model(model_dir=LOCAL_MODEL_DIR, label=best_jn)
    if snap:
        print(f"  Previous model snapshotted before overwrite")

    model_artifact = completed[best_jn]["ModelArtifacts"]["S3ModelArtifacts"]
    download_and_extract(model_artifact, LOCAL_MODEL_DIR, args.region)
    print(f"\nModel downloaded to {LOCAL_MODEL_DIR}")


if __name__ == "__main__":
    main()
