"""
Download model artifacts from a completed SageMaker training job.

Fetches the model.tar.gz from S3, extracts dps_net.pt, scalers.pkl, and
nn_metadata.json into the local model directory.

Usage:
    python download_model.py --s3-uri s3://bucket/sagemaker/models/job/output/model.tar.gz
    python download_model.py --job-name dps-retrain-20260417-120000 --region eu-north-1
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile

import boto3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

from snapshot import snapshot_current_model
LOCAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "local", "nn_website_model", "deep_nn")
DEFAULT_SNAPSHOTS_DIR = os.path.join(PROJECT_ROOT, "local", "nn_website_model", "snapshots")


def parse_args():
    parser = argparse.ArgumentParser(description="Download SageMaker model artifacts")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--s3-uri",
        type=str,
        help="Full S3 URI to model.tar.gz artifact.",
    )
    group.add_argument(
        "--job-name",
        type=str,
        help="SageMaker training job name (will look up the model artifact URI).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="eu-north-1",
        help="AWS region.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=LOCAL_MODEL_DIR,
        help="Local directory to extract model artifacts into.",
    )
    return parser.parse_args()


def get_model_s3_uri_from_job(job_name, region):
    """Look up the model artifact S3 URI from a training job name."""
    sm = boto3.client("sagemaker", region_name=region)
    job = sm.describe_training_job(TrainingJobName=job_name)

    status = job["TrainingJobStatus"]
    if status != "Completed":
        print(f"WARNING: Training job status is '{status}', not 'Completed'")

    model_uri = job.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    if not model_uri:
        print(f"ERROR: No model artifact found for job {job_name}")
        sys.exit(1)

    return model_uri


def parse_s3_uri(s3_uri):
    """Parse s3://bucket/key into (bucket, key)."""
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    path = s3_uri[5:]
    bucket, _, key = path.partition("/")
    return bucket, key


def download_and_extract(s3_uri, output_dir, region):
    """Download model.tar.gz from S3 and extract to output_dir."""
    bucket, key = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3", region_name=region)

    os.makedirs(output_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        print(f"Downloading {s3_uri} ...")
        s3.download_file(bucket, key, tmp_path)
        size_kb = os.path.getsize(tmp_path) / 1024
        print(f"  Downloaded {size_kb:.1f} KB")

        print(f"Extracting to {output_dir} ...")
        with tarfile.open(tmp_path, "r:gz") as tar:
            members = tar.getnames()
            print(f"  Archive contains: {', '.join(members)}")
            tar.extractall(path=output_dir)
    finally:
        os.unlink(tmp_path)

    # Verify expected files
    expected = ["dps_net.pt", "scalers.pkl", "nn_metadata.json"]
    for fname in expected:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            print(f"  {fname}: {os.path.getsize(fpath) / 1024:.1f} KB")
        else:
            print(f"  WARNING: Expected {fname} not found in artifact")


def download_to_snapshot(s3_uri, snapshot_name, region, label,
                         snapshots_dir=DEFAULT_SNAPSHOTS_DIR):
    """Download model to snapshots/{snapshot_name}/ and write snapshot_info.json.

    Returns the nn_metadata.json contents (dict), or empty dict on failure.
    """
    snap_dir = os.path.join(snapshots_dir, snapshot_name)
    download_and_extract(s3_uri, snap_dir, region)

    # Read nn_metadata.json for metrics
    meta_path = os.path.join(snap_dir, "nn_metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    # Write snapshot_info.json
    from datetime import datetime, timezone
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "type": "full",
        "test_mae": meta.get("test_mae"),
        "train_mae": meta.get("train_mae"),
        "test_rmse": meta.get("test_rmse"),
        "train_r2": meta.get("train_r2"),
        "test_r2": meta.get("test_r2"),
        "n_specs": meta.get("n_specs"),
        "train_samples": meta.get("train_samples"),
        "test_samples": meta.get("test_samples"),
        "per_spec_mae": meta.get("per_spec_mae", {}),
        "cv_folds": meta.get("cv_folds"),
        "cv_mae": meta.get("cv_mae"),
        "files": [f for f in os.listdir(snap_dir) if f != "snapshot_info.json"],
    }
    with open(os.path.join(snap_dir, "snapshot_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return meta


def main():
    args = parse_args()

    if args.s3_uri:
        s3_uri = args.s3_uri
    else:
        s3_uri = get_model_s3_uri_from_job(args.job_name, args.region)
        print(f"Resolved job '{args.job_name}' -> {s3_uri}")

    # Snapshot current model before overwriting
    label = "download"
    if args.job_name:
        label = args.job_name
    snap = snapshot_current_model(model_dir=args.output_dir, label=label)
    if snap:
        print(f"Previous model snapshotted before overwrite")
    else:
        print("No existing model to snapshot; first download.")

    download_and_extract(s3_uri, args.output_dir, args.region)
    print(f"\nModel artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
