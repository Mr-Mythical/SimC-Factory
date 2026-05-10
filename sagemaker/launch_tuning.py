"""
Launch a SageMaker Automatic Model Tuning (AMT) job for hyperparameter
optimization of the DPS prediction model.

Replaces the local Optuna-based search with SageMaker's Bayesian tuning.
Parameter ranges are derived from the v3 analysis of 300 Optuna trials.

On completion, extracts the best hyperparameters and saves them to
best_hyperparameters.json for use by the daily retraining pipeline.

Uses boto3 directly (no sagemaker SDK dependency).

Usage:
    python launch_tuning.py --s3-bucket my-bucket --role-arn arn:aws:iam::...
    python launch_tuning.py --max-jobs 100 --max-parallel 5
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

# PyTorch 2.5.1 DLC image — same as launch_training.py
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
        account=account,
        region=region,
        framework=FRAMEWORK_VERSION,
        py=PY_VERSION,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Launch SageMaker hyperparameter tuning job")
    parser.add_argument("--s3-bucket", type=str, required=True)
    parser.add_argument("--role-arn", type=str, required=True)
    parser.add_argument("--region", type=str, default="eu-north-1")
    parser.add_argument("--instance-type", type=str, default="ml.g4dn.xlarge")
    parser.add_argument("--max-jobs", type=int, default=100,
                        help="Maximum number of tuning trials.")
    parser.add_argument("--max-parallel", type=int, default=3,
                        help="Maximum parallel training jobs.")
    parser.add_argument("--max-run", type=int, default=3600,
                        help="Max training time per trial in seconds.")
    parser.add_argument("--max-wait", type=int, default=7200,
                        help="Max total wait time per trial (spot delays).")
    parser.add_argument("--no-spot", action="store_true")
    parser.add_argument("--no-wait", action="store_true",
                        help="Submit and return immediately.")
    parser.add_argument("--data-file", type=str, default=DATA_FILE)
    parser.add_argument("--image-uri", type=str, default="",
                        help="Override the training container image URI.")
    return parser.parse_args()


def upload_training_data(s3, s3_bucket, data_file):
    s3_key = "sagemaker/training-data/all_specs_training_data.csv"
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    print(f"Uploading {data_file} -> {s3_uri}")
    s3.upload_file(data_file, s3_bucket, s3_key)
    return f"s3://{s3_bucket}/sagemaker/training-data"


def package_source_dir(source_dir, s3, s3_bucket, job_name):
    """Tar the training source dir and upload to S3."""
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


def extract_best_hyperparameters(sm, tuning_job_name):
    """Extract best hyperparameters from a completed tuning job."""
    result = sm.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name
    )

    best_job_name = result["BestTrainingJob"]["TrainingJobName"]
    best_job = sm.describe_training_job(TrainingJobName=best_job_name)
    raw_params = best_job["HyperParameters"]

    # Convert SageMaker string params back to typed values
    best_params = {
        "n_hidden_layers": int(raw_params.get("n-hidden-layers", "3")),
        "hidden_dim_1": int(raw_params.get("hidden-dim-1", "256")),
        "hidden_dim_2": int(raw_params.get("hidden-dim-2", "128")),
        "hidden_dim_3": int(raw_params.get("hidden-dim-3", "64")),
        "activation": raw_params.get("activation", "relu").strip('"'),
        "dropout": float(raw_params.get("dropout", "0.01")),
        "batch_size": int(raw_params.get("batch-size", "512")),
        "epochs": int(raw_params.get("epochs", "300")),
        "learning_rate": float(raw_params.get("learning-rate", "0.001")),
        "weight_decay": float(raw_params.get("weight-decay", "1e-6")),
        "test_size": float(raw_params.get("test-size", "0.15")),
        "random_state": int(raw_params.get("random-state", "42")),
        "cv_folds": int(raw_params.get("cv-folds", "0")),
    }

    best_metric = result["BestTrainingJob"]["FinalHyperParameterTuningJobObjectiveMetric"]
    return best_params, best_metric


def wait_for_tuning(sm, tuning_job_name):
    """Poll until tuning job completes or fails."""
    print("\nWaiting for tuning job to complete...")
    while True:
        resp = sm.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=tuning_job_name
        )
        status = resp["HyperParameterTuningJobStatus"]
        counts = resp.get("TrainingJobStatusCounters", {})
        completed = counts.get("Completed", 0)
        in_progress = counts.get("InProgress", 0)
        total = resp["HyperParameterTuningJobConfig"]["ResourceLimits"]["MaxNumberOfTrainingJobs"]
        print(f"  Status: {status} — {completed}/{total} completed, {in_progress} in progress",
              flush=True)

        if status in ("Completed", "Failed", "Stopped"):
            return resp

        time.sleep(60)


def rebuild_combined_csv(data_file, training_data_dir):
    """Build all_specs_training_data.csv from per-spec CSVs if it doesn't exist."""
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


def main():
    args = parse_args()
    wait = not args.no_wait

    # Always rebuild the combined CSV from per-spec files to pick up new data
    if not rebuild_combined_csv(args.data_file, TRAINING_DATA_DIR):
        if not os.path.exists(args.data_file):
            print(f"ERROR: Training data not found at {args.data_file}")
            print(f"       and no per-spec CSVs in {TRAINING_DATA_DIR}")
            sys.exit(1)

    s3 = boto3.client("s3", region_name=args.region)
    sm = boto3.client("sagemaker", region_name=args.region)

    training_input = upload_training_data(s3, args.s3_bucket, args.data_file)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    tuning_job_name = f"dps-tuning-{timestamp}"
    use_spot = not args.no_spot

    output_path = f"s3://{args.s3_bucket}/sagemaker/tuning"
    checkpoint_s3 = f"s3://{args.s3_bucket}/sagemaker/checkpoints"

    # Package source code
    source_s3 = package_source_dir(SCRIPT_DIR, s3, args.s3_bucket, tuning_job_name)

    image_uri = args.image_uri or get_image_uri(args.region)

    # Static hyperparameters (not tuned)
    # Keys use dashes to match train.py's argparse flags
    static_hyperparams = {
        "batch-size": "512",
        "activation": "relu",
        "n-hidden-layers": "2",
        "test-size": "0.15",
        "random-state": "42",
        "cv-folds": "0",
        "sagemaker_submit_directory": source_s3,
        "sagemaker_program": "train.py",
    }

    # Stopping condition
    stopping = {"MaxRuntimeInSeconds": args.max_run}
    if use_spot:
        stopping["MaxWaitTimeInSeconds"] = args.max_wait

    # Build the tuning job config
    tuning_config = {
        "HyperParameterTuningJobName": tuning_job_name,
        "HyperParameterTuningJobConfig": {
            "Strategy": "Bayesian",
            "HyperParameterTuningJobObjective": {
                "Type": "Minimize",
                "MetricName": "test_mae",
            },
            "ResourceLimits": {
                "MaxNumberOfTrainingJobs": args.max_jobs,
                "MaxParallelTrainingJobs": args.max_parallel,
            },
            "ParameterRanges": {
                "IntegerParameterRanges": [
                    {
                        "Name": "hidden-dim-1",
                        "MinValue": "180",
                        "MaxValue": "280",
                    },
                    {
                        "Name": "hidden-dim-2",
                        "MinValue": "150",
                        "MaxValue": "250",
                    },
                    {
                        "Name": "hidden-dim-3",
                        "MinValue": "8",
                        "MaxValue": "128",
                    },
                    {
                        "Name": "epochs",
                        "MinValue": "1000",
                        "MaxValue": "2000",
                    },
                ],
                "ContinuousParameterRanges": [
                    {
                        "Name": "dropout",
                        "MinValue": "0.0",
                        "MaxValue": "0.05",
                    },
                    {
                        "Name": "learning-rate",
                        "MinValue": "8e-5",
                        "MaxValue": "1.5e-4",
                        "ScalingType": "Logarithmic",
                    },
                    {
                        "Name": "weight-decay",
                        "MinValue": "1e-7",
                        "MaxValue": "5e-5",
                        "ScalingType": "Logarithmic",
                    },
                ],
                "CategoricalParameterRanges": [],
            },
        },
        "TrainingJobDefinition": {
            "StaticHyperParameters": static_hyperparams,
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
            "RoleArn": args.role_arn,
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": training_input,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ContentType": "text/csv",
                }
            ],
            "OutputDataConfig": {"S3OutputPath": output_path},
            "ResourceConfig": {
                "InstanceType": args.instance_type,
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            },
            "StoppingCondition": stopping,
            "EnableManagedSpotTraining": use_spot,
            "CheckpointConfig": {
                "S3Uri": checkpoint_s3,
                "LocalPath": "/opt/ml/checkpoints",
            },
        },
        "Tags": [
            {"Key": "Project", "Value": "simc-batch-array"},
            {"Key": "Workflow", "Value": "hyperparameter-tuning"},
        ],
    }

    print(f"\nLaunching SageMaker tuning job: {tuning_job_name}")
    print(f"  Instance: {args.instance_type} ({'spot' if use_spot else 'on-demand'})")
    print(f"  Image: {image_uri}")
    print(f"  Max jobs: {args.max_jobs}, Max parallel: {args.max_parallel}")
    print(f"  Strategy: Bayesian, Objective: minimize test_mae")
    print(f"\nParameter ranges (Phase 2 - narrowed):")
    print(f"  n_hidden_layers: Fixed 2 layers")
    print(f"  hidden_dim_1: Integer [180, 280]")
    print(f"  hidden_dim_2: Integer [150, 250]")
    print(f"  hidden_dim_3: Integer [8, 128] (not used)")
    print(f"  epochs: Integer [900, 1500]")
    print(f"  dropout: Continuous [0.0, 0.05]")
    print(f"  learning_rate: Continuous [8e-5, 1.5e-4] (Logarithmic)")
    print(f"  weight_decay: Continuous [1e-7, 5e-5] (Logarithmic)")

    sm.create_hyper_parameter_tuning_job(**tuning_config)
    print(f"  Tuning job created successfully.")

    if wait:
        wait_for_tuning(sm, tuning_job_name)

        best_params, best_metric = extract_best_hyperparameters(sm, tuning_job_name)

        print(f"\nTuning complete: {tuning_job_name}")
        print(f"  Best objective: {best_metric['MetricName']} = {best_metric['Value']:.4f}")
        print(f"\nBest hyperparameters:")
        for k, v in sorted(best_params.items()):
            print(f"  {k}: {v}")

        with open(HYPERPARAMS_FILE, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\nSaved best hyperparameters to {HYPERPARAMS_FILE}")
        print("Daily retraining will now use these parameters.")
    else:
        print(f"\nTuning job submitted: {tuning_job_name}")
        print(f"  Monitor with:")
        print(f"  aws sagemaker describe-hyper-parameter-tuning-job \\")
        print(f"    --hyper-parameter-tuning-job-name {tuning_job_name} --region {args.region}")


if __name__ == "__main__":
    main()
