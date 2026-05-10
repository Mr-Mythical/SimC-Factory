"""
Run a test-size grid experiment on SageMaker.

Launches training jobs across test-size values with multiple seeds to find which
split ratio produces the best model. Saves results to local/experiment_results/
and updates best_hyperparameters.json with the winning settings.

cv_folds is NOT varied in the grid — it doesn't affect model training, only adds
a post-training cross-validation evaluation. A single cv_folds value is used for
all jobs (default: from best_hyperparameters.json).

Usage:
    python launch_experiment.py --s3-bucket my-bucket --role-arn arn:...
    python launch_experiment.py --test-sizes 0.10,0.15,0.20
    python launch_experiment.py --test-sizes 0.10,0.15,0.20 --cv-folds 5
"""

import argparse
import json
import os
import statistics
import sys
import tarfile
import tempfile
import time
from datetime import datetime

import glob as globmod

import boto3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FILE = os.path.join(PROJECT_ROOT, "all_specs_training_data.csv")
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
HYPERPARAMS_FILE = os.path.join(SCRIPT_DIR, "best_hyperparameters.json")
LOCAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "local", "nn_website_model", "deep_nn")

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
        raise ValueError(f"No DLC account known for region {region}.")
    return IMAGE_URI_TEMPLATE.format(
        account=account, region=region,
        framework=FRAMEWORK_VERSION, py=PY_VERSION,
    )


def parse_args():
    p = argparse.ArgumentParser(description="CV / test-size grid experiment")
    p.add_argument("--s3-bucket", required=True, help="S3 bucket name.")
    p.add_argument("--role-arn", required=True, help="SageMaker execution role ARN.")
    p.add_argument("--region", default="eu-north-1")
    p.add_argument("--instance-type", default="ml.g4dn.xlarge")
    p.add_argument("--max-run", type=int, default=3600,
                   help="Max training time per job in seconds.")
    p.add_argument("--max-wait", type=int, default=7200,
                   help="Max total wait time per job (spot delays).")
    p.add_argument("--no-spot", action="store_true",
                   help="Use on-demand instances instead of spot.")
    p.add_argument("--max-parallel", type=int, default=3,
                   help="Maximum concurrent SageMaker training jobs.")
    p.add_argument("--seeds-per-combo", type=int, default=2,
                   help="Runs per combo to measure variance.")
    p.add_argument("--hyperparams-file", default=HYPERPARAMS_FILE)
    p.add_argument("--data-file", default=DATA_FILE)
    p.add_argument("--image-uri", default="")
    p.add_argument("--no-wait", action="store_true",
                   help="Submit jobs and return immediately.")

    # What to vary
    p.add_argument("--cv-folds", type=int, default=None,
                   help="CV folds for all jobs (default: from hyperparams file). "
                        "Does not affect model training, only adds CV evaluation.")
    p.add_argument("--test-sizes", default="0.10,0.15,0.20",
                   help="Comma-separated test-size values to test (default: 0.10,0.15,0.20).")
    p.add_argument("--no-test-size", action="store_true",
                   help="Don't vary test-size — use the value from hyperparams file.")
    p.add_argument("--resume", type=str, default="",
                   help="Resume a previous experiment by its timestamp prefix (e.g. 'dps-exp-20260420-153000'). "
                        "Skips jobs that already exist in SageMaker.")
    return p.parse_args()


# ── Shared helpers ────────────────────────────────────────────────────────


def rebuild_combined_csv(data_file, training_data_dir):
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


def upload_training_data(s3, bucket, data_file):
    s3_key = "sagemaker/training-data/all_specs_training_data.csv"
    print(f"Uploading {data_file} -> s3://{bucket}/{s3_key}")
    s3.upload_file(data_file, bucket, s3_key)
    print(f"  Upload complete ({os.path.getsize(data_file) / 1024:.1f} KB)")
    return f"s3://{bucket}/sagemaker/training-data"


def package_source_dir(source_dir, s3, bucket, job_name):
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            for fname in os.listdir(source_dir):
                fpath = os.path.join(source_dir, fname)
                if os.path.isfile(fpath) and not fname.startswith("."):
                    tar.add(fpath, arcname=fname)
        s3_key = f"sagemaker/source/{job_name}/sourcedir.tar.gz"
        s3.upload_file(tmp_path, bucket, s3_key)
        return f"s3://{bucket}/{s3_key}"
    finally:
        os.unlink(tmp_path)


def create_training_job(sm, job_name, image_uri, role_arn, training_input,
                        output_path, source_s3, sm_hyperparams,
                        instance_type, max_run, max_wait, use_spot, bucket):
    checkpoint_s3 = f"s3://{bucket}/sagemaker/checkpoints/{job_name}"
    params = {
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
            {"Key": "Workflow", "Value": "experiment"},
        ],
    }
    if use_spot:
        params["EnableManagedSpotTraining"] = True
        params["StoppingCondition"]["MaxWaitTimeInSeconds"] = max_wait
    sm.create_training_job(**params)


def extract_metrics(resp):
    return {m["MetricName"]: m["Value"] for m in resp.get("FinalMetricDataList", [])}


# ── Launch with concurrency limit ─────────────────────────────────────────


def launch_jobs_with_limit(sm, job_specs, max_parallel, image_uri, role_arn,
                           training_input, output_path, source_s3,
                           instance_type, max_run, max_wait, use_spot, bucket):
    """Launch jobs respecting max_parallel. Returns list of job names."""
    launched = []
    queue = list(job_specs)

    while queue:
        # Count running
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


def wait_for_jobs(sm, job_names):
    pending = set(job_names)
    results = {}
    print(f"\nWaiting for {len(pending)} job(s) to complete...")
    while pending:
        for jn in list(pending):
            resp = sm.describe_training_job(TrainingJobName=jn)
            status = resp["TrainingJobStatus"]
            if status in ("Completed", "Failed", "Stopped"):
                results[jn] = resp
                pending.discard(jn)
                short = jn.split(f"-")[-1] if "-" in jn else jn
                if status == "Completed":
                    m = extract_metrics(resp)
                    print(f"  {short}: Completed (test_mae={m.get('test_mae', '?')})")
                else:
                    print(f"  {short}: {status} — {resp.get('FailureReason', '?')}")
        if pending:
            print(f"  ... {len(pending)} job(s) still running", flush=True)
            time.sleep(30)
    return results


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # Rebuild CSV
    if not rebuild_combined_csv(args.data_file, TRAINING_DATA_DIR):
        if not os.path.exists(args.data_file):
            print(f"ERROR: No training data found.")
            sys.exit(1)
    if not os.path.exists(args.hyperparams_file):
        print(f"ERROR: Hyperparameters file not found: {args.hyperparams_file}")
        sys.exit(1)

    with open(args.hyperparams_file) as f:
        hyperparams = json.load(f)

    # CV folds — single value, not a grid axis (doesn't affect model training)
    cv_folds = args.cv_folds if args.cv_folds is not None else hyperparams.get("cv_folds", 0)

    # Build grid axes — only test_size × seeds
    if args.no_test_size:
        ts_grid = [hyperparams.get("test_size", 0.15)]
    else:
        ts_grid = [float(x.strip()) for x in args.test_sizes.split(",")]

    base_seed = hyperparams.get("random_state", 42)
    seeds_per = max(1, args.seeds_per_combo)
    total_jobs = len(ts_grid) * seeds_per

    print(f"\n{'=' * 78}")
    print("EXPERIMENT: Test-Size Grid Search")
    print(f"{'=' * 78}")
    print(f"  CV folds:        {cv_folds} (fixed — does not affect model)")
    print(f"  Test sizes:      {ts_grid}")
    print(f"  Seeds per combo: {seeds_per}")
    print(f"  Total jobs:      {total_jobs}")
    print(f"  Max parallel:    {args.max_parallel}")
    print(f"  Instance:        {args.instance_type} ({'spot' if not args.no_spot else 'on-demand'})")

    s3 = boto3.client("s3", region_name=args.region)
    sm = boto3.client("sagemaker", region_name=args.region)

    training_input = upload_training_data(s3, args.s3_bucket, args.data_file)
    use_spot = not args.no_spot
    image_uri = args.image_uri or get_image_uri(args.region)
    output_path = f"s3://{args.s3_bucket}/sagemaker/experiments"

    # Resume mode: reuse existing base_name, skip already-launched jobs
    if args.resume:
        base_name = args.resume
        timestamp = base_name.replace("dps-exp-", "")
        source_s3 = package_source_dir(SCRIPT_DIR, s3, args.s3_bucket, base_name)
        print(f"\nRESUMING experiment: {base_name}")
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        base_name = f"dps-exp-{timestamp}"
        source_s3 = package_source_dir(SCRIPT_DIR, s3, args.s3_bucket, base_name)

    # Build job specs
    job_specs = []
    job_meta = {}  # job_name -> {cv, ts, seed, label}
    for ts in ts_grid:
        for s in range(seeds_per):
            seed = base_seed + s
            hp = dict(hyperparams)
            hp["cv_folds"] = cv_folds
            hp["test_size"] = ts
            hp["random_state"] = seed

            suffix = f"ts{int(ts*100)}-s{seed}"
            job_name = f"{base_name}-{suffix}"
            label = f"ts={ts:.2f} seed={seed}"

            job_specs.append({
                "name": job_name,
                "hyperparams": hp,
                "label": label,
            })
            job_meta[job_name] = {"cv": cv_folds, "ts": ts, "seed": seed, "label": label}

    # In resume mode, check which jobs already exist and skip them
    already_launched = []
    if args.resume:
        for spec in list(job_specs):
            try:
                resp = sm.describe_training_job(TrainingJobName=spec["name"])
                status = resp["TrainingJobStatus"]
                print(f"  {spec['name']}: already {status}")
                already_launched.append(spec["name"])
                if status in ("Completed", "Failed", "Stopped"):
                    job_specs.remove(spec)
                elif status == "InProgress":
                    job_specs.remove(spec)
            except Exception:
                pass  # Job doesn't exist yet, will be launched

        print(f"  {len(already_launched)} job(s) already exist, "
              f"{len(job_specs)} remaining to launch")

    print(f"\nLaunching {len(job_specs)} experiment jobs...")
    newly_launched = launch_jobs_with_limit(
        sm, job_specs, max(1, args.max_parallel), image_uri, args.role_arn,
        training_input, output_path, source_s3,
        args.instance_type, args.max_run, args.max_wait, use_spot, args.s3_bucket,
    )

    if args.no_wait:
        print(f"\nExperiment jobs submitted. Monitor in SageMaker console.")
        return

    # Wait for all (newly launched + already in-progress from resume)
    jobs_to_wait = newly_launched + [
        jn for jn in already_launched
        if jn not in newly_launched
    ]
    # For jobs already completed/failed, fetch their results directly
    results = {}
    for jn in already_launched:
        try:
            resp = sm.describe_training_job(TrainingJobName=jn)
            if resp["TrainingJobStatus"] in ("Completed", "Failed", "Stopped"):
                results[jn] = resp
                if jn in jobs_to_wait:
                    jobs_to_wait.remove(jn)
        except Exception:
            pass

    if jobs_to_wait:
        results.update(wait_for_jobs(sm, jobs_to_wait))

    completed = {jn: r for jn, r in results.items()
                 if r["TrainingJobStatus"] == "Completed"}

    failed = len(results) - len(completed)
    if failed:
        print(f"\nWARNING: {failed}/{total_jobs} job(s) failed.")
    if not completed:
        print("\nERROR: All experiment runs failed.")
        sys.exit(1)

    # ── Download ALL completed models as snapshots ───────────────────────
    from download_model import download_to_snapshot, download_and_extract
    print(f"\nDownloading {len(completed)} model(s) to snapshots...")
    model_metadata = {}  # job_name -> nn_metadata.json contents
    for jn, resp in completed.items():
        s3_uri = resp["ModelArtifacts"]["S3ModelArtifacts"]
        meta_info = job_meta[jn]
        label = f"experiment {meta_info['label']}"
        model_metadata[jn] = download_to_snapshot(s3_uri, jn, args.region, label)

    # ── Per-job results table (using downloaded metadata for accuracy) ───
    print(f"\n{'=' * 88}")
    print("ALL RUNS")
    print(f"{'=' * 88}")
    print(f"  {'Label':<28s} {'test_mae':>10s} {'train_mae':>10s} "
          f"{'test_r2':>10s} {'cv_mae':>10s}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    best_mae = min(
        m.get("test_mae", float("inf")) for m in model_metadata.values()
    )

    for jn in sorted(model_metadata, key=lambda j: job_meta[j]["label"]):
        m = model_metadata[jn]
        label = job_meta[jn]["label"]
        test_mae = m.get("test_mae")
        train_mae = m.get("train_mae")
        test_r2 = m.get("test_r2")
        cv_mae = m.get("cv_mae")

        cols = [f"  {label:<28s}"]
        cols.append(f"{test_mae:10.1f}" if test_mae is not None else f"{'—':>10s}")
        cols.append(f"{train_mae:10.1f}" if train_mae is not None else f"{'—':>10s}")
        cols.append(f"{test_r2:10.4f}" if test_r2 is not None else f"{'—':>10s}")
        cols.append(f"{cv_mae:10.1f}" if cv_mae is not None else f"{'—':>10s}")
        marker = " <-- best" if test_mae == best_mae else ""
        print(" ".join(cols) + marker)

    # ── Grouped summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 88}")
    print("SUMMARY (averaged across seeds)")
    print(f"{'=' * 88}")
    print(f"  {'test_size':>10s} {'mean_mae':>10s} "
          f"{'std_mae':>10s} {'best_mae':>10s} {'mean_r2':>10s} {'n':>5s}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*5}")

    combo_stats = {}
    for ts in ts_grid:
        maes, r2s = [], []
        for jn in model_metadata:
            meta_info = job_meta[jn]
            if meta_info["ts"] == ts:
                m = model_metadata[jn]
                if m.get("test_mae") is not None:
                    maes.append(m["test_mae"])
                if m.get("test_r2") is not None:
                    r2s.append(m["test_r2"])

        if not maes:
            continue

        mean_mae = statistics.mean(maes)
        std_mae = statistics.stdev(maes) if len(maes) > 1 else 0.0
        best_combo_mae = min(maes)
        mean_r2 = statistics.mean(r2s) if r2s else None
        combo_stats[ts] = {
            "test_size": ts,
            "mean_test_mae": mean_mae, "std_test_mae": std_mae,
            "best_test_mae": best_combo_mae,
            "mean_test_r2": mean_r2, "n_runs": len(maes),
        }

        r2_s = f"{mean_r2:10.4f}" if mean_r2 is not None else f"{'—':>10s}"
        print(f"  {ts:10.2f} {mean_mae:10.1f} "
              f"{std_mae:10.1f} {best_combo_mae:10.1f} {r2_s} {len(maes):5d}")

    print(f"{'=' * 88}")

    # ── Select best individual run (not mean across combos) ──────────────
    best_jn = min(
        model_metadata,
        key=lambda jn: model_metadata[jn].get("test_mae", float("inf")),
    )
    best_meta = model_metadata[best_jn]
    best_info = job_meta[best_jn]
    print(f"\nBest individual run: {best_jn}")
    print(f"  test_size={best_info['ts']}, "
          f"seed={best_info['seed']}, test_mae={best_meta['test_mae']:.1f}")

    # ── Per-spec MAE for best model ──────────────────────────────────────
    per_spec = best_meta.get("per_spec_mae", {})
    if per_spec:
        print(f"\nPer-spec MAE (best experiment model):")
        for spec, mae in sorted(per_spec.items(), key=lambda x: -x[1]):
            print(f"  {spec:<45s} {mae:8.1f}")

    # ── Save results locally ─────────────────────────────────────────────
    per_job_results = []
    for jn in sorted(model_metadata):
        m = model_metadata[jn]
        meta_info = job_meta[jn]
        per_job_results.append({
            "job_name": jn,
            "cv_folds": meta_info["cv"],
            "test_size": meta_info["ts"],
            "seed": meta_info["seed"],
            "test_mae": m.get("test_mae"),
            "train_mae": m.get("train_mae"),
            "test_r2": m.get("test_r2"),
            "cv_mae": m.get("cv_mae"),
            "per_spec_mae": m.get("per_spec_mae", {}),
        })

    experiment_data = {
        "timestamp": timestamp,
        "cv_folds": cv_folds,
        "test_size_grid": ts_grid,
        "seeds_per_combo": seeds_per,
        "total_jobs": total_jobs,
        "completed_jobs": len(completed),
        "failed_jobs": failed,
        "per_job_results": per_job_results,
        "combo_summary": list(combo_stats.values()),
        "best_run": {
            "job_name": best_jn,
            "test_size": best_info["ts"],
            "seed": best_info["seed"],
            "test_mae": best_meta.get("test_mae"),
        },
    }

    results_dir = os.path.join(PROJECT_ROOT, "local", "experiment_results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"experiment_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(experiment_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Update best_hyperparameters.json with winning run's settings ─────
    hp_path = args.hyperparams_file
    with open(hp_path) as f:
        hp = json.load(f)

    old_ts = hp.get("test_size")
    hp["test_size"] = best_info["ts"]

    with open(hp_path, "w") as f:
        json.dump(hp, f, indent=2)
        f.write("\n")

    print(f"\nUpdated {hp_path}:")
    if old_ts != best_info["ts"]:
        print(f"  test_size: {old_ts} -> {best_info['ts']}")
    else:
        print(f"  No change (test_size={best_info['ts']})")

    # ── Auto-promote if better than current production model ─────────────
    prod_meta_path = os.path.join(LOCAL_MODEL_DIR, "nn_metadata.json")
    prod_mae = None
    prod_meta = {}
    if os.path.exists(prod_meta_path):
        with open(prod_meta_path, encoding="utf-8") as f:
            prod_meta = json.load(f)
        prod_mae = prod_meta.get("test_mae")

    best_experiment_mae = best_meta.get("test_mae")

    if prod_mae is not None and best_experiment_mae is not None:
        print(f"\nProduction model test_mae: {prod_mae:.1f}")
        print(f"Best experiment test_mae:  {best_experiment_mae:.1f}")

        if best_experiment_mae < prod_mae:
            improvement = prod_mae - best_experiment_mae
            print(f"  Improvement: {improvement:.1f} — promoting to production!")

            # Per-spec comparison
            prod_per_spec = prod_meta.get("per_spec_mae", {})
            if per_spec and prod_per_spec:
                print(f"\n  {'Spec':<45s} {'Old':>8s} {'New':>8s} {'Delta':>8s}")
                print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8}")
                all_specs = sorted(set(prod_per_spec) | set(per_spec))
                for spec in all_specs:
                    old_v = prod_per_spec.get(spec)
                    new_v = per_spec.get(spec)
                    if old_v is not None and new_v is not None:
                        delta = new_v - old_v
                        sign = "+" if delta > 0 else ""
                        print(f"  {spec:<45s} {old_v:8.1f} {new_v:8.1f} {sign}{delta:7.1f}")

            from snapshot import snapshot_current_model
            snapshot_current_model(model_dir=LOCAL_MODEL_DIR, label=f"pre-experiment-{best_jn}")
            download_and_extract(
                completed[best_jn]["ModelArtifacts"]["S3ModelArtifacts"],
                LOCAL_MODEL_DIR, args.region,
            )
            print(f"\nProduction model updated to {best_jn}")
        else:
            print(f"  Current production model is equal or better — no change.")
    elif prod_mae is None:
        print(f"\nNo production model found — installing best experiment model.")
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        download_and_extract(
            completed[best_jn]["ModelArtifacts"]["S3ModelArtifacts"],
            LOCAL_MODEL_DIR, args.region,
        )
        print(f"\nProduction model set to {best_jn}")

    print(f"\nExperiment complete.")


if __name__ == "__main__":
    main()
