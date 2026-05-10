# Mr. Mythical: SimC Factory

Mr. Mythical: SimC Factory is an internal operations tool for producing, refreshing, and validating SimulationCraft training data at scale. It coordinates profile generation, AWS Batch simulation runs, result collection, model evaluation, and addon export from one controlled server-side workflow.

This is not a public web service. The FastAPI dashboard is an internal operator console for running simulations, inspecting model health, managing AWS-backed workflows, and preparing addon releases. The exported addon, Mr. Mythical: DPS Predictor, is the only artifact intended for end users.

## Project Goal

The goal is to keep the user-facing addon lightweight, current, and easy to validate while moving the expensive work into a repeatable internal pipeline.

That means this project is built to:

- scale SimulationCraft runs without running them on user machines
- keep cloud infrastructure and credentials away from addon users
- refresh stale or incomplete spec data with minimal manual work
- evaluate model quality before new addon data is exported
- make Mr. Mythical: DPS Predictor releases reproducible through export and offline parity checks

## What the Tool Does

- Runs high-volume SimulationCraft jobs across AWS Batch array workers.
- Generates and refreshes per-spec training data for DPS prediction.
- Tracks stale, incomplete, or underperforming specs so they can be regenerated.
- Evaluates model quality and supports local/offline addon parity checks.
- Exports trained model data into Lua tables consumed by Mr. Mythical: DPS Predictor.
- Provides an internal dashboard for operators to launch jobs, inspect pipelines, manage credentials, and review model/spec status.

## Intended Shape

The system is split into internal tooling and a user-facing addon:

- **Internal server or workstation:** runs the orchestrator, dashboard, model tooling, AWS polling, result merging, and addon export.
- **AWS Batch workers:** run only SimulationCraft plus a small shell worker, keeping cloud jobs lightweight and disposable.
- **S3:** acts as temporary transport for chunk inputs, manifests, and output zips.
- **Mr. Mythical: DPS Predictor:** ships the exported model data and prediction logic to users without exposing the dashboard, AWS infrastructure, or training workflow.

This keeps Python, pandas, numpy, boto3, training artifacts, credentials, and orchestration logic on a trusted internal host.

## Repository Map

### Internal orchestrator
- `local/sim_orchestrator_batch.py`
- `local/requirements.txt`

### Internal dashboard
- `run_web.py`
- `web/`

### Mr. Mythical: DPS Predictor export and validation
- `local/export_wow_addon.py`
- `local/test_addon_offline.py`
- `local/run_addon_offline_test.bat`

### Lightweight worker image
- `worker/batch_worker.sh`
- `worker/Dockerfile`

### Terraform
- `terraform/main.tf`
- `terraform/variables.tf`
- `terraform/outputs.tf`
- `terraform/terraform.tfvars.example`

## Simulation Workflow

For each spec:

1. the internal orchestrator determines the missing simulation range
2. it chunks the work exactly like your current distributed scheduler
3. it generates `.simc` files on the orchestrator host for each chunk
4. it zips and uploads each chunk to S3
5. it uploads one small manifest with S3 object keys for every chunk input/output object
6. it submits **one AWS Batch array job** for that spec
7. each Batch child uses `AWS_BATCH_JOB_ARRAY_INDEX` to pick its chunk from the manifest
8. the worker uses its AWS Batch job role to download only its chunk zip, runs SimulationCraft, and uploads only its JSON output zip
9. the internal orchestrator waits for the array parent to finish, downloads each shard, merges locally, and deletes temp S3 data

This keeps Python, pandas, numpy, boto3, and the orchestration logic entirely on the trusted internal host.

## 1. Build and push the worker image

The worker image is intentionally minimal:
- `simulationcraftorg/simc:latest`
- `bash`
- `curl`
- `awscli`
- `jq`
- `zip` / `unzip`

Build and push it to ECR:

```bash
docker build -t simc-batch-worker ./worker
aws ecr create-repository --repository-name simc-batch-worker
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=eu-north-1
ECR_REPO="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/simc-batch-worker"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
docker tag simc-batch-worker:latest "$ECR_REPO:latest"
docker push "$ECR_REPO:latest"
```

## 2. Deploy the Terraform stack

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars
terraform init
terraform apply
```

The important outputs are:
- `s3_bucket_name`
- `batch_job_queue_name`
- `batch_job_definition_name`
- `staleness_batch_job_queue_name`
- `staleness_batch_job_definition_name`

## 3. Install local Python dependencies

```bash
pip install -r local/requirements.txt
```

## 4. Run the internal orchestrator

Example:

```bash
python local/sim_orchestrator_batch.py \
  --specs MID1_Mage_Frost,MID1_Mage_Fire \
  --worker-parallel 4 \
  --s3-bucket simc-batch-bucket \
  --batch-job-queue simc-batch-array-queue \
  --batch-job-definition simc-batch-array-worker \
  --aws-region eu-north-1
```

## 4a. Run the dashboard

The dashboard binds to `127.0.0.1` by default, which is the recommended mode for a server reached through SSH tunneling, a private reverse proxy, or a service manager on the same host:

```bash
python run_web.py
```

For an internal server bind, set the host and disable reload. Keep it behind a private network, VPN, or reverse proxy, and configure Basic Auth at minimum:

```bash
SIMC_DASHBOARD_USERNAME=admin
SIMC_DASHBOARD_PASSWORD=change-me
SIMC_DASHBOARD_HOST=0.0.0.0
SIMC_DASHBOARD_PORT=8000
SIMC_DASHBOARD_RELOAD=false
python run_web.py
```

## 4b. Export and validate Mr. Mythical: DPS Predictor

Mr. Mythical: DPS Predictor is the only artifact intended for end users. Re-export it after model changes and run the offline parity check before packaging:

```bash
cd local
python export_wow_addon.py
python test_addon_offline.py
```

## 5. Run sample-based staleness checks on dedicated cloud capacity

Staleness checks now default to:
- sample-based detection
- remote execution on dedicated staleness Batch resources
- `5000` iterations per sampled point
- `10` sampled stat distributions from existing training CSVs

Example:

```bash
python local/sim_orchestrator_batch.py \
  --check-staleness \
  --staleness-method sample \
  --staleness-execution remote \
  --staleness-batch-job-queue simc-batch-array-staleness-queue \
  --staleness-batch-job-definition simc-batch-array-staleness-worker \
  --s3-bucket simc-batch-bucket \
  --aws-region eu-north-1
```

## Important behavior

- The orchestrator uploads chunk inputs and a manifest to S3.
- Workers use the AWS Batch job role to download manifests/inputs and upload output zips directly to S3.
- After the orchestrator downloads the results, it deletes:
  - the manifest object
  - all uploaded input zips
  - all uploaded output zips
- The S3 bucket remains for future runs.
- The Terraform lifecycle rule is only a safety net for orphaned temp files.
