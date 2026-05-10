#!/usr/bin/env bash
set -euo pipefail

# Rebuild the SimC Batch worker Docker image and push to ECR.
# This ensures the SimC binary stays in sync with the latest profiles
# fetched from the midnight branch HEAD.
#
# Environment variables (all have sensible defaults):
#   ECR_REPO              — full ECR repository URI (without tag)
#   AWS_ACCOUNT_ID        — AWS account ID used when ECR_REPO is not set
#   ECR_REPOSITORY_NAME   — ECR repository name when ECR_REPO is not set
#   AWS_REGION            — AWS region for ECR login
#   WORKER_DIR            — path to the worker/ directory containing Dockerfile

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-eu-north-1}}"
ECR_REPOSITORY_NAME="${ECR_REPOSITORY_NAME:-simc-batch-worker}"
WORKER_DIR="${WORKER_DIR:-$PROJECT_ROOT/worker}"

if [[ -z "${ECR_REPO:-}" ]]; then
  if [[ -z "${AWS_ACCOUNT_ID:-}" ]]; then
    AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION")"
  fi
  ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY_NAME}"
fi

ECR_REGISTRY="${ECR_REPO%/*}"
LOCAL_TAG="simc-batch-worker:latest"
REMOTE_TAG="${ECR_REPO}:latest"

echo "=== SimC Batch Worker — ECR Image Rebuild ==="
echo "  ECR repo:   $ECR_REPO"
echo "  Region:     $AWS_REGION"
echo "  Dockerfile: $WORKER_DIR/Dockerfile"
echo ""

# 1. Pull the latest upstream SimC image (bypass Docker cache)
echo "--- Step 1/5: Pulling simulationcraftorg/simc:latest ---"
docker pull simulationcraftorg/simc:latest
echo ""

# 2. Build worker image with --no-cache to pick up the fresh base
echo "--- Step 2/5: Building worker image ---"
docker build --pull --no-cache -t "$LOCAL_TAG" "$WORKER_DIR"
echo ""

# 3. Print the SimC version baked into the new image
echo "--- Step 3/5: Verifying SimC version ---"
MSYS_NO_PATHCONV=1 docker run --rm --entrypoint /usr/local/bin/simc "$LOCAL_TAG" --version 2>&1 || true
echo ""

# 4. Log in to ECR
echo "--- Step 4/5: Logging in to ECR ---"
# Write ECR auth directly into a temp config — bypasses docker login
# and the Windows credential store entirely.
DOCKER_CFG_DIR=$(mktemp -d)
trap 'rm -rf "$DOCKER_CFG_DIR"' EXIT
export DOCKER_CONFIG="$DOCKER_CFG_DIR"
ECR_PASSWORD=$(aws ecr get-login-password --region "$AWS_REGION")
AUTH_TOKEN=$(printf 'AWS:%s' "$ECR_PASSWORD" | base64 -w0)
cat > "$DOCKER_CFG_DIR/config.json" <<ENDCFG
{"auths":{"$ECR_REGISTRY":{"auth":"$AUTH_TOKEN"}}}
ENDCFG
echo "Login Succeeded (token written to temp config)"
echo ""

# 5. Tag and push
echo "--- Step 5/5: Pushing to ECR ---"
docker tag "$LOCAL_TAG" "$REMOTE_TAG"
docker push "$REMOTE_TAG"
echo ""

# Print final digest
DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "$REMOTE_TAG" 2>/dev/null || echo "unknown")
echo "=== Done ==="
echo "  Pushed:  $REMOTE_TAG"
echo "  Digest:  $DIGEST"
echo ""
echo "Note: Existing Batch jobs will pick up the new image on their next run."
echo "No Terraform changes needed — the job definition already references :latest."
