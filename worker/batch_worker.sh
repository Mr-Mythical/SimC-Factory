#!/usr/bin/env bash
echo "[worker] manifest key: ${SIMC_MANIFEST_KEY:-unset}"

set -euo pipefail

: "${SIMC_S3_BUCKET:?SIMC_S3_BUCKET is required}"
: "${SIMC_MANIFEST_KEY:?SIMC_MANIFEST_KEY is required}"
ARRAY_INDEX="${AWS_BATCH_JOB_ARRAY_INDEX:-0}"
JOB_SCOPE="${AWS_BATCH_JOB_ID:-manual}-${ARRAY_INDEX}"
WORKDIR_ROOT="${SIMC_WORKDIR:-/tmp/simc-batch}"
WORKDIR="${WORKDIR_ROOT}/${JOB_SCOPE}"
PARALLEL="${SIMC_WORKER_PARALLEL:-4}"
AWS_REGION_OPT=()

if [[ -n "${SIMC_AWS_REGION:-}" ]]; then
  AWS_REGION_OPT=(--region "$SIMC_AWS_REGION")
fi

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
MANIFEST_PATH="$WORKDIR/manifest.json"
INPUT_ZIP="$WORKDIR/input.zip"
OUTPUT_ZIP="$WORKDIR/output.zip"
CHUNK_DIR="$WORKDIR/chunk"
mkdir -p "$CHUNK_DIR"

echo "[worker] downloading manifest for array index ${ARRAY_INDEX}"
aws s3api get-object "${AWS_REGION_OPT[@]}" \
  --bucket "$SIMC_S3_BUCKET" \
  --key "$SIMC_MANIFEST_KEY" \
  "$MANIFEST_PATH" >/dev/null

read_manifest_value() {
  local jq_expr="$1"
  jq -r "$jq_expr" "$MANIFEST_PATH"
}

CHUNK_COUNT="$(read_manifest_value '.chunk_count')"
if [[ "$ARRAY_INDEX" =~ ^[0-9]+$ ]] && [[ "$ARRAY_INDEX" -ge "$CHUNK_COUNT" ]]; then
  echo "[worker] array index ${ARRAY_INDEX} >= chunk_count ${CHUNK_COUNT}; nothing to do"
  exit 0
fi

INPUT_KEY="$(read_manifest_value ".chunks[${ARRAY_INDEX}].input_key")"
OUTPUT_KEY="$(read_manifest_value ".chunks[${ARRAY_INDEX}].output_key")"
SPEC_NAME="$(read_manifest_value '.spec')"

echo "[worker] spec=${SPEC_NAME} array_index=${ARRAY_INDEX} parallel=${PARALLEL}"

echo "[worker] downloading input chunk"
aws s3api get-object "${AWS_REGION_OPT[@]}" \
  --bucket "$SIMC_S3_BUCKET" \
  --key "$INPUT_KEY" \
  "$INPUT_ZIP" >/dev/null

unzip -q "$INPUT_ZIP" -d "$CHUNK_DIR"
find "$CHUNK_DIR" -maxdepth 1 -name '*.json' -delete || true

SIMC_FILES=$(find "$CHUNK_DIR" -maxdepth 1 -name '*.simc' | wc -l | tr -d ' ')
if [[ "$SIMC_FILES" == "0" ]]; then
  echo "[worker] no .simc files found after extraction"
  exit 1
fi

SIMC_BIN="${SIMC_BIN:-}"

if [[ -z "$SIMC_BIN" ]]; then
  if command -v simc >/dev/null 2>&1; then
    SIMC_BIN="$(command -v simc)"
  else
    SIMC_BIN="$(find / -type f -name simc 2>/dev/null | head -n 1 || true)"
  fi
fi

if [[ -z "$SIMC_BIN" ]]; then
  echo "[worker] could not find simc binary in container"
  exit 1
fi

echo "[worker] using simc binary: $SIMC_BIN"
echo "[worker] running ${SIMC_FILES} simulations"

(
  cd "$CHUNK_DIR"
  find . -maxdepth 1 -name '*.simc' -print0 \
    | xargs -0 -I{} -P "$PARALLEL" sh -c '"$1" "$2"' _ "$SIMC_BIN" "{}"
)
JSON_FILES=$(find "$CHUNK_DIR" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')
if [[ "$JSON_FILES" == "0" ]]; then
  echo "[worker] no JSON outputs were produced"
  exit 1
fi

echo "[worker] zipping ${JSON_FILES} JSON outputs"
(
  cd "$CHUNK_DIR"
  zip -q -j "$OUTPUT_ZIP" ./*.json
)

echo "[worker] uploading result zip"
aws s3api put-object "${AWS_REGION_OPT[@]}" \
  --bucket "$SIMC_S3_BUCKET" \
  --key "$OUTPUT_KEY" \
  --body "$OUTPUT_ZIP" \
  --content-type application/zip >/dev/null

echo "[worker] done"
