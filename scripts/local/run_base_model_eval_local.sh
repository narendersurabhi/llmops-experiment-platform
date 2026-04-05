#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B}"
MODEL_URI="${MODEL_URI:-hf://${MODEL_ID}}"
EVAL_CONFIG_PATH="${EVAL_CONFIG_PATH:-$REPO_ROOT/configs/evaluation/qwen2_5_0_5b.public.single_model.sample.json}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-}"
HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/hf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
TORCH_THREADS="${TORCH_THREADS:-2}"
DEVICE="${DEVICE:-auto}"

cd "$REPO_ROOT"

CMD=(
  uv run python scripts/local/run_single_model_eval_local.py
  --eval-config-path "$EVAL_CONFIG_PATH"
  --model-id "$MODEL_ID"
  --model-uri "$MODEL_URI"
  --hf-home "$HF_HOME"
  --host "$HOST"
  --port "$PORT"
  --server-start-timeout "$SERVER_START_TIMEOUT"
  --torch-threads "$TORCH_THREADS"
  --device "$DEVICE"
)

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--output-dir "$OUTPUT_DIR")
fi

if [[ -n "$MLFLOW_TRACKING_URI" ]]; then
  CMD+=(--mlflow-tracking-uri "$MLFLOW_TRACKING_URI")
fi

PYTHONPATH=. "${CMD[@]}"
