#!/usr/bin/env bash
set -euo pipefail

ROLE=""
MODEL_ID=""
MODEL_URI="${MODEL_URI:-}"
LOCAL_MODEL_SOURCE_PATH="${LOCAL_MODEL_PATH:-}"
LOCAL_MODEL_MOUNT_PATH="${LOCAL_MODEL_MOUNT_PATH:-/models/local-model}"
NAMESPACE="${LLMOPS_NAMESPACE:-llmops}"
IMAGE="${QWEN_SERVICE_IMAGE:-llmops/qwen-service:dev}"
WAIT_TIMEOUT="${INFERENCE_WAIT_TIMEOUT:-1800s}"
PYTHON_BIN="${PYTHON:-python}"
KUBECTL_BIN="${KUBECTL:-kubectl}"
REPLICAS="${QWEN_REPLICAS:-1}"
MAX_NEW_TOKENS="${QWEN_DEFAULT_MAX_NEW_TOKENS:-192}"
TEMPERATURE="${QWEN_DEFAULT_TEMPERATURE:-0.2}"
TORCH_THREADS="${QWEN_TORCH_THREADS:-2}"
CPU_REQUEST="${QWEN_CPU_REQUEST:-1000m}"
MEMORY_REQUEST="${QWEN_MEMORY_REQUEST:-3Gi}"
MEMORY_LIMIT="${QWEN_MEMORY_LIMIT:-8Gi}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --role)
            ROLE="$2"
            shift 2
            ;;
        --model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        --model-uri)
            MODEL_URI="$2"
            shift 2
            ;;
        --local-model-source-path)
            LOCAL_MODEL_SOURCE_PATH="$2"
            shift 2
            ;;
        --local-model-mount-path)
            LOCAL_MODEL_MOUNT_PATH="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$ROLE" ]]; then
    echo "--role is required (baseline or candidate)" >&2
    exit 1
fi

if [[ -z "$MODEL_ID" && -z "$LOCAL_MODEL_SOURCE_PATH" ]]; then
    echo "Either --model-id or --local-model-source-path is required" >&2
    exit 1
fi

ARGS=(
    scripts/deploy/render_qwen_manifest.py
    --role "$ROLE"
    --namespace "$NAMESPACE"
    --image "$IMAGE"
    --replicas "$REPLICAS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --torch-threads "$TORCH_THREADS"
    --cpu-request "$CPU_REQUEST"
    --memory-request "$MEMORY_REQUEST"
    --memory-limit "$MEMORY_LIMIT"
)

if [[ -n "$MODEL_ID" ]]; then
    ARGS+=(--model-id "$MODEL_ID")
fi

if [[ -n "$MODEL_URI" ]]; then
    ARGS+=(--model-uri "$MODEL_URI")
fi

if [[ -n "$LOCAL_MODEL_SOURCE_PATH" ]]; then
    ARGS+=(--local-model-source-path "$LOCAL_MODEL_SOURCE_PATH")
    ARGS+=(--local-model-mount-path "$LOCAL_MODEL_MOUNT_PATH")
fi

"$PYTHON_BIN" "${ARGS[@]}" | "$KUBECTL_BIN" apply -f -
"$KUBECTL_BIN" rollout status "deployment/qwen-${ROLE}" -n "$NAMESPACE" --timeout="$WAIT_TIMEOUT"
