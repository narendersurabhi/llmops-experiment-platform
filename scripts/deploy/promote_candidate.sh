#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${LLMOPS_NAMESPACE:-llmops}"
CANDIDATE_DEPLOYMENT="${CANDIDATE_DEPLOYMENT_NAME:-qwen-candidate}"
IMAGE="${QWEN_SERVICE_IMAGE:-}"
MODEL_ID="${MODEL_ID:-}"
LOCAL_MODEL_SOURCE_PATH="${LOCAL_MODEL_PATH:-}"
LOCAL_MODEL_MOUNT_PATH="${LOCAL_MODEL_MOUNT_PATH:-}"
KUBECTL_BIN="${KUBECTL:-kubectl}"

if [[ -z "$MODEL_ID" ]]; then
    MODEL_ID="$("$KUBECTL_BIN" get deployment "$CANDIDATE_DEPLOYMENT" -n "$NAMESPACE" -o jsonpath="{.spec.template.spec.containers[0].env[?(@.name=='QWEN_MODEL_ID')].value}")"
fi

if [[ -z "$IMAGE" ]]; then
    IMAGE="$("$KUBECTL_BIN" get deployment "$CANDIDATE_DEPLOYMENT" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}')"
fi

if [[ -z "$LOCAL_MODEL_SOURCE_PATH" ]]; then
    LOCAL_MODEL_SOURCE_PATH="$("$KUBECTL_BIN" get deployment "$CANDIDATE_DEPLOYMENT" -n "$NAMESPACE" -o jsonpath="{.metadata.annotations['llmops/local-model-source-path']}")"
fi

if [[ -z "$LOCAL_MODEL_MOUNT_PATH" ]]; then
    LOCAL_MODEL_MOUNT_PATH="$("$KUBECTL_BIN" get deployment "$CANDIDATE_DEPLOYMENT" -n "$NAMESPACE" -o jsonpath="{.metadata.annotations['llmops/local-model-mount-path']}")"
fi

if [[ -z "$MODEL_ID" ]]; then
    echo "Failed to resolve candidate model ID from deployment/$CANDIDATE_DEPLOYMENT" >&2
    exit 1
fi

ARGS=(
    scripts/deploy/apply_qwen_inference.sh
    --role baseline \
    --namespace "$NAMESPACE" \
    --image "$IMAGE"
)

if [[ -n "$LOCAL_MODEL_SOURCE_PATH" ]]; then
    ARGS+=(--local-model-source-path "$LOCAL_MODEL_SOURCE_PATH")
    if [[ -n "$LOCAL_MODEL_MOUNT_PATH" ]]; then
        ARGS+=(--local-model-mount-path "$LOCAL_MODEL_MOUNT_PATH")
    fi
else
    ARGS+=(--model-id "$MODEL_ID")
fi

bash "${ARGS[@]}"
