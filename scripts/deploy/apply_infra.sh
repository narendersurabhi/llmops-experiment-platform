#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${LLMOPS_NAMESPACE:-llmops}"
WAIT_TIMEOUT="${INFRA_WAIT_TIMEOUT:-300s}"
BOOTSTRAP_MINIO_BUCKET="${BOOTSTRAP_MINIO_BUCKET:-true}"

kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/postgres/postgres.yaml
kubectl apply -f infra/minio/minio.yaml
kubectl apply -f infra/mlflow/mlflow.yaml

kubectl wait --for=condition=available --timeout="$WAIT_TIMEOUT" deployment/postgres -n "$NAMESPACE"
kubectl wait --for=condition=available --timeout="$WAIT_TIMEOUT" deployment/minio -n "$NAMESPACE"

if [[ "$BOOTSTRAP_MINIO_BUCKET" == "true" ]]; then
    bash scripts/deploy/bootstrap_minio_bucket.sh
fi

kubectl wait --for=condition=available --timeout="$WAIT_TIMEOUT" deployment/mlflow -n "$NAMESPACE"
