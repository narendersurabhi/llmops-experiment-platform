#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${LLMOPS_NAMESPACE:-llmops}"
MINIO_SERVICE="${MINIO_SERVICE:-minio}"
MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://${MINIO_SERVICE}.${NAMESPACE}.svc.cluster.local:9000}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}"
MLFLOW_BUCKET_NAME="${MLFLOW_BUCKET_NAME:-mlflow-artifacts}"
MINIO_MC_IMAGE="${MINIO_MC_IMAGE:-minio/mc:latest}"
JOB_NAME="${MINIO_BOOTSTRAP_JOB_NAME:-minio-bootstrap-mlflow-artifacts}"
WAIT_TIMEOUT="${MINIO_WAIT_TIMEOUT:-300s}"
JOB_TIMEOUT="${MINIO_BOOTSTRAP_TIMEOUT:-120s}"

kubectl wait --for=condition=available --timeout="$WAIT_TIMEOUT" deployment/"$MINIO_SERVICE" -n "$NAMESPACE"
kubectl delete job "$JOB_NAME" -n "$NAMESPACE" --ignore-not-found
kubectl wait --for=delete --timeout=60s job/"$JOB_NAME" -n "$NAMESPACE" >/dev/null 2>&1 || true

kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
spec:
  backoffLimit: 1
  template:
    metadata:
      labels:
        app: ${JOB_NAME}
    spec:
      restartPolicy: Never
      containers:
        - name: bucket-bootstrap
          image: ${MINIO_MC_IMAGE}
          command: ["/bin/sh", "-c"]
          args:
            - |
              set -eu
              mc alias set local "${MINIO_ENDPOINT}" "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}"
              mc mb --ignore-existing "local/${MLFLOW_BUCKET_NAME}"
EOF

if ! kubectl wait --for=condition=complete --timeout="$JOB_TIMEOUT" job/"$JOB_NAME" -n "$NAMESPACE"; then
    kubectl describe job/"$JOB_NAME" -n "$NAMESPACE" || true
    kubectl logs job/"$JOB_NAME" -n "$NAMESPACE" || true
    exit 1
fi

kubectl logs job/"$JOB_NAME" -n "$NAMESPACE"
