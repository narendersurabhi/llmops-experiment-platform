#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/postgres/postgres.yaml
kubectl apply -f infra/minio/minio.yaml
kubectl apply -f infra/mlflow/mlflow.yaml
