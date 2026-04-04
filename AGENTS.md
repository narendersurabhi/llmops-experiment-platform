# AGENTS Operational Log

## Purpose
This file tracks repository changes and current status so future agents can quickly understand what has been implemented.

## Current Repository Status (2026-04-04)
- Bootstrapped a production-style local-first LLMOps repository for simulated LLM training/evaluation on Kubernetes.
- Implemented decoupled training and evaluation simulation components with MLflow logging.
- Implemented KFP v2 pipelines for training and evaluation.
- Added Dockerfiles for base, trainer, evaluator images.
- Added Kubernetes manifests for namespace, PostgreSQL, MinIO, and MLflow.
- Added JSON schemas for dataset/run/eval/scorecard contracts.
- Added helper scripts for pipeline compilation and infra deployment.
- Added unit tests and pipeline compile smoke tests.
- Added full README with architecture, setup, deployment, usage, and outputs.
- Added a GitHub Actions deployment pipeline to validate/compile and optionally deploy infra.
- Added KFP submission scripts and docs for manually triggering training and evaluation runs.

## Change Log
### 2026-04-04
- Created required repo structure under:
  - `configs/`, `training/`, `evaluation/`, `dataset/`, `pipelines/`, `components/`, `docker/`, `infra/`, `schemas/`, `scripts/`, `tests/`.
- Added simulation code:
  - `components/trainer/simulate_training.py`
  - `components/evaluator/simulate_evaluation.py`
  - `components/utils/io_utils.py`
  - `components/utils/mlflow_utils.py`
- Added KFP pipelines:
  - `pipelines/training/training_pipeline.py`
  - `pipelines/evaluation/evaluation_pipeline.py`
- Added infra manifests:
  - `infra/k8s/namespace.yaml`
  - `infra/postgres/postgres.yaml`
  - `infra/minio/minio.yaml`
  - `infra/mlflow/mlflow.yaml`
- Added configs/dataset samples and schemas.
- Added automation scripts and tests.
- Added top-level `README.md` and `requirements.txt`.
- Added deployment workflow:
  - `.github/workflows/deploy-platform.yml`
- Added KFP run submission scripts:
  - `scripts/pipelines/submit_training_run.py`
  - `scripts/pipelines/submit_evaluation_run.py`
- Expanded user docs for deployment automation and run triggering:
  - `README.md`
  - `training/README.md`
  - `evaluation/README.md`
- Added a root `Makefile` with local developer workflows for:
  - dependency install, tests, and pipeline compile
  - Docker image builds
  - local Kubernetes infra apply/status
  - MLflow port-forward
  - KFP training/evaluation run submission with parameterized make targets
- Updated `README.md` to use `make` commands for local deployment/build/compile/run tasks.
- Documented UI capability boundaries:
  - MLflow UI for experiment tracking/inspection
  - Kubeflow UI for runtime parameter configuration and run launch

## Notes for Future Agents
- This repo intentionally uses simulated metrics/checkpoints/benchmarks for platform validation.
- MLflow connectivity depends on env vars (`MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT_URL`, S3 credentials).
- Before running tests, install dependencies from `requirements.txt`.
- Added `tests/conftest.py` to ensure repository root is available on `PYTHONPATH` during test discovery.
- Updated tests to gracefully skip when optional dependencies (`mlflow`, `kfp`) are unavailable in offline/sandboxed environments.
- Deployment automation via GitHub Actions requires `KUBE_CONFIG_DATA` secret (base64 kubeconfig) for the deploy step.
- KFP submission scripts require `KFP_HOST` or `--host` and compiled pipeline YAML packages.
