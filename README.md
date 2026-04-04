# LLM Ops Platform (Local-First)

A production-style, local-first LLMOps repository that validates **training** and **evaluation** pipelines end-to-end on Kubernetes using:

- **Kubeflow Pipelines v2** for orchestration
- **MLflow** for tracking, artifacts, and model registry
- **MinIO** for S3-compatible artifact storage
- **PostgreSQL** for MLflow backend metadata

> This repository intentionally uses **simulation logic** (fake training/evaluation outputs) to validate platform wiring before introducing real model workloads.

## 1) Architecture Overview

### Pipelines

1. **Training Pipeline** (`pipelines/training/training_pipeline.py`)
   - Resolve config
   - Validate inputs
   - Load dataset metadata
   - Tokenization check (simulated)
   - Launch training job (simulated)
   - Periodic validation (simulated)
   - Select best checkpoint
   - Log to MLflow

2. **Evaluation Pipeline** (`pipelines/evaluation/evaluation_pipeline.py`)
   - Resolve eval config
   - Load candidate + baseline
   - Intrinsic evaluation (simulated perplexity)
   - Public benchmark (mock MMLU)
   - Domain benchmark (mock)
   - Aggregate results
   - Compare vs baseline
   - Generate scorecard
   - Log to MLflow

### Runtime

- Local Kubernetes: Docker Desktop Kubernetes
- CPU-only execution
- Artifacts stored in MinIO buckets
- MLflow metadata in PostgreSQL
- KFP v2 components run as container components

## 2) Local Setup Instructions

### Prerequisites

- Docker Desktop with Kubernetes enabled
- Python 3.11+
- `kubectl`
- `kfp` SDK

### Python setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Deploy Infrastructure to Kubernetes

```bash
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/postgres/postgres.yaml
kubectl apply -f infra/minio/minio.yaml
kubectl apply -f infra/mlflow/mlflow.yaml
```

(Optional) Apply Kubeflow-specific resources:

```bash
kubectl apply -f infra/kubeflow/
```

## 4) Build Docker Images

```bash
docker build -f docker/base/Dockerfile -t llmops/base:latest .
docker build -f docker/trainer/Dockerfile -t llmops/trainer:latest .
docker build -f docker/evaluator/Dockerfile -t llmops/evaluator:latest .
```

## 5) Compile Pipelines

```bash
python pipelines/training/training_pipeline.py
python pipelines/evaluation/evaluation_pipeline.py
```

Compiled pipeline YAMLs are generated in `pipelines/training/compiled/` and `pipelines/evaluation/compiled/`.

## 6) Run Training Pipeline

Use Kubeflow Pipelines UI/SDK and submit `training_pipeline.yaml` with runtime parameters:

- `run_config_path` (e.g., `configs/training/run.sample.json`)
- `dataset_metadata_path` (e.g., `dataset/metadata.sample.json`)

The pipeline writes intermediate artifacts to mounted paths and logs params/metrics/artifacts to MLflow.

## 7) Run Evaluation Pipeline

Submit `evaluation_pipeline.yaml` with parameters:

- `eval_config_path` (e.g., `configs/evaluation/eval.sample.json`)
- `candidate_model_uri`
- `baseline_model_uri`

Pipeline generates scorecard JSON and logs to MLflow.

## 8) MLflow UI Usage

If MLflow service is exposed internally:

```bash
kubectl port-forward -n llmops svc/mlflow 5000:5000
```

Open: `http://localhost:5000`

Experiment naming convention:

- `llmops-training-{project}`
- `llmops-evaluation-{project}`

Run tags:

- `dataset_version`
- `config_version`
- `pipeline_type`
- `run_id`

## 9) Example Outputs

- Fake checkpoints: `checkpoint_epoch_*.ckpt`
- Training summary: `training_summary.json`
- Eval scorecard: `scorecard.json`
- MLflow metrics:
  - training: `train_loss`, `val_loss`, `tokens_per_sec`
  - evaluation: `intrinsic_perplexity`, `public_benchmark_score`, `domain_benchmark_score`, `overall_score`

## 10) Testing

```bash
pytest -q
```

Includes:

- unit tests for trainer/evaluator simulation logic
- pipeline compile smoke tests
