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
make infra-apply
```

(Optional) Apply Kubeflow-specific resources:

```bash
make infra-apply-kubeflow
```

Verify resources:

```bash
make k8s-status
```

## 4) Build Docker Images

```bash
make docker-build
```

## 5) Compile Pipelines

```bash
make pipelines-compile
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
make mlflow-port-forward
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


## 11) Deployment Pipeline (GitHub Actions)

A GitHub Actions workflow is available at `.github/workflows/deploy-platform.yml`.

- Trigger: **manual** (`workflow_dispatch`)
- Job 1 (`validate-and-compile`): installs dependencies, runs `pytest -q`, compiles pipelines.
- Job 2 (`deploy-infra`): optionally applies infra manifests with `scripts/deploy/apply_infra.sh`.

Required GitHub secret for deploy step:

- `KUBE_CONFIG_DATA` (base64-encoded kubeconfig)

## 12) Trigger Training and Evaluation Jobs

### Option A: Kubeflow UI

1. Port-forward Kubeflow Pipelines endpoint/UI based on your install.
2. Upload compiled pipelines:
   - `pipelines/training/compiled/training_pipeline.yaml`
   - `pipelines/evaluation/compiled/evaluation_pipeline.yaml`
3. Start runs with the runtime arguments described below.

### Option B: Python submit scripts (recommended)

Set your KFP endpoint:

```bash
export KFP_HOST="http://localhost:8080"
```

Submit training run:

```bash
make submit-training \
  RUN_CONFIG_PATH=configs/training/run.sample.json \
  DATASET_METADATA_PATH=dataset/metadata.sample.json \
  KFP_HOST=http://localhost:8080
```

Submit evaluation run:

```bash
make submit-evaluation \
  EVAL_CONFIG_PATH=configs/evaluation/eval.sample.json \
  CANDIDATE_MODEL_URI=models:/candidate-llm/1 \
  BASELINE_MODEL_URI=models:/baseline-llm/1 \
  KFP_HOST=http://localhost:8080
```

### Runtime arguments reference

Training pipeline:
- `run_config_path`
- `dataset_metadata_path`

Evaluation pipeline:
- `eval_config_path`
- `candidate_model_uri`
- `baseline_model_uri`

## 13) Can experiments be configured from UI?

Short answer: **partially**.

- **MLflow UI**: best for *tracking and analysis* (metrics, params, artifacts, model versions). It does **not** author or launch Kubeflow pipelines. You can compare and inspect runs here, but pipeline runtime inputs still come from run submissions.
- **Kubeflow Pipelines UI**: yes, you can configure run arguments at submission time (for example `run_config_path`, `dataset_metadata_path`, `eval_config_path`, and model URIs) and launch runs directly from the UI.
- For reproducibility, keep canonical experiment definitions in versioned JSON config files under `configs/` and submit runs from those paths (UI or script).
