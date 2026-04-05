# LLM Ops Platform (Local-First)

A production-style, local-first LLMOps repository that validates **training** and **evaluation** pipelines end-to-end on Kubernetes using:

- **Kubeflow Pipelines v2** for orchestration
- **MLflow** for tracking, artifacts, and model registry
- **MinIO** for S3-compatible artifact storage
- **PostgreSQL** for MLflow backend metadata

> This repository now supports a **continued pretraining** path for the training pipeline using raw text splits and causal language modeling. The default local sample uses a tiny Hugging Face model for CPU-only validation on Docker Desktop. Real Qwen continued pretraining still needs a proper GPU-backed cluster.

For the end-to-end workflow from base-model evaluation through continued pretraining and post-training re-evaluation, see [docs/base-model-cpt-workflow.md](docs/base-model-cpt-workflow.md).

## 1) Architecture Overview

### Pipelines

1. **Training Pipeline** (`pipelines/training/training_pipeline.py`)
   - Resolve config
   - Validate inputs
   - Load dataset metadata
   - Tokenization check
   - Launch continued pretraining job
   - Validate on held-out text
   - Evaluate on test text
   - Select best checkpoint
   - Log to MLflow

2. **Evaluation Pipeline** (`pipelines/evaluation/evaluation_pipeline.py`)
   - Resolve eval config
   - Assume candidate and baseline endpoints are already deployed
   - Call local candidate and baseline inference services over HTTP
   - Score public prompts from real model responses for both models
   - Score domain prompts from real model responses for both models
   - Derive an intrinsic response-quality proxy for both models
   - Aggregate candidate and baseline results
   - Compute deltas and winner
   - Generate scorecard and model cards
   - Log metrics and artifacts to MLflow

3. **Single-Model Evaluation Pipeline** (`pipelines/evaluation/single_model_evaluation_pipeline.py`)
   - Resolve eval config
   - Call one deployed inference endpoint over HTTP
   - Score public and domain prompts from real model responses
   - Derive an intrinsic response-quality proxy
   - Generate `scorecard.json` and `model_card.md`
   - Log metrics and artifacts to MLflow

### Runtime

- Local Kubernetes: Docker Desktop Kubernetes
- CPU-only execution
- Tiny-model continued pretraining validation on local CPU
- Separate stable baseline and candidate inference endpoints
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
uv sync --extra dev
```

Use `uv sync --all-extras` if you want the full local host-side stack, including trainer and inference dependencies.

To launch a simple local-only UI for running training and evaluation jobs on your MacBook instead of Kubernetes:

```bash
make run-local-ui
```

That UI wraps the existing local entrypoints:

- local training via `components.trainer.run_pretraining`
- local single-model eval via `scripts/local/run_single_model_eval_local.py`

## 3) Deploy Infrastructure to Kubernetes

```bash
make infra-apply
```

This target now:

- applies the `llmops` namespace, PostgreSQL, MinIO, and MLflow manifests
- waits for PostgreSQL and MinIO to become available
- bootstraps the `mlflow-artifacts` bucket inside MinIO
- waits for MLflow to become available

Install standalone Kubeflow Pipelines:

```bash
make infra-apply-kubeflow
```

On Docker Desktop Kubernetes for Apple Silicon, the repository defaults to a local compatibility mode that:

- disables `proxy-agent` and `metadata-writer`, because those upstream images may not publish usable arm64 variants
- patches the `seaweedfs` service to expose port `9000 -> 8333`, which avoids local artifact upload failures from KFP launcher pods

This does not block local port-forwarded use of the KFP UI/API.

Port-forward the KFP UI/API and set the SDK endpoint:

```bash
make kfp-port-forward
export KFP_HOST="http://localhost:3000"
```

Verify resources:

```bash
make k8s-status
make kfp-status
```

Deploy the stable baseline endpoint first:

```bash
make baseline-apply MODEL_ID=Qwen/Qwen2.5-0.5B
```

`make base-model-apply ...` is an alias for the same stable endpoint flow.

Deploy a candidate endpoint when you have a new model to compare:

```bash
make candidate-apply MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
```

The first startup may take several minutes because the pods download the selected model from Hugging Face and load it on CPU.
`qwen-inference` is the stable service alias for the baseline deployment. The evaluation pipeline compares `qwen-candidate` against `qwen-baseline`.

If you want to serve a checkpoint directory instead of downloading from Hugging Face, mount it directly:

```bash
make base-model-apply \
  LOCAL_MODEL_PATH=/absolute/path/to/model-dir \
  MODEL_URI=models:/base-model/local
```

When `LOCAL_MODEL_PATH` is set, the pod mounts that directory read-only and loads the model from the mounted path instead of `MODEL_ID`. The path must be visible to the Kubernetes node filesystem.

For continued pretraining, the sample config in `configs/training/run.sample.json` uses `sshleifer/tiny-gpt2` because realistic Qwen continued pretraining is not practical on Docker Desktop CPU nodes. If you move to a real GPU cluster, replace `base_model_id` and `tokenizer_name` with your Qwen checkpoint.

## 4) Build Docker Images

```bash
make docker-build
```

The compiled pipeline specs reference these images by default:

- `llmops/base:dev`
- `llmops/trainer:dev`
- `llmops/evaluator:dev`
- `llmops/qwen-service:dev`

The Kubernetes MLflow deployment also expects:

- `llmops/mlflow:dev`

If your Kubernetes cluster pulls from a registry instead of a local Docker daemon, set `LLMOPS_BASE_IMAGE`, `LLMOPS_TRAINER_IMAGE`, and `LLMOPS_EVALUATOR_IMAGE` before compiling the pipelines.
For Docker Desktop Kubernetes, the default `:dev` tags are intentional so Kubernetes uses the locally built images with `IfNotPresent` semantics instead of trying to pull `:latest` from a registry.

## 5) Compile Pipelines

```bash
make pipelines-compile
```

Compiled pipeline YAMLs are generated in `pipelines/training/compiled/` and `pipelines/evaluation/compiled/`.
The training pipeline runs the custom trainer container for continued pretraining, the comparison evaluation pipeline runs against already-deployed candidate and baseline endpoints, and the single-model evaluation pipeline runs against one deployed endpoint.

## 6) Run Training Pipeline

Use Kubeflow Pipelines UI/SDK and submit `training_pipeline.yaml` with runtime parameters:

- `run_config_path` (e.g., `configs/training/run.sample.json`)
- `dataset_metadata_path` (e.g., `dataset/metadata.sample.json`)

The pipeline writes intermediate artifacts to mounted paths and logs params/metrics/artifacts to MLflow.

The training dataset manifest points to raw text corpus splits:

- `train`
- `validation`
- `test`

If your actual corpus uses `eval` instead of `validation`, that alias is accepted by the local trainer and dataset validator.

The sample manifest is `dataset/metadata.sample.json`, and the sample corpus files live under `dataset/cpt/`.
The recommended flow is to treat training as artifact production, then deploy a chosen model behind the candidate endpoint before running evaluation. The local Docker Desktop path still uses manually selected Hugging Face model IDs for serving rather than auto-serving the training output.

## 7) Run Evaluation Pipeline

Submit `evaluation_pipeline.yaml` with parameters:

- `eval_config_path` (e.g., `configs/evaluation/eval.sample.json`)
- `candidate_model_uri`
- `baseline_model_uri`

Pipeline generates a comparison scorecard plus model-card artifacts from candidate and baseline prompt responses, then logs metrics/artifacts to MLflow.

Recommended order:

1. Run training and review the resulting MLflow artifacts.
2. Deploy or refresh the stable baseline endpoint with `make baseline-apply ...`.
3. Deploy or refresh the candidate endpoint with `make candidate-apply ...`.
4. Submit the evaluation pipeline to compare candidate vs baseline.
5. If the candidate wins, promote it with `make promote-candidate`.

## 8) Run Single-Model Evaluation Pipeline

Submit `single_model_evaluation_pipeline.yaml` with parameters:

- `eval_config_path` (e.g., `configs/evaluation/single_model.sample.json`)
- `model_uri`

This path produces a standalone `scorecard.json` and `model_cards/model_card.md` for one served model, with no baseline or comparison fields.

For host-local evaluation on a MacBook instead of Docker/Kubernetes, use `scripts/local/run_single_model_eval_local.py`.

To validate your real dataset manifest and golden set before training or evaluation:

```bash
make validate-local-data \
  DATASET_METADATA_PATH=path/to/your/metadata.json \
  GOLDEN_SET_PATH=path/to/your/golden.json
```

To build a real public-source CPT corpus from FineWeb-Edu plus arXiv:

```bash
uv sync --extra dev --extra data
make build-public-data
```

This writes:

- `dataset/cpt/qwen2_5_0_5b_public/train.jsonl`
- `dataset/cpt/qwen2_5_0_5b_public/eval.jsonl`
- `dataset/cpt/qwen2_5_0_5b_public/test.jsonl`
- `dataset/golden/qwen2_5_0_5b_public.golden.json`
- `dataset/metadata.qwen2_5_0_5b.public.json`
- `configs/evaluation/qwen2_5_0_5b.public.single_model.sample.json`

## 9) MLflow UI Usage

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

## 10) Example Outputs

- Fake checkpoints: `checkpoint_epoch_*.ckpt`
- Training summary: `training_summary.json`
- Eval scorecard: `scorecard.json`
- Model cards:
  - comparison mode: `candidate_model_card.md`, `baseline_model_card.md`, `comparison_report.md`
  - single-model mode: `model_card.md`
- MLflow metrics:
  - training: `train_loss`, `validation_loss`, `validation_perplexity`, `test_loss`, `test_perplexity`, `tokens_per_second`
  - evaluation: `candidate_overall_score`, `baseline_overall_score`, `delta_vs_baseline`, candidate/baseline public/domain scores, candidate/baseline intrinsic perplexity proxies
  - single-model evaluation: `overall_score`, `public_benchmark_score`, `domain_benchmark_score`, `intrinsic_perplexity`

When `evaluation_backend` is `qwen_service`, `intrinsic_perplexity` is a response-quality proxy derived from the model outputs rather than a true perplexity computation.

## 11) Testing

```bash
uv run pytest -q
```

Includes:

- unit tests for continued pretraining/evaluation logic
- pipeline compile smoke tests


## 12) Deployment Pipeline (GitHub Actions)

A GitHub Actions workflow is available at `.github/workflows/deploy-platform.yml`.

- Trigger: **manual** (`workflow_dispatch`)
- Job 1 (`validate-and-compile`): installs dependencies with `uv`, runs `pytest -q`, compiles pipelines.
- Job 2 (`deploy-infra`): optionally applies infra manifests with `scripts/deploy/apply_infra.sh`.

Required GitHub secret for deploy step:

- `KUBE_CONFIG_DATA` (base64-encoded kubeconfig)

## 13) Trigger Training and Evaluation Jobs

### Option A: Kubeflow UI

1. Install standalone Kubeflow Pipelines:
   `make infra-apply-kubeflow`
2. Port-forward the Kubeflow Pipelines UI/API:
   `make kfp-port-forward`
3. Open `http://localhost:3000`
4. Upload compiled pipelines:
   - `pipelines/training/compiled/training_pipeline.yaml`
   - `pipelines/evaluation/compiled/evaluation_pipeline.yaml`
   - `pipelines/evaluation/compiled/single_model_evaluation_pipeline.yaml`
5. Start runs with the runtime arguments described below.

### Option B: Python submit scripts (recommended)

Set your KFP endpoint:

```bash
export KFP_HOST="http://localhost:3000"
```

Submit training run:

```bash
make submit-training \
  RUN_CONFIG_PATH=configs/training/run.sample.json \
  DATASET_METADATA_PATH=dataset/metadata.sample.json \
  KFP_HOST=http://localhost:3000
```

Submit evaluation run:

```bash
make submit-evaluation \
  EVAL_CONFIG_PATH=configs/evaluation/eval.sample.json \
  CANDIDATE_MODEL_URI=hf://Qwen/Qwen2.5-0.5B-Instruct \
  BASELINE_MODEL_URI=hf://Qwen/Qwen2.5-0.5B \
  KFP_HOST=http://localhost:3000
```

Submit single-model evaluation run:

```bash
make submit-single-evaluation \
  EVAL_CONFIG_PATH=configs/evaluation/single_model.sample.json \
  MODEL_URI=hf://Qwen/Qwen2.5-0.5B \
  KFP_HOST=http://localhost:3000
```

This repository uses standalone Kubeflow Pipelines rather than the full Kubeflow AI reference platform. The install path follows the official Kubeflow Pipelines standalone manifests.
On Docker Desktop local mode, the submit scripts also patch the generated Workflow to remove `runAsNonRoot` from Argo-managed pod specs, which avoids an Apple Silicon local compatibility issue with `argoexec`. They also re-apply the local `seaweedfs` service compatibility port before watching the submitted Workflow, so future runs do not depend on a fresh Kubeflow reinstall.

For the Qwen-backed comparison path, prefer `make baseline-apply ...` and `make candidate-apply ...` before submitting evaluation jobs. `make inference-apply` is still available as a convenience target that deploys the default baseline and candidate pair.
For the single-model path, deploy one endpoint and point `configs/evaluation/single_model.sample.json` at it via `inference_endpoint`.

### Runtime arguments reference

Training pipeline:
- `run_config_path`
- `dataset_metadata_path`

Evaluation pipeline:
- `eval_config_path`
- `candidate_model_uri`
- `baseline_model_uri`

Single-model evaluation pipeline:
- `eval_config_path`
- `model_uri`

## 14) Can experiments be configured from UI?

Short answer: **partially**.

- **MLflow UI**: best for *tracking and analysis* (metrics, params, artifacts, model versions). It does **not** author or launch Kubeflow pipelines. You can compare and inspect runs here, but pipeline runtime inputs still come from run submissions.
- **Kubeflow Pipelines UI**: yes, you can configure run arguments at submission time (for example `run_config_path`, `dataset_metadata_path`, `eval_config_path`, and model URIs) and launch runs directly from the UI.
- For reproducibility, keep canonical experiment definitions in versioned JSON config files under `configs/` and submit runs from those paths (UI or script).
