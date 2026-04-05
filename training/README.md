# Training workspace

This folder documents the continued pretraining path used by the training pipeline.

## Trigger a training pipeline run

1. Ensure the training pipeline is compiled:

```bash
uv sync --extra dev
uv run python pipelines/training/training_pipeline.py
```

The compiled pipeline references `llmops/base:dev` for helper steps and `llmops/trainer:dev` for the continued pretraining task by default.

The sample training config uses `sshleifer/tiny-gpt2` so the pipeline can validate the full causal language modeling flow on Docker Desktop CPU. If you have a real GPU-backed cluster, replace `base_model_id` and `tokenizer_name` with your Qwen checkpoint in `configs/training/run.sample.json`.

Training and serving are intentionally separate stages in this repository. After training, deploy the selected model behind the candidate endpoint and run the evaluation pipeline against that endpoint rather than spinning up inference inside the training or evaluation pipeline itself.

2. Submit a run to Kubeflow Pipelines:

```bash
export KFP_HOST="http://localhost:3000"
uv run python scripts/pipelines/submit_training_run.py \
  --run-config-path configs/training/run.sample.json \
  --dataset-metadata-path dataset/metadata.sample.json
```

### Parameters

- `run_config_path`: training runtime config JSON.
- `dataset_metadata_path`: dataset metadata manifest JSON. The manifest points to `train`, `validation`, and `test` raw text splits, currently under `dataset/cpt/`. If your real corpus uses `eval` instead of `validation`, that alias is accepted too.

The training backend is causal language modeling continued pretraining. The pipeline logs final training loss, validation/test loss, validation/test perplexity, and an estimated tokens-per-second metric to MLflow.
