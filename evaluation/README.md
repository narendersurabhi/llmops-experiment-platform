# Evaluation workspace

This folder contains the local evaluation configuration used by the Kubeflow evaluation pipelines.

## Trigger a comparison evaluation pipeline run

1. Ensure the evaluation pipeline is compiled:

```bash
uv run python pipelines/evaluation/evaluation_pipeline.py
```

The compiled pipeline references `llmops/base:dev` for helper steps and `llmops/evaluator:dev` for the evaluator task by default.

2. Deploy or verify the stable baseline endpoint:

```bash
make baseline-apply MODEL_ID=Qwen/Qwen2.5-0.5B
```

`make base-model-apply ...` is an alias for this stable endpoint flow.

3. Deploy the candidate endpoint you want to evaluate:

```bash
make candidate-apply MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
```

To serve a local checkpoint directory instead of downloading from Hugging Face:

```bash
make candidate-apply \
  LOCAL_MODEL_PATH=/absolute/path/to/model-dir \
  MODEL_URI=models:/candidate/local
```

The mounted directory is used as the model load path inside the container. The source path must be readable by the Kubernetes node.

This starts two CPU-only FastAPI services in Kubernetes:

- `qwen-baseline` as the stable baseline service
- `qwen-candidate` as the staged comparison target

`qwen-inference` is a stable alias that points at the baseline deployment.

The first pod startup may take several minutes because the models are downloaded on demand.

4. Submit a run to Kubeflow Pipelines:

```bash
export KFP_HOST="http://localhost:3000"
uv run python scripts/pipelines/submit_evaluation_run.py \
  --eval-config-path configs/evaluation/eval.sample.json \
  --candidate-model-uri hf://Qwen/Qwen2.5-0.5B-Instruct \
  --baseline-model-uri hf://Qwen/Qwen2.5-0.5B
```

### Parameters

- `eval_config_path`: evaluation runtime config JSON.
- `candidate_model_uri`: candidate model label recorded in the scorecard and MLflow.
- `baseline_model_uri`: baseline model label recorded in the scorecard and MLflow.

The scorecard output is now produced from real candidate and baseline responses to the configured benchmark prompts. The evaluator writes:

- `scorecard.json`
- `model_cards/candidate_model_card.md`
- `model_cards/baseline_model_card.md`
- `model_cards/comparison_report.md`

These are logged to MLflow as artifacts. If `baseline_inference_endpoint` is omitted, the evaluator next tries the stable `qwen-inference` endpoint before falling back to `baseline_overall_score` from the config.

5. If the candidate wins, promote it:

```bash
make promote-candidate
```

## Trigger a single-model evaluation pipeline run

1. Ensure the single-model evaluation pipeline is compiled:

```bash
uv run python pipelines/evaluation/single_model_evaluation_pipeline.py
```

2. Deploy or verify the endpoint you want to score:

```bash
make baseline-apply MODEL_ID=Qwen/Qwen2.5-0.5B
```

3. Submit a run to Kubeflow Pipelines:

```bash
export KFP_HOST="http://localhost:3000"
uv run python scripts/pipelines/submit_single_model_evaluation_run.py \
  --eval-config-path configs/evaluation/single_model.sample.json \
  --model-uri hf://Qwen/Qwen2.5-0.5B
```

This evaluator writes:

- `scorecard.json`
- `model_cards/model_card.md`

These are logged to MLflow as artifacts with no baseline or comparison fields.

## Run single-model evaluation locally on your Mac

If you want to run the same single-model evaluation flow directly on your MacBook instead of Docker/Kubernetes, use:

```bash
uv sync --extra dev --extra inference

uv run python scripts/local/run_single_model_eval_local.py \
  --eval-config-path configs/evaluation/single_model.sample.json \
  --model-id Qwen/Qwen2.5-0.5B
```

This script:

- starts the local FastAPI inference server on `127.0.0.1:8000`
- rewrites the eval config to point at the local endpoint
- runs the single-model evaluator directly
- writes `scorecard.json` and `model_cards/model_card.md` under `artifacts/local-eval/...`
- logs MLflow metadata to a local SQLite database in the same output directory

You can also evaluate a local checkpoint directory:

```bash
make run-local-training

make run-local-eval \
  LOCAL_MODEL_PATH=artifacts/local-train/qwen2_5_0_5b/model \
  EVAL_CONFIG_PATH=configs/evaluation/qwen2_5_0_5b.single_model.sample.json
```

Or run the same steps directly:

```bash
uv run python -m components.trainer.run_pretraining \
  --config-path configs/training/qwen2_5_0_5b.cpt.sample.json \
  --dataset-metadata-path dataset/metadata.qwen2_5_0_5b.sample.json \
  --output-dir artifacts/local-train/qwen2_5_0_5b

uv run python scripts/local/run_single_model_eval_local.py \
  --eval-config-path configs/evaluation/qwen2_5_0_5b.single_model.sample.json \
  --local-model-path artifacts/local-train/qwen2_5_0_5b/model \
  --model-uri file://$PWD/artifacts/local-train/qwen2_5_0_5b/model
```

If `LOCAL_MODEL_PATH` does not exist, the script now fails fast with a clear error before starting the local inference server.
