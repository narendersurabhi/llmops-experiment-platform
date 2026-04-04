# Evaluation workspace

This folder is reserved for future evaluation assets and scripts.

## Trigger an evaluation pipeline run

1. Ensure the evaluation pipeline is compiled:

```bash
python pipelines/evaluation/evaluation_pipeline.py
```

2. Submit a run to Kubeflow Pipelines:

```bash
export KFP_HOST="http://localhost:8080"
python scripts/pipelines/submit_evaluation_run.py \
  --eval-config-path configs/evaluation/eval.sample.json \
  --candidate-model-uri models:/candidate-llm/1 \
  --baseline-model-uri models:/baseline-llm/1
```

### Parameters

- `eval_config_path`: evaluation runtime config JSON.
- `candidate_model_uri`: evaluated model URI.
- `baseline_model_uri`: baseline model URI.
