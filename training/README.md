# Training workspace

This folder is reserved for future training assets and scripts.

## Trigger a training pipeline run

1. Ensure the training pipeline is compiled:

```bash
python pipelines/training/training_pipeline.py
```

2. Submit a run to Kubeflow Pipelines:

```bash
export KFP_HOST="http://localhost:8080"
python scripts/pipelines/submit_training_run.py \
  --run-config-path configs/training/run.sample.json \
  --dataset-metadata-path dataset/metadata.sample.json
```

### Parameters

- `run_config_path`: training runtime config JSON.
- `dataset_metadata_path`: dataset metadata JSON.
