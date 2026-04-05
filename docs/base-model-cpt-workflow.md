# Base Model to CPT Workflow

This guide documents the repository workflow for the following goal:

1. Take a base model
2. Create a model card
3. Evaluate it with a golden set
4. Perform continued pretraining on the base model
5. Evaluate the trained model
6. Create a model card for the trained model

## What "golden set" means in this repository

This repository can represent the golden set in either of two ways:

- inline in the evaluation config JSON as `benchmark_prompts`
- in a separate JSON file referenced by `golden_set_path`

Each golden-set record contains:

- `id`
- `suite`
- `prompt`
- optional `reference_answer`
- optional `expected_keywords`
- optional `minimum_response_chars`

The evaluator scores the model response against those fields and writes `scorecard.json` plus model-card artifacts to MLflow.

## Prerequisites

1. Install Python dependencies:

```bash
uv sync --extra dev
```

2. Build images:

```bash
make docker-build
```

3. Deploy platform infrastructure:

```bash
make infra-apply
make infra-apply-kubeflow
```

4. Port-forward Kubeflow Pipelines and export the endpoint:

```bash
make kfp-port-forward
export KFP_HOST=http://localhost:3000
```

5. Compile pipelines:

```bash
make pipelines-compile
```

## Step 1: Prepare the golden set

You can use the ready-made starter files added to the repo:

- `dataset/metadata.qwen2_5_0_5b.sample.json`
- `dataset/cpt/qwen2_5_0_5b/train.jsonl`
- `dataset/cpt/qwen2_5_0_5b/eval.jsonl`
- `dataset/cpt/qwen2_5_0_5b/test.jsonl`
- `dataset/golden/qwen2_5_0_5b.golden.json`
- `configs/evaluation/qwen2_5_0_5b.single_model.sample.json`

Or create your own evaluation config by copying the single-model sample config and pointing it at a golden-set file:

```bash
cp configs/evaluation/qwen2_5_0_5b.single_model.sample.json configs/evaluation/base_model_eval.json
```

Edit `configs/evaluation/base_model_eval.json` and update:

- `project`
- `run_id`
- `inference_endpoint`
- `golden_set_path`

Use one file for the base-model evaluation and a second file for the trained-model evaluation if you want distinct run IDs:

```bash
cp configs/evaluation/qwen2_5_0_5b.single_model.sample.json configs/evaluation/trained_model_eval.json
```

Keep the golden set the same in both files if you want a direct before/after comparison.

## Step 2: Serve the base model

If the base model should be downloaded from Hugging Face:

```bash
make base-model-apply MODEL_ID=Qwen/Qwen2.5-0.5B
```

If the base model already exists as a local model directory and that directory is visible to the Kubernetes node:

```bash
make base-model-apply \
  LOCAL_MODEL_PATH=/absolute/path/to/base-model-dir \
  MODEL_URI=models:/base-model/local
```

The stable base-model endpoint is:

```text
http://qwen-inference.llmops.svc.cluster.local:8000/v1/generate
```

Set that endpoint in `configs/evaluation/base_model_eval.json`.

## Step 3: Evaluate the base model and create the first model card

Run the single-model evaluation pipeline:

```bash
make submit-single-evaluation \
  EVAL_CONFIG_PATH=configs/evaluation/base_model_eval.json \
  MODEL_URI=hf://Qwen/Qwen2.5-0.5B \
  KFP_HOST=http://localhost:3000
```

This produces:

- `scorecard.json`
- `model_cards/model_card.md`

Artifacts are logged to MLflow under the evaluation experiment for your project.

## Step 4: Run continued pretraining on the base model

Create or update the training config:

```bash
cp configs/training/run.sample.json configs/training/base_model_cpt.json
```

Edit `configs/training/base_model_cpt.json` and set:

- `project`
- `run_id`
- `base_model_id`
- `tokenizer_name`
- `epochs`
- `batch_size`
- `learning_rate`
- other CPT hyperparameters

Point the dataset manifest at your CPT corpus splits:

```bash
cp dataset/metadata.sample.json dataset/metadata.cpt.json
```

Edit `dataset/metadata.cpt.json` and set:

- `dataset_name`
- `dataset_version`
- `text_field`
- `splits.train`
- `splits.validation` or `splits.eval`
- `splits.test`

Before running training, validate your actual files locally:

```bash
make validate-local-data \
  DATASET_METADATA_PATH=dataset/metadata.cpt.json \
  GOLDEN_SET_PATH=dataset/golden/my_actual_golden.json
```

Submit the training run:

```bash
make submit-training \
  RUN_CONFIG_PATH=configs/training/base_model_cpt.json \
  DATASET_METADATA_PATH=dataset/metadata.cpt.json \
  KFP_HOST=http://localhost:3000
```

This logs training metrics and a `training_summary.json` artifact to MLflow.

## Step 5: Make the trained model available for inference

This is the current manual handoff point in the repository.

The trainer creates a final model directory during continued pretraining, but the training pipeline does not yet automatically publish that directory as a deployable serving artifact or wire it into `candidate-apply`.

Today you need to do one of the following manually:

1. Export the trained model directory to a path that is visible to the Kubernetes node filesystem.
2. Bake the trained model into a serving image.
3. Upload the trained model to a location that you can mount back into the inference pod.

Once you have a node-visible model directory, deploy it as the candidate endpoint:

```bash
make candidate-apply \
  LOCAL_MODEL_PATH=/absolute/path/to/trained-model-dir \
  MODEL_URI=models:/trained-model/local
```

The candidate endpoint is:

```text
http://qwen-candidate.llmops.svc.cluster.local:8000/v1/generate
```

## Step 6: Evaluate the trained model and create the second model card

Update `configs/evaluation/trained_model_eval.json` so that `inference_endpoint` points at the candidate endpoint:

```text
http://qwen-candidate.llmops.svc.cluster.local:8000/v1/generate
```

Then submit the single-model evaluation run:

```bash
make submit-single-evaluation \
  EVAL_CONFIG_PATH=configs/evaluation/trained_model_eval.json \
  MODEL_URI=models:/trained-model/local \
  KFP_HOST=http://localhost:3000
```

This produces another:

- `scorecard.json`
- `model_cards/model_card.md`

for the trained model.

## Optional: Compare base model vs trained model directly

If you also want a direct before/after comparison report, keep the base model on the stable baseline endpoint and deploy the trained model on the candidate endpoint, then run:

```bash
make submit-evaluation \
  EVAL_CONFIG_PATH=configs/evaluation/eval.sample.json \
  CANDIDATE_MODEL_URI=models:/trained-model/local \
  BASELINE_MODEL_URI=hf://Qwen/Qwen2.5-0.5B \
  KFP_HOST=http://localhost:3000
```

This produces:

- `scorecard.json`
- `model_cards/candidate_model_card.md`
- `model_cards/baseline_model_card.md`
- `model_cards/comparison_report.md`

## Current limitation

The missing automation in the current repository is the handoff from continued pretraining output to inference deployment. Training and evaluation both work, but automatic "train model -> publish deployable artifact -> deploy candidate endpoint" is not implemented yet.
