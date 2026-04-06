# Qwen LoRA on MacBook

This guide explains how to use LoRA with `Qwen/Qwen2.5-0.5B` on a local MacBook and why the repository's local LoRA preset uses the values it does.

## What LoRA does

LoRA keeps the base model frozen and trains a small low-rank update on top of selected weight matrices.

Instead of retraining a full matrix `W`, LoRA learns a small update:

```text
W' = W + (alpha / r) * B * A
```

Where:

- `W` is the original frozen weight
- `A` and `B` are the trainable low-rank adapter matrices
- `r` is the adapter rank
- `alpha` scales the learned update

At inference time, the runtime loads the base model and then applies the saved adapter weights.

## Why LoRA is better for local runs

For a MacBook local workflow, LoRA is usually a better default than full-parameter continued pretraining because it:

- trains far fewer parameters
- uses less optimizer memory
- writes much smaller output artifacts
- is less likely to overfit or damage the base model on a small domain dataset

This is especially important for local `Qwen/Qwen2.5-0.5B` runs on Apple Silicon.

## Recommended starting values

For local healthcare continued pretraining on a MacBook, start with:

- `lora_r: 8`
- `lora_alpha: 16`
- `lora_dropout: 0.05`
- `lora_target_modules: ["q_proj", "v_proj"]`

These values are intentionally conservative.

### Why `r = 8`

`r` controls adapter capacity.

- lower `r` means fewer trainable parameters and lower memory pressure
- higher `r` means more expressiveness, but more memory and a higher risk of overfitting on small local datasets

For local domain adaptation, `8` is a good first stop. If the run is stable and the domain shift is strong, `16` is the next value to try.

### Why `alpha = 16`

`alpha` scales the adapter update relative to the base weight.

For `r = 8`, `alpha = 16` gives a moderate update size. It is large enough to move the model, but not so large that the adapter dominates quickly on a small corpus.

Practical rule:

- start with `alpha` around `2 * r`
- if the model barely changes, increase it carefully
- if the model becomes unstable or degrades sharply, reduce it

### Why `dropout = 0.05`

LoRA dropout regularizes the adapter path.

For a small local healthcare corpus, `0.05` is enough to reduce brittle memorization without slowing training too much. If the dataset gets larger and cleaner, `0.0` to `0.05` is reasonable. For noisier small corpora, `0.1` can help.

## Which target modules to use

The most important setting after learning rate is the target module list.

### Best starting point on a MacBook

Use:

```json
["q_proj", "v_proj"]
```

Why:

- this is the lightest useful attention-only adaptation
- it is widely used as a conservative LoRA baseline
- it keeps trainable parameters and memory lower than adapting every attention and MLP projection

### Next step if you want more capacity

If `q_proj` and `v_proj` are stable but the improvement is too small, expand to:

```json
["q_proj", "k_proj", "v_proj", "o_proj"]
```

This is a good balanced option for stronger adaptation.

### Avoid this as your first local run

Do not start by adapting all attention and MLP projections together:

```json
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

That can work, but it is a bigger update, uses more memory, and is more likely to overshoot on a small local corpus.

## Recommended local recipe for this repository

For `Qwen/Qwen2.5-0.5B` on your MacBook:

- `adaptation_method: "lora"`
- `learning_rate: 5e-6`
- `warmup_ratio: 0.03`
- `batch_size: 1`
- `gradient_accumulation_steps: 4`
- `lora_r: 8`
- `lora_alpha: 16`
- `lora_dropout: 0.05`
- `lora_target_modules: ["q_proj", "v_proj"]`

This is the current repository preset:

- [local_qwen2_5_0_5b_healthcare_pubmed_500.lora.json](/Users/narendersurabhi/llmops-experiment-platform/configs/training/local_qwen2_5_0_5b_healthcare_pubmed_500.lora.json)

## How to decide whether to increase capacity

Stay on the smaller config if:

- the run is already near your memory limit
- the model quality is unstable between checkpoints
- the domain dataset is still fairly small

Increase capacity gradually if:

- training is stable
- eval metrics improve only a little
- you want stronger domain adaptation

Recommended escalation path:

1. Keep `r = 8`, expand target modules to `q/k/v/o`
2. Move to `r = 16`
3. Revisit `alpha`, for example `16` to `32`

Do not change everything at once, or you will not know which change helped.

## How to run it

### In the local UI

Run:

```bash
make run-local-ui
```

Then in the Training tab choose:

- `Qwen Healthcare PubMed LoRA (500/500/500)`

### On the CLI

```bash
make run-local-training \
  RUN_CONFIG_PATH=configs/training/local_qwen2_5_0_5b_healthcare_pubmed_500.lora.json \
  DATASET_METADATA_PATH=dataset/metadata.qwen2_5_0_5b.healthcare.pubmed.500.json \
  OUTPUT_DIR=artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500_lora
```

## How evaluation works for a LoRA model

The repository can now evaluate a saved LoRA adapter directory directly.

For CPT-style evaluation:

```bash
make run-local-eval \
  LOCAL_MODEL_PATH=artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500_lora/model \
  EVAL_CONFIG_PATH=configs/evaluation/qwen2_5_0_5b.public.cpt.sample.json \
  OUTPUT_DIR=artifacts/local-eval/qwen2_5_0_5b_healthcare_pubmed_500_lora_cpt
```

The loader detects `adapter_config.json`, loads the base Qwen model, then applies the LoRA adapter before evaluating.

## Practical guidance

- Use LoRA when the goal is domain adaptation on limited local hardware.
- Use full-model CPT only when you have much more data and much more compute.
- Measure improvement with held-out CPT metrics first, not only chat-style prompt benchmarks.
- If the domain dataset is small, LoRA is usually the more defensible baseline.
