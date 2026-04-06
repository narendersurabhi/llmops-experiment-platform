# CPT Metrics Glossary

This document explains the main metrics used by the local continued pretraining (CPT) evaluation flow in this repository.

Relevant implementation:
- `components/evaluator/run_evaluation.py`

## Scope

These metrics apply to the CPT evaluation backend:

- `evaluation_backend: "cpt_model"`

They are different from the older chat-style prompt evaluation metrics, which use keyword matching over generated answers.

## Validation Loss

Field:
- `validation_loss`

How it is calculated:
- The evaluator tokenizes the full validation split.
- It appends an EOS token between documents.
- It packs tokens into fixed-size blocks using `block_size`.
- Padding tokens are masked with `-100`, so they do not contribute to loss.
- For each batch, the model computes causal language modeling loss.
- The evaluator multiplies batch loss by the number of active tokens, sums across batches, then divides by the total number of active tokens.

Interpretation:
- Average negative log-likelihood per token on the validation split.
- Lower is better.

Use:
- Good for model selection and checkpoint tuning.

## Test Loss

Field:
- `test_loss`

How it is calculated:
- Same procedure as validation loss, but on the held-out test split.

Interpretation:
- Average negative log-likelihood per token on unseen held-out text.
- Lower is better.

Use:
- Better than validation loss for final comparison between runs.

## Validation Perplexity

Field:
- `validation_perplexity`

How it is calculated:
- `exp(validation_loss)`
- The implementation applies a numerical safety cap: `exp(min(loss, 20.0))`

Interpretation:
- A transformed version of loss.
- Lower is better.
- It ranks models the same way as validation loss.

Use:
- Easier to read than raw loss for many practitioners.

## Test Perplexity

Field:
- `test_perplexity`

How it is calculated:
- `exp(test_loss)`
- Same numerical safety cap as validation perplexity.

Interpretation:
- A transformed version of test loss.
- Lower is better.
- It ranks models the same way as test loss.

Use:
- Primary top-line CPT comparison metric in this repo.

## Completion Reference Token Recall

Field:
- `completion_reference_token_recall`

How it is calculated:
- The evaluator samples a small number of documents from a chosen split, usually `test`.
- For each sampled document:
  - the first `completion_prompt_tokens` tokens become the prompt
  - the next `completion_reference_tokens` tokens become the reference continuation
- The model generates up to `max_new_tokens` tokens from the prompt.
- Both generated text and reference text are normalized into lowercase alphanumeric tokens.
- The metric computes:

`matched unique reference tokens / total unique reference tokens`

- The final reported metric is the average across sampled completions.

Interpretation:
- Measures token-set overlap with the held-out continuation.
- Higher is better.

Important limitation:
- It ignores token order.
- It ignores repeated tokens.
- It ignores fluency and factuality.
- It uses only a small sample of completions.

Use:
- Lightweight qualitative sanity check.
- Do not trust it more than held-out loss/perplexity.

## Why Loss And Perplexity Matter More Than Recall

In this CPT workflow:

- `validation_loss` and `test_loss` use the full split
- `validation_perplexity` and `test_perplexity` are direct transforms of that loss
- `completion_reference_token_recall` uses only a small sampled subset

That means:
- loss/perplexity are more stable
- recall is noisier
- recall is useful, but secondary

## Recommended Reading Order For Results

When comparing two CPT runs, read metrics in this order:

1. `test_loss`
2. `test_perplexity`
3. `validation_loss`
4. `validation_perplexity`
5. `completion_reference_token_recall`
6. completion sample text

## Practical Rule Of Thumb

Use this interpretation:

- Lower `test_perplexity` with the same dataset and eval config usually means the CPT run improved.
- Higher `completion_reference_token_recall` is nice, but by itself is not enough to declare a winner.
- If perplexity improves and recall is flat, that is still a valid CPT gain.
- If recall improves but perplexity worsens, treat the run as suspicious until reviewed manually.
