#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset" / "cpt" / "qwen2_5_0_5b_3000"
GOLDEN_PATH = REPO_ROOT / "dataset" / "golden" / "qwen2_5_0_5b_300.golden.json"
METADATA_PATH = REPO_ROOT / "dataset" / "metadata.qwen2_5_0_5b.3000.json"
EVAL_CONFIG_PATH = REPO_ROOT / "configs" / "evaluation" / "qwen2_5_0_5b.300.single_model.sample.json"

SPLIT_COUNTS = {"train": 2400, "eval": 300, "test": 300}

SUBJECTS = [
    ("kubernetes scheduler", "platform"),
    ("kubernetes rollout controller", "platform"),
    ("kubeflow pipeline step", "platform"),
    ("mlflow experiment tracker", "platform"),
    ("minio artifact bucket", "platform"),
    ("postgres metadata store", "platform"),
    ("qwen inference service", "serving"),
    ("candidate endpoint", "evaluation"),
    ("baseline endpoint", "evaluation"),
    ("continued pretraining job", "cpt"),
    ("tokenization workflow", "cpt"),
    ("held-out evaluation split", "cpt"),
    ("model card artifact", "evaluation"),
    ("scorecard report", "evaluation"),
    ("local macbook inference run", "serving"),
    ("kubeflow launcher pod", "platform"),
    ("minio s3 artifact object", "platform"),
    ("training summary artifact", "cpt"),
    ("validation perplexity metric", "cpt"),
    ("test loss metric", "cpt"),
]

ACTIONS = [
    "records operational state for downstream review",
    "helps engineers detect regressions before promotion",
    "coordinates artifacts between training and evaluation stages",
    "keeps pipeline behavior reproducible across local and cluster runs",
    "captures metadata that supports debugging and release decisions",
    "improves observability for experiments, checkpoints, and evaluations",
    "supports repeatable continued pretraining on domain text corpora",
    "makes model quality comparisons easier to audit over time",
    "provides context required for model cards and scorecards",
    "reduces manual handoffs between training, serving, and evaluation",
]

DETAILS = [
    "It is used to align datasets, checkpoints, and benchmark outputs with a consistent run identifier.",
    "Teams rely on it to inspect metrics, artifacts, and qualitative responses before promoting a model.",
    "In this repository it connects storage, orchestration, and evaluation outputs into a single operating loop.",
    "The design favors explicit artifacts so failures can be diagnosed from saved summaries instead of console logs alone.",
    "This pattern is useful when comparing a base model against a newly continued-pretrained checkpoint.",
    "A well-structured workflow keeps train, eval, test, and golden data separate to reduce leakage.",
    "Engineers can trace failures faster when operational metadata and model outputs are stored side by side.",
    "That separation matters because release decisions should depend on held-out evidence rather than training-only signals.",
    "It also makes local MacBook validation feasible before moving the same workflow to Kubernetes.",
    "Good platform hygiene requires reproducible configs, stable artifact paths, and explicit evaluation criteria.",
]

QUALIFIERS = [
    "under CPU-only constraints",
    "during local-first experimentation",
    "when validating Kubeflow and MLflow integration",
    "for Qwen continued pretraining workflows",
    "while comparing candidate and baseline endpoints",
    "when evaluating real served model behavior",
    "for audit-friendly model release gates",
    "under small-batch local training conditions",
]

PUBLIC_TOPICS = [
    ("kubernetes-orchestration", "Explain what Kubernetes does for containerized applications.", ["kubernetes", "container", "orchestration", "cluster"]),
    ("mlflow-tracking", "Describe what MLflow tracks during a machine learning run.", ["mlflow", "metrics", "artifacts", "parameters"]),
    ("object-storage", "Explain why object storage is useful in machine learning platforms.", ["storage", "artifact", "object", "checkpoint"]),
    ("checkpoint-selection", "Explain why checkpoints matter during model training.", ["checkpoint", "training", "model", "evaluation"]),
    ("held-out-evaluation", "Explain why held-out evaluation is necessary in machine learning.", ["evaluation", "held-out", "generalization", "test"]),
]

DOMAIN_TOPICS = [
    ("candidate-vs-baseline", "In this repository, explain how candidate and baseline endpoints are used during evaluation.", ["candidate", "baseline", "endpoint", "evaluation"]),
    ("continued-pretraining", "Explain continued pretraining and how it differs from instruction tuning.", ["continued pretraining", "raw text", "next-token", "instruction tuning"]),
    ("mlflow-minio", "Explain how MLflow and MinIO work together in this repository.", ["mlflow", "minio", "artifact", "s3"]),
    ("kubeflow-mlflow-loop", "Describe how Kubeflow Pipelines and MLflow interact in the local-first workflow.", ["kubeflow", "mlflow", "pipeline", "artifacts"]),
    ("train-eval-test", "Explain why the corpus is split into train, eval, and test sets.", ["train", "eval", "test", "validation"]),
    ("model-card-purpose", "Explain what a model card should communicate after evaluation.", ["model card", "evaluation", "metrics", "limitations"]),
    ("scorecard-purpose", "Explain what information a scorecard captures in this repository.", ["scorecard", "benchmark", "overall score", "prompt"]),
    ("endpoint-vs-offline", "Explain why served endpoint evaluation is useful in addition to offline model evaluation.", ["endpoint", "serving", "model", "latency"]),
    ("local-macbook-flow", "Describe the local MacBook flow for training and evaluation in this repository.", ["local", "macbook", "training", "evaluation"]),
    ("artifact-lineage", "Explain why artifact lineage matters for model release decisions.", ["artifact", "lineage", "run", "release"]),
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_record(index: int, split: str) -> dict:
    subject, domain = SUBJECTS[index % len(SUBJECTS)]
    action = ACTIONS[(index // len(SUBJECTS)) % len(ACTIONS)]
    detail = DETAILS[(index // (len(SUBJECTS) * len(ACTIONS))) % len(DETAILS)]
    qualifier = QUALIFIERS[index % len(QUALIFIERS)]
    text = (
        f"The {subject} {action} {qualifier}. "
        f"{detail} "
        f"For this repository, practitioners expect the {subject} to expose reliable artifacts, "
        f"stable configuration, and measurable outcomes for training and evaluation. "
        f"This example document is part of the {split} split for continued pretraining."
    )
    return {
        "id": f"{split}-{index + 1:04d}",
        "text": text,
        "source": "llmops-generated-corpus",
        "language": "en",
        "domain": domain,
        "split": split,
    }


def write_split(split: str, count: int) -> Path:
    path = DATASET_DIR / f"{split}.jsonl"
    ensure_parent(path)
    lines = [json.dumps(build_record(i, split), ensure_ascii=True) for i in range(count)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_reference_answer(topic_prompt: str, keywords: list[str], variant: int, suite: str) -> str:
    keyword_phrase = ", ".join(keywords[:3])
    closing = (
        "The answer should stay grounded in the repository workflow and connect architecture to measurable evaluation."
        if suite == "domain"
        else "The answer should stay factual and avoid implementation noise."
    )
    return (
        f"{topic_prompt} A strong answer should mention {keyword_phrase}. "
        f"It should explain the concept clearly, tie it to model quality or platform behavior, "
        f"and avoid vague repetition. {closing} Variant {variant + 1}."
    )


def build_prompt_record(index: int) -> dict:
    topic_pool = PUBLIC_TOPICS if index % 3 == 0 else DOMAIN_TOPICS
    topic_id, prompt, keywords = topic_pool[index % len(topic_pool)]
    suite = "public" if topic_pool is PUBLIC_TOPICS else "domain"
    variant = index // len(topic_pool)
    prompt_suffix = [
        "Answer in three concise sentences.",
        "Focus on operational implications.",
        "Use repository-specific language where relevant.",
        "Explain it for an engineer reviewing a release.",
        "Keep the answer direct and concrete.",
    ][index % 5]
    return {
        "id": f"golden-{topic_id}-{index + 1:03d}",
        "suite": suite,
        "prompt": f"{prompt} {prompt_suffix}",
        "reference_answer": build_reference_answer(prompt, keywords, variant, suite),
        "expected_keywords": keywords,
        "minimum_response_chars": 120 if suite == "public" else 150,
    }


def build_golden_set() -> dict:
    prompts = [build_prompt_record(i) for i in range(300)]
    return {
        "dataset_name": "qwen2-5-0-5b-llmops-golden-300",
        "dataset_version": "golden-v300",
        "description": "Generated 300-prompt golden set for local Qwen continued pretraining evaluation.",
        "benchmark_prompts": prompts,
    }


def build_metadata() -> dict:
    return {
        "dataset_name": "qwen2-5-0-5b-llmops-cpt-3000",
        "dataset_version": "qwen2-5-0-5b-cpt-3000-v1",
        "format": "jsonl",
        "text_field": "text",
        "source": "llmops-generated-corpus",
        "language": "en",
        "splits": {
            "train": "cpt/qwen2_5_0_5b_3000/train.jsonl",
            "eval": "cpt/qwen2_5_0_5b_3000/eval.jsonl",
            "test": "cpt/qwen2_5_0_5b_3000/test.jsonl",
        },
    }


def build_eval_config() -> dict:
    return {
        "project": "llmops-local",
        "config_version": "eval-single-golden-v300",
        "run_id": "eval-qwen2-5-0-5b-golden-300",
        "dataset_version": "golden-v300",
        "evaluation_mode": "single_model",
        "evaluation_backend": "qwen_service",
        "inference_endpoint": "http://qwen-inference.llmops.svc.cluster.local:8000/v1/generate",
        "judge_model_id": "single-model-run",
        "publish_model_cards": True,
        "golden_set_path": "dataset/golden/qwen2_5_0_5b_300.golden.json",
        "max_new_tokens": 192,
        "temperature": 0.0,
        "request_timeout_seconds": 120,
    }


def main() -> None:
    for split, count in SPLIT_COUNTS.items():
        write_split(split, count)
    write_json(GOLDEN_PATH, build_golden_set())
    write_json(METADATA_PATH, build_metadata())
    write_json(EVAL_CONFIG_PATH, build_eval_config())
    print(f"Wrote CPT data to {DATASET_DIR}")
    print(f"Wrote dataset metadata to {METADATA_PATH}")
    print(f"Wrote golden set to {GOLDEN_PATH}")
    print(f"Wrote eval config to {EVAL_CONFIG_PATH}")


if __name__ == "__main__":
    main()
