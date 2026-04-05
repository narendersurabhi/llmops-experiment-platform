from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import mlflow

from components.trainer import simulate_training
from components.trainer.simulate_training import resolve_output_paths
from components.utils.io_utils import read_json, write_json
from components.utils.mlflow_utils import start_run


def resolve_split_path(dataset_metadata_path: str, split_path: str) -> Path:
    split = Path(split_path)
    if split.is_absolute():
        return split
    return Path(dataset_metadata_path).resolve().parent / split


def normalize_dataset_splits(splits: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(splits)
    if "validation" not in normalized and "eval" in normalized:
        normalized["validation"] = normalized["eval"]
    return normalized


def load_jsonl_texts(path: Path, text_field: str) -> list[str]:
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        text = str(record.get(text_field, "")).strip()
        if text:
            texts.append(text)
    if not texts:
        raise ValueError(f"No non-empty '{text_field}' records found in {path}")
    return texts


def load_dataset_manifest(dataset_metadata_path: str) -> dict[str, Any]:
    manifest = read_json(dataset_metadata_path)
    splits = normalize_dataset_splits(manifest.get("splits", {}))
    if not isinstance(splits, dict):
        raise ValueError("dataset metadata must contain a 'splits' object")

    text_field = str(manifest.get("text_field", "text"))
    resolved_splits: dict[str, dict[str, Any]] = {}
    for split_name in ["train", "validation", "test"]:
        split_ref = splits.get(split_name)
        if not split_ref:
            raise ValueError(f"dataset metadata missing '{split_name}' split")
        split_path = resolve_split_path(dataset_metadata_path, str(split_ref))
        if not split_path.exists():
            raise FileNotFoundError(f"{split_name} split not found: {split_path}")
        texts = load_jsonl_texts(split_path, text_field)
        resolved_splits[split_name] = {
            "path": str(split_path),
            "texts": texts,
            "document_count": len(texts),
            "character_count": sum(len(text) for text in texts),
            "word_count": sum(len(text.split()) for text in texts),
        }

    manifest["text_field"] = text_field
    manifest["splits"] = splits
    manifest["resolved_splits"] = resolved_splits
    manifest["num_samples"] = sum(split["document_count"] for split in resolved_splits.values())
    return manifest


def estimate_tokens_per_second(train_result_metrics: dict[str, Any], cfg: dict[str, Any]) -> float | None:
    steps_per_second = train_result_metrics.get("train_steps_per_second")
    if steps_per_second is None:
        return None
    block_size = int(cfg.get("block_size", 128))
    batch_size = int(cfg.get("batch_size", 1))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    return round(float(steps_per_second) * block_size * batch_size * grad_accum, 2)


def with_perplexity(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    payload = dict(metrics)
    loss = payload.get(f"{prefix}_loss")
    if loss is not None:
        payload[f"{prefix}_perplexity"] = round(math.exp(min(float(loss), 20.0)), 4)
    return payload


def build_training_arguments(cfg: dict[str, Any], output_dir: Path, train_dataset_size: int, training_arguments_cls):
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": float(cfg.get("epochs", 1)),
        "per_device_train_batch_size": int(cfg.get("batch_size", 1)),
        "per_device_eval_batch_size": int(cfg.get("batch_size", 1)),
        "learning_rate": float(cfg.get("learning_rate", 5e-5)),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 1)),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "warmup_ratio": float(cfg.get("warmup_ratio", 0.0)),
        "max_grad_norm": float(cfg.get("max_grad_norm", 1.0)),
        "logging_strategy": "steps",
        "logging_steps": max(1, train_dataset_size // max(int(cfg.get("batch_size", 1)), 1)),
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
    }

    init_parameters = inspect.signature(training_arguments_cls.__init__).parameters
    evaluation_key = "eval_strategy" if "eval_strategy" in init_parameters else "evaluation_strategy"
    training_kwargs[evaluation_key] = "epoch"
    return training_arguments_cls(**training_kwargs)


def patch_accelerator_class_unwrap_model_compat(accelerator_cls) -> None:
    unwrap_model = accelerator_cls.unwrap_model
    parameters = inspect.signature(unwrap_model).parameters
    if "keep_torch_compile" in parameters:
        return
    if getattr(unwrap_model, "_llmops_keep_torch_compile_compat", False):
        return

    def unwrap_model_compat(self, model, keep_fp32_wrapper: bool = True, keep_torch_compile=None):
        return unwrap_model(self, model, keep_fp32_wrapper=keep_fp32_wrapper)

    unwrap_model_compat._llmops_keep_torch_compile_compat = True
    accelerator_cls.unwrap_model = unwrap_model_compat


def patch_accelerate_unwrap_model_compat() -> None:
    from accelerate import Accelerator

    patch_accelerator_class_unwrap_model_compat(Accelerator)


def resolve_trainer_device(trainer, model) -> str:
    trainer_args = getattr(trainer, "args", None)
    trainer_device = getattr(trainer_args, "device", None)
    if trainer_device is not None:
        return str(trainer_device)
    try:
        return str(next(model.parameters()).device)
    except (AttributeError, StopIteration, TypeError):
        return "unknown"


def execute_hf_continued_pretraining(
    cfg: dict[str, Any],
    dataset_manifest: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    patch_accelerate_unwrap_model_compat()
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    class PackedTextDataset(Dataset):
        def __init__(self, texts: list[str], tokenizer, block_size: int):
            eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
            pad_token_id = tokenizer.pad_token_id or eos_token_id
            token_buffer: list[int] = []

            for text in texts:
                token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                token_buffer.extend(token_ids)
                if eos_token_id is not None:
                    token_buffer.append(eos_token_id)

            if len(token_buffer) < 2:
                raise ValueError("Not enough tokens to build a continued pretraining dataset")

            self.examples: list[dict[str, torch.Tensor]] = []
            for start in range(0, len(token_buffer), block_size):
                chunk = token_buffer[start : start + block_size]
                if len(chunk) < 2:
                    continue
                attention_mask = [1] * len(chunk)
                labels = list(chunk)
                if len(chunk) < block_size:
                    padding = [pad_token_id] * (block_size - len(chunk))
                    chunk = chunk + padding
                    attention_mask = attention_mask + [0] * len(padding)
                    labels = labels + [-100] * len(padding)
                self.examples.append(
                    {
                        "input_ids": torch.tensor(chunk, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                )

            if not self.examples:
                raise ValueError("No training examples were created after packing the corpus")

            self.token_count = len(token_buffer)

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return self.examples[index]

    set_seed(int(cfg.get("seed", 42)))

    model_id = str(cfg.get("base_model_id", "sshleifer/tiny-gpt2"))
    tokenizer_name = str(cfg.get("tokenizer_name", model_id))
    block_size = int(cfg.get("block_size", 128))
    output_checkpoints_dir = output_dir / "checkpoints"
    final_model_dir = output_dir / "model"
    output_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = PackedTextDataset(dataset_manifest["resolved_splits"]["train"]["texts"], tokenizer, block_size)
    validation_dataset = PackedTextDataset(dataset_manifest["resolved_splits"]["validation"]["texts"], tokenizer, block_size)
    test_dataset = PackedTextDataset(dataset_manifest["resolved_splits"]["test"]["texts"], tokenizer, block_size)

    model = AutoModelForCausalLM.from_pretrained(model_id)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    training_args = build_training_arguments(
        cfg=cfg,
        output_dir=output_checkpoints_dir,
        train_dataset_size=len(train_dataset),
        training_arguments_cls=TrainingArguments,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )
    resolved_device = resolve_trainer_device(trainer, model)

    train_started_at = time.time()
    train_result = trainer.train()
    train_duration = time.time() - train_started_at
    validation_metrics = with_perplexity(trainer.evaluate(validation_dataset, metric_key_prefix="validation"), "validation")
    test_metrics = with_perplexity(trainer.evaluate(test_dataset, metric_key_prefix="test"), "test")

    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    checkpoint_dirs = sorted(str(path) for path in output_checkpoints_dir.glob("checkpoint-*") if path.is_dir())
    best_checkpoint = trainer.state.best_model_checkpoint or (checkpoint_dirs[-1] if checkpoint_dirs else str(final_model_dir))
    train_metrics = dict(train_result.metrics)
    train_metrics["train_wall_time_seconds"] = round(train_duration, 2)
    tokens_per_second = estimate_tokens_per_second(train_metrics, cfg)
    if tokens_per_second is not None:
        train_metrics["tokens_per_second_estimate"] = tokens_per_second

    return {
        "training_backend": "continued_pretraining",
        "base_model_id": model_id,
        "tokenizer_name": tokenizer_name,
        "resolved_device": resolved_device,
        "block_size": block_size,
        "best_checkpoint": best_checkpoint,
        "final_model_dir": str(final_model_dir),
        "checkpoint_dirs": checkpoint_dirs,
        "train_dataset": {
            "documents": dataset_manifest["resolved_splits"]["train"]["document_count"],
            "tokens": train_dataset.token_count,
        },
        "validation_dataset": {
            "documents": dataset_manifest["resolved_splits"]["validation"]["document_count"],
            "tokens": validation_dataset.token_count,
        },
        "test_dataset": {
            "documents": dataset_manifest["resolved_splits"]["test"]["document_count"],
            "tokens": test_dataset.token_count,
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "log_history": trainer.state.log_history,
    }


def run(
    config_path: str,
    dataset_metadata_path: str,
    output_dir: str | None = None,
    summary_output_path: str | None = None,
) -> str:
    cfg = read_json(config_path)
    backend = str(cfg.get("training_backend", "continued_pretraining"))

    if backend == "simulation":
        return simulate_training.run(
            config_path=config_path,
            dataset_metadata_path=dataset_metadata_path,
            output_dir=output_dir,
            summary_output_path=summary_output_path,
        )
    if backend != "continued_pretraining":
        raise ValueError(f"Unsupported training_backend: {backend}")

    dataset_manifest = load_dataset_manifest(dataset_metadata_path)
    resolved_output_dir, summary_path = resolve_output_paths(output_dir, summary_output_path)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    tags = {
        "dataset_version": str(dataset_manifest.get("dataset_version", "unknown")),
        "config_version": str(cfg.get("config_version", "v1")),
        "pipeline_type": "training",
        "run_id": str(cfg.get("run_id", "local-run")),
        "training_backend": "continued_pretraining",
    }
    experiment_name = f"llmops-training-{cfg.get('project', 'default')}"

    with start_run(experiment_name, tags):
        mlflow.log_params(
            {
                "project": cfg.get("project", "default"),
                "base_model_id": cfg.get("base_model_id", "sshleifer/tiny-gpt2"),
                "tokenizer_name": cfg.get("tokenizer_name", cfg.get("base_model_id", "sshleifer/tiny-gpt2")),
                "epochs": cfg.get("epochs", 1),
                "batch_size": cfg.get("batch_size", 1),
                "learning_rate": cfg.get("learning_rate", 5e-5),
                "block_size": cfg.get("block_size", 128),
                "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 1),
                "dataset_name": dataset_manifest.get("dataset_name", "unknown"),
                "dataset_format": dataset_manifest.get("format", "jsonl"),
                "dataset_text_field": dataset_manifest.get("text_field", "text"),
            }
        )

        summary = execute_hf_continued_pretraining(cfg, dataset_manifest, resolved_output_dir)
        train_metrics = summary.get("train_metrics", {})
        validation_metrics = summary.get("validation_metrics", {})
        test_metrics = summary.get("test_metrics", {})
        if summary.get("resolved_device") is not None:
            mlflow.log_param("resolved_device", str(summary["resolved_device"]))

        if train_metrics.get("train_loss") is not None:
            mlflow.log_metric("train_loss", float(train_metrics["train_loss"]))
        if train_metrics.get("tokens_per_second_estimate") is not None:
            mlflow.log_metric("tokens_per_second", float(train_metrics["tokens_per_second_estimate"]))
        if validation_metrics.get("validation_loss") is not None:
            mlflow.log_metric("validation_loss", float(validation_metrics["validation_loss"]))
        if validation_metrics.get("validation_perplexity") is not None:
            mlflow.log_metric("validation_perplexity", float(validation_metrics["validation_perplexity"]))
        if test_metrics.get("test_loss") is not None:
            mlflow.log_metric("test_loss", float(test_metrics["test_loss"]))
        if test_metrics.get("test_perplexity") is not None:
            mlflow.log_metric("test_perplexity", float(test_metrics["test_perplexity"]))

        summary_path_str = str(summary_path)
        write_json(summary_path_str, summary)
        mlflow.log_artifact(summary_path_str, artifact_path="summaries")

    return str(summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--dataset-metadata-path", required=True)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output-dir")
    output_group.add_argument("--summary-output-path")
    args = parser.parse_args()
    result = run(
        args.config_path,
        args.dataset_metadata_path,
        output_dir=args.output_dir,
        summary_output_path=args.summary_output_path,
    )
    summary_payload = read_json(result)
    print(f"Training resolved device: {summary_payload.get('resolved_device', 'unknown')}")
    print(result)
