from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import mlflow

from components.utils.io_utils import read_json, write_json
from components.utils.mlflow_utils import start_run


def simulate_losses(epoch: int) -> tuple[float, float]:
    train_loss = max(0.1, 3.0 * math.exp(-0.35 * epoch) + random.uniform(-0.05, 0.05))
    val_loss = max(0.1, 3.3 * math.exp(-0.30 * epoch) + random.uniform(-0.08, 0.08))
    return round(train_loss, 4), round(val_loss, 4)


def run(config_path: str, dataset_metadata_path: str, output_dir: str) -> str:
    cfg = read_json(config_path)
    ds = read_json(dataset_metadata_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(output_dir) / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    tags = {
        "dataset_version": str(ds.get("dataset_version", "unknown")),
        "config_version": str(cfg.get("config_version", "v1")),
        "pipeline_type": "training",
        "run_id": str(cfg.get("run_id", "local-run")),
    }
    experiment_name = f"llmops-training-{cfg.get('project', 'default')}"

    best_ckpt = ""
    best_val = float("inf")
    epochs = int(cfg.get("epochs", 5))

    with start_run(experiment_name, tags):
        mlflow.log_params(
            {
                "project": cfg.get("project", "default"),
                "epochs": epochs,
                "batch_size": cfg.get("batch_size", 4),
                "learning_rate": cfg.get("learning_rate", 0.0005),
                "dataset_name": ds.get("dataset_name", "unknown"),
            }
        )

        metrics = []
        for epoch in range(1, epochs + 1):
            train_loss, val_loss = simulate_losses(epoch)
            tokens_per_sec = round(2500 + random.uniform(-150, 150), 2)

            ckpt = checkpoints_dir / f"checkpoint_epoch_{epoch}.ckpt"
            ckpt.write_text(f"fake checkpoint for epoch {epoch}\n", encoding="utf-8")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("tokens_per_sec", tokens_per_sec, step=epoch)
            mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")

            metrics.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "tokens_per_sec": tokens_per_sec,
                    "checkpoint": str(ckpt),
                }
            )
            if val_loss < best_val:
                best_val = val_loss
                best_ckpt = str(ckpt)

        summary_path = str(Path(output_dir) / "training_summary.json")
        write_json(
            summary_path,
            {
                "best_checkpoint": best_ckpt,
                "best_val_loss": best_val,
                "metrics": metrics,
            },
        )
        mlflow.log_artifact(summary_path, artifact_path="summaries")

    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--dataset-metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(args.config_path, args.dataset_metadata_path, args.output_dir)
    print(result)
