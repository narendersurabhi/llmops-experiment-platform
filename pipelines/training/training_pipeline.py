from __future__ import annotations

from kfp import compiler, dsl
from kfp.dsl import Output, Artifact

BASE_IMAGE = "python:3.11-slim"


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["jsonschema"])
def resolve_config_op(run_config_path: str, resolved_config: Output[Artifact]):
    import json
    from pathlib import Path

    cfg = json.loads(Path(run_config_path).read_text(encoding="utf-8"))
    Path(resolved_config.path).write_text(json.dumps(cfg), encoding="utf-8")


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["jsonschema"])
def validate_inputs_op(resolved_config: Artifact, dataset_metadata_path: str):
    import json
    from pathlib import Path

    _ = json.loads(Path(resolved_config.path).read_text(encoding="utf-8"))
    _ = json.loads(Path(dataset_metadata_path).read_text(encoding="utf-8"))


@dsl.component(base_image=BASE_IMAGE)
def tokenization_check_op(resolved_config: Artifact, tokenization_report: Output[Artifact]):
    import json
    from pathlib import Path

    cfg = json.loads(Path(resolved_config.path).read_text(encoding="utf-8"))
    report = {"status": "ok", "tokenizer": cfg.get("tokenizer", "mock-tokenizer")}
    Path(tokenization_report.path).write_text(json.dumps(report), encoding="utf-8")


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["mlflow", "boto3"])
def launch_training_job_op(
    resolved_config: Artifact,
    dataset_metadata_path: str,
    training_summary: Output[Artifact],
):
    import json
    import math
    import os
    import random
    from pathlib import Path

    import mlflow

    cfg = json.loads(Path(resolved_config.path).read_text(encoding="utf-8"))
    ds = json.loads(Path(dataset_metadata_path).read_text(encoding="utf-8"))

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.llmops.svc.cluster.local:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"llmops-training-{cfg.get('project', 'default')}")

    out_dir = Path(training_summary.path).parent
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tags = {
        "dataset_version": str(ds.get("dataset_version", "unknown")),
        "config_version": str(cfg.get("config_version", "v1")),
        "pipeline_type": "training",
        "run_id": str(cfg.get("run_id", "kfp-local")),
    }

    best_val = float("inf")
    best_ckpt = ""
    metrics = []

    with mlflow.start_run(tags=tags):
        mlflow.log_params(cfg)
        epochs = int(cfg.get("epochs", 5))
        for epoch in range(1, epochs + 1):
            train_loss = max(0.1, 3.0 * math.exp(-0.35 * epoch) + random.uniform(-0.05, 0.05))
            val_loss = max(0.1, 3.3 * math.exp(-0.3 * epoch) + random.uniform(-0.08, 0.08))
            tps = round(2500 + random.uniform(-150, 150), 2)

            ckpt = ckpt_dir / f"checkpoint_epoch_{epoch}.ckpt"
            ckpt.write_text(f"fake checkpoint for epoch {epoch}\n", encoding="utf-8")

            mlflow.log_metric("train_loss", float(train_loss), step=epoch)
            mlflow.log_metric("val_loss", float(val_loss), step=epoch)
            mlflow.log_metric("tokens_per_sec", tps, step=epoch)
            mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")

            if val_loss < best_val:
                best_val = val_loss
                best_ckpt = str(ckpt)

            metrics.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "tokens_per_sec": tps})

        summary = {"best_checkpoint": best_ckpt, "best_val_loss": best_val, "metrics": metrics}
        Path(training_summary.path).write_text(json.dumps(summary), encoding="utf-8")
        mlflow.log_artifact(training_summary.path, artifact_path="summaries")


@dsl.component(base_image=BASE_IMAGE)
def select_best_checkpoint_op(training_summary: Artifact, best_checkpoint: Output[Artifact]):
    import json
    from pathlib import Path

    summary = json.loads(Path(training_summary.path).read_text(encoding="utf-8"))
    Path(best_checkpoint.path).write_text(summary["best_checkpoint"], encoding="utf-8")


@dsl.pipeline(name="llmops-training-pipeline")
def training_pipeline(run_config_path: str, dataset_metadata_path: str):
    resolved = resolve_config_op(run_config_path=run_config_path)
    validate = validate_inputs_op(resolved_config=resolved.outputs["resolved_config"], dataset_metadata_path=dataset_metadata_path)
    tok = tokenization_check_op(resolved_config=resolved.outputs["resolved_config"])
    train = launch_training_job_op(
        resolved_config=resolved.outputs["resolved_config"],
        dataset_metadata_path=dataset_metadata_path,
    )
    train.after(validate, tok)
    _ = select_best_checkpoint_op(training_summary=train.outputs["training_summary"])


if __name__ == "__main__":
    compiler.Compiler().compile(training_pipeline, "pipelines/training/compiled/training_pipeline.yaml")
