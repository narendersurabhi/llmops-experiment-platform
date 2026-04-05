import os
from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import Artifact, Input, Output

BASE_IMAGE = os.getenv("LLMOPS_BASE_IMAGE", "llmops/base:dev")
TRAINER_IMAGE = os.getenv("LLMOPS_TRAINER_IMAGE", "llmops/trainer:dev")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.llmops.svc.cluster.local:5000")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio.llmops.svc.cluster.local:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def set_runtime_env(task):
    task.set_env_variable(name="MLFLOW_TRACKING_URI", value=MLFLOW_TRACKING_URI)
    task.set_env_variable(name="MLFLOW_S3_ENDPOINT_URL", value=MLFLOW_S3_ENDPOINT_URL)
    task.set_env_variable(name="AWS_ACCESS_KEY_ID", value=AWS_ACCESS_KEY_ID)
    task.set_env_variable(name="AWS_SECRET_ACCESS_KEY", value=AWS_SECRET_ACCESS_KEY)
    task.set_env_variable(name="AWS_DEFAULT_REGION", value=AWS_DEFAULT_REGION)
    return task


@dsl.component(base_image=BASE_IMAGE)
def resolve_config_op(run_config_path: str, resolved_config: Output[Artifact]):
    import json
    from pathlib import Path

    cfg = json.loads(Path(run_config_path).read_text(encoding="utf-8"))
    Path(resolved_config.path).write_text(json.dumps(cfg), encoding="utf-8")


@dsl.component(base_image=BASE_IMAGE)
def validate_inputs_op(resolved_config: Input[Artifact], dataset_metadata_path: str):
    import json
    from pathlib import Path

    import jsonschema

    run_schema = json.loads(Path("schemas/run.schema.json").read_text(encoding="utf-8"))
    dataset_schema = json.loads(Path("schemas/dataset.schema.json").read_text(encoding="utf-8"))
    cfg = json.loads(Path(resolved_config.path).read_text(encoding="utf-8"))
    ds = json.loads(Path(dataset_metadata_path).read_text(encoding="utf-8"))

    jsonschema.validate(cfg, run_schema)
    jsonschema.validate(ds, dataset_schema)

    manifest_dir = Path(dataset_metadata_path).resolve().parent
    for split_name, split_ref in ds["splits"].items():
        split_path = Path(split_ref)
        if not split_path.is_absolute():
            split_path = manifest_dir / split_path
        if not split_path.exists():
            raise FileNotFoundError(f"Missing dataset split {split_name}: {split_path}")


@dsl.component(base_image=BASE_IMAGE)
def tokenization_check_op(resolved_config: Input[Artifact], tokenization_report: Output[Artifact]):
    import json
    from pathlib import Path

    cfg = json.loads(Path(resolved_config.path).read_text(encoding="utf-8"))
    report = {
        "status": "ok",
        "training_backend": cfg.get("training_backend", "continued_pretraining"),
        "base_model_id": cfg.get("base_model_id", "sshleifer/tiny-gpt2"),
        "tokenizer_name": cfg.get("tokenizer_name", cfg.get("base_model_id", "sshleifer/tiny-gpt2")),
        "block_size": int(cfg.get("block_size", 128)),
    }
    Path(tokenization_report.path).write_text(json.dumps(report), encoding="utf-8")


@dsl.container_component
def launch_training_job_op(
    resolved_config: Input[Artifact],
    dataset_metadata_path: str,
    training_summary: Output[Artifact],
):
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE,
        command=["python", "-m", "components.trainer.run_pretraining"],
        args=[
            "--config-path",
            resolved_config.path,
            "--dataset-metadata-path",
            dataset_metadata_path,
            "--summary-output-path",
            training_summary.path,
        ],
    )


@dsl.component(base_image=BASE_IMAGE)
def select_best_checkpoint_op(training_summary: Input[Artifact], best_checkpoint: Output[Artifact]):
    import json
    from pathlib import Path

    summary = json.loads(Path(training_summary.path).read_text(encoding="utf-8"))
    Path(best_checkpoint.path).write_text(summary["best_checkpoint"], encoding="utf-8")


@dsl.pipeline(name="llmops-training-pipeline")
def training_pipeline(run_config_path: str, dataset_metadata_path: str):
    resolved = resolve_config_op(run_config_path=run_config_path)
    validate = validate_inputs_op(resolved_config=resolved.outputs["resolved_config"], dataset_metadata_path=dataset_metadata_path)
    tok = tokenization_check_op(resolved_config=resolved.outputs["resolved_config"])
    train = set_runtime_env(
        launch_training_job_op(
            resolved_config=resolved.outputs["resolved_config"],
            dataset_metadata_path=dataset_metadata_path,
        )
    )
    train.after(validate, tok)
    _ = select_best_checkpoint_op(training_summary=train.outputs["training_summary"])


if __name__ == "__main__":
    Path("pipelines/training/compiled").mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(training_pipeline, "pipelines/training/compiled/training_pipeline.yaml")
