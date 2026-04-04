from __future__ import annotations

from kfp import compiler, dsl
from kfp.dsl import Artifact, Output

BASE_IMAGE = "python:3.11-slim"


@dsl.component(base_image=BASE_IMAGE)
def resolve_eval_config_op(eval_config_path: str, resolved_eval_config: Output[Artifact]):
    from pathlib import Path

    Path(resolved_eval_config.path).write_text(Path(eval_config_path).read_text(encoding="utf-8"), encoding="utf-8")


@dsl.component(base_image=BASE_IMAGE)
def load_models_op(candidate_model_uri: str, baseline_model_uri: str, model_info: Output[Artifact]):
    import json
    from pathlib import Path

    Path(model_info.path).write_text(
        json.dumps({"candidate": candidate_model_uri, "baseline": baseline_model_uri}),
        encoding="utf-8",
    )


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["mlflow", "boto3"])
def evaluate_and_log_op(
    resolved_eval_config: Artifact,
    model_info: Artifact,
    scorecard: Output[Artifact],
):
    import json
    import os
    import random
    from pathlib import Path

    import mlflow

    cfg = json.loads(Path(resolved_eval_config.path).read_text(encoding="utf-8"))
    models = json.loads(Path(model_info.path).read_text(encoding="utf-8"))

    intrinsic_perplexity = round(random.uniform(8.0, 25.0), 3)
    public_score = round(random.uniform(0.45, 0.82), 3)
    domain_score = round(random.uniform(0.50, 0.90), 3)
    overall = round((1 / intrinsic_perplexity) * 10 + public_score * 0.45 + domain_score * 0.45, 3)
    baseline_overall = float(cfg.get("baseline_overall_score", 1.0))
    delta = round(overall - baseline_overall, 3)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.llmops.svc.cluster.local:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"llmops-evaluation-{cfg.get('project', 'default')}")

    tags = {
        "dataset_version": str(cfg.get("dataset_version", "unknown")),
        "config_version": str(cfg.get("config_version", "v1")),
        "pipeline_type": "evaluation",
        "run_id": str(cfg.get("run_id", "kfp-eval-local")),
    }

    with mlflow.start_run(tags=tags):
        mlflow.log_params({
            "candidate_model_uri": models["candidate"],
            "baseline_model_uri": models["baseline"],
            "project": cfg.get("project", "default"),
        })
        mlflow.log_metric("intrinsic_perplexity", intrinsic_perplexity)
        mlflow.log_metric("public_benchmark_score", public_score)
        mlflow.log_metric("domain_benchmark_score", domain_score)
        mlflow.log_metric("overall_score", overall)
        mlflow.log_metric("delta_vs_baseline", delta)

        out = {
            "candidate_model_uri": models["candidate"],
            "baseline_model_uri": models["baseline"],
            "intrinsic_perplexity": intrinsic_perplexity,
            "public_benchmark_score": public_score,
            "domain_benchmark_score": domain_score,
            "overall_score": overall,
            "baseline_overall_score": baseline_overall,
            "delta_vs_baseline": delta,
            "result": "pass" if delta >= 0 else "fail",
        }
        Path(scorecard.path).write_text(json.dumps(out), encoding="utf-8")
        mlflow.log_artifact(scorecard.path, artifact_path="scorecards")


@dsl.pipeline(name="llmops-evaluation-pipeline")
def evaluation_pipeline(eval_config_path: str, candidate_model_uri: str, baseline_model_uri: str):
    cfg = resolve_eval_config_op(eval_config_path=eval_config_path)
    models = load_models_op(candidate_model_uri=candidate_model_uri, baseline_model_uri=baseline_model_uri)
    _ = evaluate_and_log_op(
        resolved_eval_config=cfg.outputs["resolved_eval_config"],
        model_info=models.outputs["model_info"],
    )


if __name__ == "__main__":
    compiler.Compiler().compile(evaluation_pipeline, "pipelines/evaluation/compiled/evaluation_pipeline.yaml")
