import os
from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import Artifact, Input, Output

BASE_IMAGE = os.getenv("LLMOPS_BASE_IMAGE", "llmops/base:dev")
EVALUATOR_IMAGE = os.getenv("LLMOPS_EVALUATOR_IMAGE", "llmops/evaluator:dev")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.llmops.svc.cluster.local:5000")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio.llmops.svc.cluster.local:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
QWEN_INFERENCE_URL = os.getenv("QWEN_INFERENCE_URL", "http://qwen-inference.llmops.svc.cluster.local:8000/v1/generate")
QWEN_CANDIDATE_INFERENCE_URL = os.getenv(
    "QWEN_CANDIDATE_INFERENCE_URL",
    "http://qwen-candidate.llmops.svc.cluster.local:8000/v1/generate",
)
QWEN_BASELINE_INFERENCE_URL = os.getenv(
    "QWEN_BASELINE_INFERENCE_URL",
    "http://qwen-baseline.llmops.svc.cluster.local:8000/v1/generate",
)


def set_runtime_env(task):
    task.set_env_variable(name="MLFLOW_TRACKING_URI", value=MLFLOW_TRACKING_URI)
    task.set_env_variable(name="MLFLOW_S3_ENDPOINT_URL", value=MLFLOW_S3_ENDPOINT_URL)
    task.set_env_variable(name="AWS_ACCESS_KEY_ID", value=AWS_ACCESS_KEY_ID)
    task.set_env_variable(name="AWS_SECRET_ACCESS_KEY", value=AWS_SECRET_ACCESS_KEY)
    task.set_env_variable(name="AWS_DEFAULT_REGION", value=AWS_DEFAULT_REGION)
    task.set_env_variable(name="QWEN_INFERENCE_URL", value=QWEN_INFERENCE_URL)
    task.set_env_variable(name="QWEN_CANDIDATE_INFERENCE_URL", value=QWEN_CANDIDATE_INFERENCE_URL)
    task.set_env_variable(name="QWEN_BASELINE_INFERENCE_URL", value=QWEN_BASELINE_INFERENCE_URL)
    return task


@dsl.component(base_image=BASE_IMAGE)
def resolve_eval_config_op(eval_config_path: str, resolved_eval_config: Output[Artifact]):
    from pathlib import Path

    Path(resolved_eval_config.path).write_text(Path(eval_config_path).read_text(encoding="utf-8"), encoding="utf-8")


@dsl.container_component
def evaluate_and_log_op(
    resolved_eval_config: Input[Artifact],
    candidate_model_uri: str,
    baseline_model_uri: str,
    scorecard: Output[Artifact],
):
    return dsl.ContainerSpec(
        image=EVALUATOR_IMAGE,
        command=["python", "-m", "components.evaluator.run_evaluation"],
        args=[
            "--eval-config-path",
            resolved_eval_config.path,
            "--candidate-model-uri",
            candidate_model_uri,
            "--baseline-model-uri",
            baseline_model_uri,
            "--scorecard-output-path",
            scorecard.path,
        ],
    )


@dsl.pipeline(name="llmops-evaluation-pipeline")
def evaluation_pipeline(eval_config_path: str, candidate_model_uri: str, baseline_model_uri: str):
    cfg = resolve_eval_config_op(eval_config_path=eval_config_path)
    _ = set_runtime_env(
        evaluate_and_log_op(
            resolved_eval_config=cfg.outputs["resolved_eval_config"],
            candidate_model_uri=candidate_model_uri,
            baseline_model_uri=baseline_model_uri,
        )
    )


if __name__ == "__main__":
    Path("pipelines/evaluation/compiled").mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(evaluation_pipeline, "pipelines/evaluation/compiled/evaluation_pipeline.yaml")
