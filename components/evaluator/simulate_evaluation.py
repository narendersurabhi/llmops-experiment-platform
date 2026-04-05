from __future__ import annotations

import argparse
import random
from pathlib import Path

import mlflow

from components.utils.io_utils import read_json, write_json
from components.utils.mlflow_utils import start_run


def resolve_output_paths(output_dir: str | None, scorecard_output_path: str | None) -> tuple[Path, Path]:
    if output_dir is None and scorecard_output_path is None:
        raise ValueError("Either output_dir or scorecard_output_path is required.")

    if scorecard_output_path is not None:
        scorecard_path = Path(scorecard_output_path)
        resolved_output_dir = scorecard_path.parent
    else:
        resolved_output_dir = Path(output_dir)
        scorecard_path = resolved_output_dir / "scorecard.json"

    if output_dir is not None and Path(output_dir) != resolved_output_dir:
        raise ValueError("output_dir must match the parent directory of scorecard_output_path.")

    return resolved_output_dir, scorecard_path


def run(
    eval_config_path: str,
    candidate_model_uri: str,
    baseline_model_uri: str,
    output_dir: str | None = None,
    scorecard_output_path: str | None = None,
) -> str:
    cfg = read_json(eval_config_path)
    resolved_output_dir, scorecard_path = resolve_output_paths(output_dir, scorecard_output_path)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    intrinsic_perplexity = round(random.uniform(8.0, 25.0), 3)
    public_score = round(random.uniform(0.45, 0.82), 3)
    domain_score = round(random.uniform(0.50, 0.90), 3)
    overall = round((1 / intrinsic_perplexity) * 10 + public_score * 0.45 + domain_score * 0.45, 3)

    baseline_overall = cfg.get("baseline_overall_score", 1.0)
    delta = round(overall - float(baseline_overall), 3)

    tags = {
        "dataset_version": str(cfg.get("dataset_version", "unknown")),
        "config_version": str(cfg.get("config_version", "v1")),
        "pipeline_type": "evaluation",
        "run_id": str(cfg.get("run_id", "local-eval")),
    }
    experiment_name = f"llmops-evaluation-{cfg.get('project', 'default')}"

    scorecard = {
        "candidate_model_uri": candidate_model_uri,
        "baseline_model_uri": baseline_model_uri,
        "intrinsic_perplexity": intrinsic_perplexity,
        "public_benchmark_score": public_score,
        "domain_benchmark_score": domain_score,
        "overall_score": overall,
        "baseline_overall_score": baseline_overall,
        "delta_vs_baseline": delta,
        "result": "pass" if delta >= 0 else "fail",
    }

    with start_run(experiment_name, tags):
        mlflow.log_params(
            {
                "candidate_model_uri": candidate_model_uri,
                "baseline_model_uri": baseline_model_uri,
                "project": cfg.get("project", "default"),
            }
        )
        mlflow.log_metric("intrinsic_perplexity", intrinsic_perplexity)
        mlflow.log_metric("public_benchmark_score", public_score)
        mlflow.log_metric("domain_benchmark_score", domain_score)
        mlflow.log_metric("overall_score", overall)
        mlflow.log_metric("delta_vs_baseline", delta)

        scorecard_path_str = str(scorecard_path)
        write_json(scorecard_path_str, scorecard)
        mlflow.log_artifact(scorecard_path_str, artifact_path="scorecards")

    return scorecard_path_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-config-path", required=True)
    parser.add_argument("--candidate-model-uri", required=True)
    parser.add_argument("--baseline-model-uri", required=True)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output-dir")
    output_group.add_argument("--scorecard-output-path")
    args = parser.parse_args()
    result = run(
        args.eval_config_path,
        args.candidate_model_uri,
        args.baseline_model_uri,
        output_dir=args.output_dir,
        scorecard_output_path=args.scorecard_output_path,
    )
    print(result)
