from __future__ import annotations

import argparse
import random
from pathlib import Path

import mlflow

from components.utils.io_utils import read_json, write_json
from components.utils.mlflow_utils import start_run


def run(eval_config_path: str, candidate_model_uri: str, baseline_model_uri: str, output_dir: str) -> str:
    cfg = read_json(eval_config_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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

    scorecard_path = str(Path(output_dir) / "scorecard.json")
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

        write_json(scorecard_path, scorecard)
        mlflow.log_artifact(scorecard_path, artifact_path="scorecards")

    return scorecard_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-config-path", required=True)
    parser.add_argument("--candidate-model-uri", required=True)
    parser.add_argument("--baseline-model-uri", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(args.eval_config_path, args.candidate_model_uri, args.baseline_model_uri, args.output_dir)
    print(result)
