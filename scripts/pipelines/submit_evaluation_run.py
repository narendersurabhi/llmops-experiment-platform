#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

from kfp import Client


DEFAULT_PIPELINE = "pipelines/evaluation/compiled/evaluation_pipeline.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit the evaluation pipeline to Kubeflow Pipelines.")
    parser.add_argument("--host", default=os.getenv("KFP_HOST"), help="Kubeflow Pipelines API host URL.")
    parser.add_argument("--pipeline-package", default=DEFAULT_PIPELINE, help="Path to compiled pipeline package.")
    parser.add_argument("--run-name", default="llmops-evaluation-manual", help="Display name for the pipeline run.")
    parser.add_argument("--experiment-name", default="llmops-evaluation", help="KFP experiment name.")
    parser.add_argument(
        "--eval-config-path",
        default="configs/evaluation/eval.sample.json",
        help="Evaluation configuration JSON path.",
    )
    parser.add_argument("--candidate-model-uri", required=True, help="Candidate model URI.")
    parser.add_argument("--baseline-model-uri", required=True, help="Baseline model URI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.host:
        raise SystemExit("Missing --host or KFP_HOST environment variable.")

    client = Client(host=args.host)
    experiment = client.create_experiment(name=args.experiment_name)
    run = client.create_run_from_pipeline_package(
        pipeline_file=args.pipeline_package,
        arguments={
            "eval_config_path": args.eval_config_path,
            "candidate_model_uri": args.candidate_model_uri,
            "baseline_model_uri": args.baseline_model_uri,
        },
        run_name=args.run_name,
        experiment_id=experiment.experiment_id,
    )

    print(f"Evaluation run submitted: {run.run_id}")


if __name__ == "__main__":
    main()
