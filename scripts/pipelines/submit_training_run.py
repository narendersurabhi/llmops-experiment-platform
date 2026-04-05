#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from kfp import Client

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.pipelines.kfp_local_compat import apply_local_k8s_workaround, local_ui_metadata_note


DEFAULT_PIPELINE = "pipelines/training/compiled/training_pipeline.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit the training pipeline to Kubeflow Pipelines.")
    parser.add_argument("--host", default=os.getenv("KFP_HOST"), help="Kubeflow Pipelines API host URL.")
    parser.add_argument("--pipeline-package", default=DEFAULT_PIPELINE, help="Path to compiled pipeline package.")
    parser.add_argument("--run-name", default="llmops-training-manual", help="Display name for the pipeline run.")
    parser.add_argument("--experiment-name", default="llmops-training", help="KFP experiment name.")
    parser.add_argument(
        "--run-config-path",
        default="configs/training/run.sample.json",
        help="Training run configuration JSON path.",
    )
    parser.add_argument(
        "--dataset-metadata-path",
        default="dataset/metadata.sample.json",
        help="Dataset metadata JSON path.",
    )
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
            "run_config_path": args.run_config_path,
            "dataset_metadata_path": args.dataset_metadata_path,
        },
        run_name=args.run_name,
        experiment_id=experiment.experiment_id,
    )

    workflow_name = apply_local_k8s_workaround(run.run_id)
    print(f"Training run submitted: {run.run_id}")
    if workflow_name:
        print(f"Workflow: {workflow_name}")
        print(local_ui_metadata_note())


if __name__ == "__main__":
    main()
