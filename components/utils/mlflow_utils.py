from __future__ import annotations

import os
from typing import Dict

import mlflow


def configure_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    mlflow.set_tracking_uri(tracking_uri)


def start_run(experiment_name: str, tags: Dict[str, str]):
    configure_mlflow()
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(tags=tags)
