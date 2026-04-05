from __future__ import annotations

import argparse
from pathlib import Path

from components.evaluator.run_evaluation import (
    resolve_output_paths,
    run_single_model_cpt_evaluation,
    run_single_model_qwen_service_evaluation,
)
from components.utils.io_utils import read_json


def run(
    eval_config_path: str,
    model_uri: str,
    output_dir: str | None = None,
    scorecard_output_path: str | None = None,
) -> str:
    cfg = read_json(eval_config_path)
    cfg["_eval_config_path"] = str(Path(eval_config_path).resolve())
    backend = str(cfg.get("evaluation_backend", "qwen_service"))

    resolved_output_dir, scorecard_path = resolve_output_paths(output_dir, scorecard_output_path)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    if backend == "qwen_service":
        return run_single_model_qwen_service_evaluation(cfg, model_uri, scorecard_path)
    if backend == "cpt_model":
        return run_single_model_cpt_evaluation(cfg, model_uri, scorecard_path)
    raise ValueError(f"Unsupported evaluation_backend for single-model evaluation: {backend}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-config-path", required=True)
    parser.add_argument("--model-uri", required=True)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output-dir")
    output_group.add_argument("--scorecard-output-path")
    args = parser.parse_args()
    result = run(
        args.eval_config_path,
        args.model_uri,
        output_dir=args.output_dir,
        scorecard_output_path=args.scorecard_output_path,
    )
    print(result)
