#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.request import urlopen


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_TORCH_THREADS = 2
DEFAULT_DEVICE = "auto"
QWEN_SERVICE_REQUIRED_MODULES = ["mlflow", "fastapi", "uvicorn", "transformers", "torch"]
CPT_REQUIRED_MODULES = ["mlflow", "transformers", "torch"]


def missing_modules(required_modules: list[str]) -> list[str]:
    return [module for module in required_modules if importlib.util.find_spec(module) is None]


def ensure_local_eval_dependencies(backend: str) -> None:
    required_modules = QWEN_SERVICE_REQUIRED_MODULES if backend == "qwen_service" else CPT_REQUIRED_MODULES
    missing = missing_modules(required_modules)
    if not missing:
        return
    missing_display = ", ".join(missing)
    raise SystemExit(
        "Missing local evaluation dependencies: "
        f"{missing_display}\n"
        "Install them with:\n"
        "  uv sync --extra dev --extra inference"
    )


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return sanitized or "local-eval"


def resolve_model_source(model_id: str | None, local_model_path: str | None) -> str:
    if model_id and local_model_path:
        raise ValueError("Use either model_id or local_model_path, not both.")
    if local_model_path:
        resolved_path = Path(local_model_path).resolve()
        if not resolved_path.exists():
            raise SystemExit(
                "Local model path does not exist: "
                f"{resolved_path}\n"
                "Run local training first, or point LOCAL_MODEL_PATH at a saved Hugging Face model directory."
            )
        if not resolved_path.is_dir():
            raise SystemExit(f"Local model path must be a directory: {resolved_path}")
        return str(resolved_path)
    if model_id:
        return model_id
    return DEFAULT_MODEL_ID


def default_model_uri(model_id: str | None, local_model_path: str | None) -> str:
    if local_model_path:
        return f"file://{Path(local_model_path).resolve()}"
    if model_id:
        return f"hf://{model_id}"
    return f"hf://{DEFAULT_MODEL_ID}"


def default_output_dir(run_id: str, repo_root: Path) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return repo_root / "artifacts" / "local-eval" / f"{sanitize_name(run_id)}-{timestamp}"


def build_local_eval_config(base_cfg: dict, endpoint: str) -> dict:
    cfg = dict(base_cfg)
    cfg["evaluation_mode"] = "single_model"
    cfg.pop("candidate_inference_endpoint", None)
    cfg.pop("baseline_inference_endpoint", None)
    cfg.pop("baseline_overall_score", None)
    cfg.pop("minimum_delta_vs_baseline", None)
    if str(cfg.get("evaluation_backend", "qwen_service")) == "qwen_service":
        cfg["evaluation_backend"] = "qwen_service"
        cfg["inference_endpoint"] = endpoint
    return cfg


def wait_for_healthcheck(url: str, process: subprocess.Popen[bytes], timeout_seconds: int) -> dict[str, str]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Local inference server exited early with code {process.returncode}")
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return json.loads(response.read().decode("utf-8"))
        except URLError:
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for local inference server health check: {url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-model evaluation locally on macOS/Linux without Kubernetes.")
    parser.add_argument(
        "--eval-config-path",
        default="configs/evaluation/single_model.sample.json",
        help="Base evaluation config JSON path. The script overrides inference_endpoint locally.",
    )
    parser.add_argument("--model-id", help=f"Hugging Face model ID. Defaults to {DEFAULT_MODEL_ID}.")
    parser.add_argument("--local-model-path", help="Local model directory to load with from_pretrained().")
    parser.add_argument("--model-uri", help="Logical model URI recorded in the scorecard and MLflow.")
    parser.add_argument("--output-dir", help="Directory to write scorecard/model-card artifacts.")
    parser.add_argument("--mlflow-tracking-uri", help="Optional MLflow tracking URI. Defaults to a local file store.")
    parser.add_argument("--hf-home", help="Optional Hugging Face cache directory. Defaults to .cache/hf in the repo.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind the local inference server.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for the local inference server.")
    parser.add_argument(
        "--server-start-timeout",
        type=int,
        default=900,
        help="Seconds to wait for the local inference server to become healthy.",
    )
    parser.add_argument("--torch-threads", type=int, default=DEFAULT_TORCH_THREADS)
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device passed to QWEN_DEVICE, e.g. auto, cpu, or mps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    eval_config_path = Path(args.eval_config_path).resolve()
    if not eval_config_path.exists():
        raise SystemExit(f"Evaluation config not found: {eval_config_path}")

    with eval_config_path.open("r", encoding="utf-8") as handle:
        base_cfg = json.load(handle)
    backend = str(base_cfg.get("evaluation_backend", "qwen_service"))
    ensure_local_eval_dependencies(backend)

    model_source = resolve_model_source(args.model_id, args.local_model_path)
    model_uri = args.model_uri or default_model_uri(args.model_id, args.local_model_path)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(
        str(base_cfg.get("run_id", "local-eval")),
        repo_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_home = Path(args.hf_home).resolve() if args.hf_home else (repo_root / ".cache" / "hf")
    hf_home.mkdir(parents=True, exist_ok=True)

    endpoint = f"http://{args.host}:{args.port}/v1/generate"
    healthcheck_url = f"http://{args.host}:{args.port}/healthz"
    local_cfg = build_local_eval_config(base_cfg, endpoint)

    mlflow_tracking_uri = args.mlflow_tracking_uri or f"sqlite:///{(output_dir / 'mlflow.db').resolve()}"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["QWEN_MODEL_ID"] = model_source
    env["QWEN_TORCH_THREADS"] = str(args.torch_threads)
    env["QWEN_DEVICE"] = args.device
    env["HF_HOME"] = str(hf_home)
    env["TRANSFORMERS_CACHE"] = str(hf_home)
    env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri

    server_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "components.inference.qwen_service:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    if backend == "qwen_service":
        print(f"Starting local inference server for {model_source}")
        server = subprocess.Popen(server_cmd, cwd=str(repo_root), env=env)
        try:
            health_status = wait_for_healthcheck(healthcheck_url, server, args.server_start_timeout)
            resolved_device = health_status.get("resolved_device", "unknown")
            requested_device = health_status.get("requested_device", args.device)
            print(
                f"Local inference endpoint is ready at {endpoint} "
                f"(requested device: {requested_device}, resolved device: {resolved_device})"
            )

            with TemporaryDirectory(prefix="llmops-local-eval-") as temp_dir:
                temp_cfg_path = Path(temp_dir) / "single_model_eval.local.json"
                temp_cfg_path.write_text(json.dumps(local_cfg, indent=2), encoding="utf-8")

                from components.evaluator.run_single_model_evaluation import run as run_single_model_evaluation

                scorecard_path = run_single_model_evaluation(
                    eval_config_path=str(temp_cfg_path),
                    model_uri=model_uri,
                    output_dir=str(output_dir),
                )

            model_card_path = output_dir / "model_cards" / "model_card.md"
            print(f"Scorecard: {scorecard_path}")
            print(f"Model card: {model_card_path}")
            print(f"MLflow tracking URI: {mlflow_tracking_uri}")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        return

    print(f"Running local CPT evaluation for {model_source}")
    print(f"Local CPT evaluation uses model loading directly (requested device: {args.device}).")
    with TemporaryDirectory(prefix="llmops-local-eval-") as temp_dir:
        temp_cfg_path = Path(temp_dir) / "single_model_eval.local.json"
        temp_cfg_path.write_text(json.dumps(local_cfg, indent=2), encoding="utf-8")

        from components.evaluator.run_single_model_evaluation import run as run_single_model_evaluation

        scorecard_path = run_single_model_evaluation(
            eval_config_path=str(temp_cfg_path),
            model_uri=model_uri,
            output_dir=str(output_dir),
        )

    model_card_path = output_dir / "model_cards" / "model_card.md"
    print(f"Scorecard: {scorecard_path}")
    print(f"Model card: {model_card_path}")
    print(f"MLflow tracking URI: {mlflow_tracking_uri}")


if __name__ == "__main__":
    main()
