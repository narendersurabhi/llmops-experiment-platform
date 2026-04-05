from __future__ import annotations

from pathlib import Path

import scripts.local.run_single_model_eval_local as local_eval_script
from scripts.local.run_single_model_eval_local import (
    build_local_eval_config,
    default_model_uri,
    resolve_model_source,
    wait_for_healthcheck,
)


def test_build_local_eval_config_forces_single_model_endpoint_mode():
    cfg = {
        "project": "p",
        "run_id": "r1",
        "evaluation_mode": "compare",
        "evaluation_backend": "qwen_service",
        "candidate_inference_endpoint": "http://candidate",
        "baseline_inference_endpoint": "http://baseline",
        "baseline_overall_score": 1.0,
        "minimum_delta_vs_baseline": 0.0,
    }

    rendered = build_local_eval_config(cfg, "http://127.0.0.1:8000/v1/generate")

    assert rendered["evaluation_mode"] == "single_model"
    assert rendered["inference_endpoint"] == "http://127.0.0.1:8000/v1/generate"
    assert "candidate_inference_endpoint" not in rendered
    assert "baseline_inference_endpoint" not in rendered
    assert "baseline_overall_score" not in rendered
    assert "minimum_delta_vs_baseline" not in rendered


def test_build_local_eval_config_preserves_cpt_backend_without_endpoint_override():
    cfg = {
        "project": "p",
        "run_id": "r2",
        "evaluation_mode": "single_model",
        "evaluation_backend": "cpt_model",
        "dataset_metadata_path": "dataset/metadata.qwen2_5_0_5b.public.json",
    }

    rendered = build_local_eval_config(cfg, "http://127.0.0.1:8000/v1/generate")

    assert rendered["evaluation_mode"] == "single_model"
    assert rendered["evaluation_backend"] == "cpt_model"
    assert rendered["dataset_metadata_path"] == "dataset/metadata.qwen2_5_0_5b.public.json"
    assert "inference_endpoint" not in rendered


def test_default_model_uri_prefers_local_path():
    uri = default_model_uri(None, "/tmp/my-model")
    assert uri == f"file://{Path('/tmp/my-model').resolve()}"


def test_resolve_model_source_defaults_to_base_model():
    assert resolve_model_source(None, None) == "Qwen/Qwen2.5-0.5B"


def test_wait_for_healthcheck_returns_status_payload(monkeypatch):
    class DummyProcess:
        def poll(self):
            return None

    class DummyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"status":"ok","resolved_device":"mps","requested_device":"auto"}'

    monkeypatch.setattr(local_eval_script, "urlopen", lambda *args, **kwargs: DummyResponse())

    payload = wait_for_healthcheck("http://127.0.0.1:8000/healthz", DummyProcess(), 1)

    assert payload["status"] == "ok"
    assert payload["resolved_device"] == "mps"
    assert payload["requested_device"] == "auto"
