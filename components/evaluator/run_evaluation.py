from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import mlflow

from components.evaluator import simulate_evaluation
from components.evaluator.simulate_evaluation import resolve_output_paths
from components.utils.io_utils import read_json, write_json
from components.utils.mlflow_utils import start_run


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise technical assistant. Answer clearly, stay factual, and prefer specific engineering terms."
)
DEFAULT_INFERENCE_ENDPOINT = "http://qwen-inference.llmops.svc.cluster.local:8000/v1/generate"
DEFAULT_CANDIDATE_INFERENCE_ENDPOINT = "http://qwen-candidate.llmops.svc.cluster.local:8000/v1/generate"
DEFAULT_BASELINE_INFERENCE_ENDPOINT = "http://qwen-baseline.llmops.svc.cluster.local:8000/v1/generate"
TOKEN_RE = re.compile(r"[a-z0-9]+")
DEFAULT_BENCHMARK_PROMPTS = [
    {
        "id": "public-kubernetes",
        "suite": "public",
        "prompt": "Explain what Kubernetes does for containerized applications in three sentences.",
        "expected_keywords": ["kubernetes", "container", "orchestration"],
        "minimum_response_chars": 120,
    },
    {
        "id": "public-mlflow",
        "suite": "public",
        "prompt": "Describe what MLflow tracks during an ML experiment run.",
        "expected_keywords": ["mlflow", "metrics", "artifacts", "experiments"],
        "minimum_response_chars": 110,
    },
    {
        "id": "domain-kubeflow-mlflow",
        "suite": "domain",
        "prompt": "In this repository, explain how Kubeflow Pipelines and MLflow work together during evaluation.",
        "expected_keywords": ["kubeflow", "mlflow", "pipeline", "artifacts"],
        "minimum_response_chars": 130,
    },
    {
        "id": "domain-minio",
        "suite": "domain",
        "prompt": "Explain why MinIO is part of this platform and what it stores.",
        "expected_keywords": ["minio", "artifact", "s3", "storage"],
        "minimum_response_chars": 110,
    },
]


class BatchEndpointUnavailable(RuntimeError):
    pass


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def maybe_round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def normalize_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def resolve_eval_config_dir(cfg: dict[str, Any]) -> Path | None:
    eval_config_path = cfg.get("_eval_config_path")
    if not eval_config_path:
        return None
    return Path(str(eval_config_path)).resolve().parent


def resolve_golden_set_path(cfg: dict[str, Any]) -> Path | None:
    golden_set_path = cfg.get("golden_set_path")
    if not golden_set_path:
        return None

    path = Path(str(golden_set_path))
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    config_dir = resolve_eval_config_dir(cfg)
    if config_dir is not None:
        candidates.append(config_dir / path)
    candidates.append(Path.cwd() / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_benchmark_prompts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    golden_set_path = resolve_golden_set_path(cfg)
    if golden_set_path is None:
        benchmark_prompts = cfg.get("benchmark_prompts") or DEFAULT_BENCHMARK_PROMPTS
        return [dict(prompt) for prompt in benchmark_prompts]

    if not golden_set_path.exists():
        raise FileNotFoundError(f"Golden set not found: {golden_set_path}")

    payload = read_json(str(golden_set_path))
    benchmark_prompts = payload.get("benchmark_prompts") or payload.get("golden_set")
    if not isinstance(benchmark_prompts, list) or not benchmark_prompts:
        raise ValueError(f"Golden set must contain a non-empty 'benchmark_prompts' array: {golden_set_path}")

    cfg["_resolved_golden_set_path"] = str(golden_set_path)
    cfg["_resolved_golden_set_name"] = str(payload.get("dataset_name", golden_set_path.stem))
    cfg["_resolved_golden_set_version"] = str(
        payload.get("dataset_version", cfg.get("dataset_version", "unknown"))
    )
    return [dict(prompt) for prompt in benchmark_prompts]


def golden_set_metadata(cfg: dict[str, Any]) -> dict[str, str]:
    golden_path = cfg.get("_resolved_golden_set_path")
    if not golden_path:
        return {}
    return {
        "golden_set_name": str(cfg.get("_resolved_golden_set_name", Path(str(golden_path)).stem)),
        "golden_set_version": str(cfg.get("_resolved_golden_set_version", cfg.get("dataset_version", "unknown"))),
        "golden_set_path": str(golden_path),
    }


def request_generation(
    endpoint: str,
    prompt: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
    ).encode("utf-8")
    request = Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen inference service returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach Qwen inference service at {endpoint}: {exc.reason}") from exc

    result = json.loads(body)
    if "text" not in result:
        raise RuntimeError(f"Qwen inference service response missing 'text': {result}")
    return result


def resolve_batch_inference_endpoint(endpoint: str) -> str:
    trimmed = endpoint.rstrip("/")
    if trimmed.endswith("/v1/generate"):
        return trimmed[: -len("/v1/generate")] + "/v1/generate_batch"
    return trimmed + "/generate_batch"


def request_generation_batch(
    endpoint: str,
    requests: list[dict[str, Any]],
    timeout_seconds: int,
) -> dict[str, Any]:
    batch_endpoint = resolve_batch_inference_endpoint(endpoint)
    payload = json.dumps({"requests": requests}).encode("utf-8")
    request = Request(batch_endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code in {404, 405}:
            raise BatchEndpointUnavailable(
                f"Batch inference endpoint is not available at {batch_endpoint}: HTTP {exc.code}"
            ) from exc
        raise RuntimeError(f"Qwen inference batch service returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach Qwen inference batch service at {batch_endpoint}: {exc.reason}") from exc

    result = json.loads(body)
    results = result.get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"Qwen inference batch service response missing 'results': {result}")
    if len(results) != len(requests):
        raise RuntimeError(
            "Qwen inference batch service returned an unexpected number of results: "
            f"expected {len(requests)}, got {len(results)}"
        )
    return result


def resolve_benchmark_request(
    benchmark: dict[str, Any],
    default_system_prompt: str,
    default_max_new_tokens: int,
    default_temperature: float,
) -> dict[str, Any]:
    return {
        "prompt": str(benchmark["prompt"]),
        "system_prompt": str(benchmark.get("system_prompt", default_system_prompt)),
        "max_new_tokens": int(benchmark.get("max_new_tokens", default_max_new_tokens)),
        "temperature": float(benchmark.get("temperature", default_temperature)),
    }


def score_response(response_text: str, benchmark: dict[str, Any]) -> dict[str, Any]:
    normalized_response = response_text.lower()
    expected_keywords = [keyword.strip().lower() for keyword in benchmark.get("expected_keywords", []) if keyword.strip()]
    matched_keywords = sorted({keyword for keyword in expected_keywords if keyword in normalized_response})
    keyword_score = 1.0 if not expected_keywords else len(matched_keywords) / len(expected_keywords)

    minimum_response_chars = max(int(benchmark.get("minimum_response_chars", 80)), 1)
    length_score = min(len(response_text.strip()) / minimum_response_chars, 1.0)
    prompt_score = round(keyword_score * 0.8 + length_score * 0.2, 3)

    return {
        "matched_keywords": matched_keywords,
        "keyword_score": round(keyword_score, 3),
        "length_score": round(length_score, 3),
        "score": prompt_score,
        "response_excerpt": response_text[:240],
    }


def summarize_benchmark_results(benchmark_results: list[dict[str, Any]]) -> dict[str, float]:
    public_score = average([result["score"] for result in benchmark_results if result["suite"] == "public"])
    domain_score = average([result["score"] for result in benchmark_results if result["suite"] == "domain"])
    if public_score == 0.0:
        public_score = average([result["score"] for result in benchmark_results])
    if domain_score == 0.0:
        domain_score = average([result["score"] for result in benchmark_results])

    intrinsic_perplexity = compute_intrinsic_perplexity_proxy(benchmark_results)
    overall = round((1 / intrinsic_perplexity) * 10 + public_score * 0.45 + domain_score * 0.45, 3)
    return {
        "intrinsic_perplexity": intrinsic_perplexity,
        "public_benchmark_score": public_score,
        "domain_benchmark_score": domain_score,
        "overall_score": overall,
    }


def compute_intrinsic_perplexity_proxy(benchmark_results: list[dict[str, Any]]) -> float:
    all_tokens: list[str] = []
    repeated_bigrams = 0
    total_bigrams = 0
    response_lengths: list[float] = []

    for result in benchmark_results:
        tokens = normalize_tokens(result["response"])
        all_tokens.extend(tokens)
        response_lengths.append(float(len(result["response"])))
        bigrams = list(zip(tokens, tokens[1:]))
        total_bigrams += len(bigrams)
        repeated_bigrams += max(len(bigrams) - len(set(bigrams)), 0)

    if not all_tokens:
        return 30.0

    lexical_diversity = len(set(all_tokens)) / len(all_tokens)
    repetition_ratio = repeated_bigrams / total_bigrams if total_bigrams else 0.0
    average_length = sum(response_lengths) / len(response_lengths)
    length_bonus = min(average_length / 240.0, 1.0)

    proxy = 24.0 - lexical_diversity * 12.0 + repetition_ratio * 10.0 - length_bonus * 2.0
    return round(min(max(proxy, 6.0), 30.0), 3)


def build_benchmark_results(
    cfg: dict[str, Any],
    inference_endpoint: str,
    timeout_seconds: int,
    default_system_prompt: str,
    default_max_new_tokens: int,
    default_temperature: float,
) -> tuple[list[dict[str, Any]], str]:
    benchmark_prompts = load_benchmark_prompts(cfg)
    benchmark_results: list[dict[str, Any]] = []
    judge_model_id = str(cfg.get("judge_model_id", "unknown"))
    inference_batch_size = max(int(cfg.get("inference_batch_size", 1)), 1)

    for start in range(0, len(benchmark_prompts), inference_batch_size):
        benchmark_chunk = benchmark_prompts[start : start + inference_batch_size]
        request_chunk = [
            resolve_benchmark_request(
                benchmark=benchmark,
                default_system_prompt=default_system_prompt,
                default_max_new_tokens=default_max_new_tokens,
                default_temperature=default_temperature,
            )
            for benchmark in benchmark_chunk
        ]

        if inference_batch_size > 1 and len(benchmark_chunk) > 1:
            batch_model_id = judge_model_id
            try:
                payload = request_generation_batch(
                    endpoint=inference_endpoint,
                    requests=request_chunk,
                    timeout_seconds=timeout_seconds,
                )
                batch_model_id = str(payload.get("model_id", judge_model_id))
                response_payloads = payload["results"]
            except BatchEndpointUnavailable:
                response_payloads = [
                    request_generation(
                        endpoint=inference_endpoint,
                        prompt=request_payload["prompt"],
                        system_prompt=request_payload["system_prompt"],
                        max_new_tokens=request_payload["max_new_tokens"],
                        temperature=request_payload["temperature"],
                        timeout_seconds=timeout_seconds,
                    )
                    for request_payload in request_chunk
                ]
            for benchmark, response_payload in zip(benchmark_chunk, response_payloads):
                response_text = str(response_payload["text"]).strip()
                judge_model_id = str(response_payload.get("model_id", batch_model_id))
                score = score_response(response_text, benchmark)
                benchmark_results.append(
                    {
                        "id": str(benchmark["id"]),
                        "suite": str(benchmark["suite"]),
                        "prompt": str(benchmark["prompt"]),
                        "response": response_text,
                        **score,
                    }
                )
            continue

        benchmark = benchmark_chunk[0]
        request_payload = request_chunk[0]
        payload = request_generation(
            endpoint=inference_endpoint,
            prompt=request_payload["prompt"],
            system_prompt=request_payload["system_prompt"],
            max_new_tokens=request_payload["max_new_tokens"],
            temperature=request_payload["temperature"],
            timeout_seconds=timeout_seconds,
        )
        response_text = str(payload["text"]).strip()
        judge_model_id = str(payload.get("model_id", judge_model_id))
        score = score_response(response_text, benchmark)
        benchmark_results.append(
            {
                "id": str(benchmark["id"]),
                "suite": str(benchmark["suite"]),
                "prompt": str(benchmark["prompt"]),
                "response": response_text,
                **score,
            }
        )

    return benchmark_results, judge_model_id


def resolve_candidate_inference_endpoint(cfg: dict[str, Any]) -> str:
    return str(
        cfg.get("candidate_inference_endpoint")
        or cfg.get("inference_endpoint")
        or os.getenv("QWEN_CANDIDATE_INFERENCE_URL")
        or os.getenv("QWEN_INFERENCE_URL")
        or DEFAULT_CANDIDATE_INFERENCE_ENDPOINT
    )


def resolve_baseline_inference_endpoint(cfg: dict[str, Any]) -> str | None:
    endpoint = (
        cfg.get("baseline_inference_endpoint")
        or os.getenv("QWEN_BASELINE_INFERENCE_URL")
        or os.getenv("QWEN_INFERENCE_URL")
    )
    if endpoint:
        return str(endpoint)
    return None


def evaluate_model(
    cfg: dict[str, Any],
    model_uri: str,
    inference_endpoint: str,
    timeout_seconds: int,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    benchmark_results, judge_model_id = build_benchmark_results(
        cfg=cfg,
        inference_endpoint=inference_endpoint,
        timeout_seconds=timeout_seconds,
        default_system_prompt=system_prompt,
        default_max_new_tokens=max_new_tokens,
        default_temperature=temperature,
    )
    return {
        "model_uri": model_uri,
        "served_model_id": judge_model_id,
        "inference_endpoint": inference_endpoint,
        "intrinsic_perplexity_is_proxy": True,
        **summarize_benchmark_results(benchmark_results),
        "benchmark_results": benchmark_results,
    }


def build_fallback_baseline(cfg: dict[str, Any], baseline_model_uri: str) -> dict[str, Any]:
    overall_score = float(cfg.get("baseline_overall_score", 1.0))
    return {
        "model_uri": baseline_model_uri,
        "served_model_id": str(cfg.get("baseline_model_id", baseline_model_uri)),
        "inference_endpoint": None,
        "intrinsic_perplexity_is_proxy": True,
        "intrinsic_perplexity": None,
        "public_benchmark_score": None,
        "domain_benchmark_score": None,
        "overall_score": overall_score,
        "benchmark_results": [],
        "evaluation_mode": "config_score",
    }


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def render_benchmark_table(benchmark_results: list[dict[str, Any]]) -> str:
    if not benchmark_results:
        return "No direct benchmark results were collected for this model.\n"

    lines = [
        "| Prompt ID | Suite | Score | Matched Keywords |",
        "| --- | --- | ---: | --- |",
    ]
    for result in benchmark_results:
        matched_keywords = ", ".join(result.get("matched_keywords", [])) or "none"
        lines.append(
            f"| {result['id']} | {result['suite']} | {result['score']:.3f} | {matched_keywords} |"
        )
    return "\n".join(lines) + "\n"


def build_model_card_markdown(
    role: str,
    cfg: dict[str, Any],
    model_summary: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    lines = [
        f"# {role.title()} Model Card",
        "",
        f"- Project: `{cfg.get('project', 'default')}`",
        f"- Run ID: `{cfg.get('run_id', 'local-eval')}`",
        f"- Model URI: `{model_summary['model_uri']}`",
        f"- Served Model ID: `{model_summary.get('served_model_id', 'unknown')}`",
        f"- Inference Endpoint: `{model_summary.get('inference_endpoint') or 'not directly evaluated'}`",
        f"- Overall Score: `{format_metric(model_summary.get('overall_score'))}`",
        f"- Public Benchmark Score: `{format_metric(model_summary.get('public_benchmark_score'))}`",
        f"- Domain Benchmark Score: `{format_metric(model_summary.get('domain_benchmark_score'))}`",
        f"- Intrinsic Perplexity Proxy: `{format_metric(model_summary.get('intrinsic_perplexity'))}`",
        "",
        "## Comparison",
        "",
        f"- Winner: `{comparison['winner']}`",
        f"- Delta vs Baseline: `{comparison['delta_vs_baseline']:.3f}`",
        f"- Result: `{comparison['result']}`",
        "",
        "## Benchmark Results",
        "",
        render_benchmark_table(model_summary.get("benchmark_results", [])).rstrip(),
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def build_single_model_card_markdown(cfg: dict[str, Any], model_summary: dict[str, Any]) -> str:
    lines = [
        "# Model Card",
        "",
        f"- Project: `{cfg.get('project', 'default')}`",
        f"- Run ID: `{cfg.get('run_id', 'local-eval')}`",
        f"- Model URI: `{model_summary['model_uri']}`",
        f"- Served Model ID: `{model_summary.get('served_model_id', 'unknown')}`",
        f"- Inference Endpoint: `{model_summary.get('inference_endpoint') or 'not directly evaluated'}`",
        f"- Overall Score: `{format_metric(model_summary.get('overall_score'))}`",
        f"- Public Benchmark Score: `{format_metric(model_summary.get('public_benchmark_score'))}`",
        f"- Domain Benchmark Score: `{format_metric(model_summary.get('domain_benchmark_score'))}`",
        f"- Intrinsic Perplexity Proxy: `{format_metric(model_summary.get('intrinsic_perplexity'))}`",
        "",
    ]
    if cfg.get("_resolved_golden_set_path"):
        lines.extend(
            [
                "## Golden Set",
                "",
                f"- Name: `{cfg.get('_resolved_golden_set_name', 'unknown')}`",
                f"- Version: `{cfg.get('_resolved_golden_set_version', cfg.get('dataset_version', 'unknown'))}`",
                f"- Path: `{cfg.get('_resolved_golden_set_path')}`",
                "",
            ]
        )
    lines.extend(
        [
        "## Benchmark Results",
        "",
        render_benchmark_table(model_summary.get("benchmark_results", [])).rstrip(),
        "",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_comparison_report_markdown(
    cfg: dict[str, Any],
    candidate_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    lines = [
        "# Evaluation Comparison Report",
        "",
        f"- Project: `{cfg.get('project', 'default')}`",
        f"- Run ID: `{cfg.get('run_id', 'local-eval')}`",
        f"- Candidate Model URI: `{candidate_summary['model_uri']}`",
        f"- Baseline Model URI: `{baseline_summary['model_uri']}`",
        f"- Winner: `{comparison['winner']}`",
        f"- Result: `{comparison['result']}`",
        "",
        "| Metric | Candidate | Baseline | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Overall Score | {format_metric(candidate_summary.get('overall_score'))} | {format_metric(baseline_summary.get('overall_score'))} | {comparison['delta_vs_baseline']:.3f} |",
        f"| Public Benchmark Score | {format_metric(candidate_summary.get('public_benchmark_score'))} | {format_metric(baseline_summary.get('public_benchmark_score'))} | {format_metric(comparison.get('delta_public_benchmark_score'))} |",
        f"| Domain Benchmark Score | {format_metric(candidate_summary.get('domain_benchmark_score'))} | {format_metric(baseline_summary.get('domain_benchmark_score'))} | {format_metric(comparison.get('delta_domain_benchmark_score'))} |",
        f"| Intrinsic Perplexity Proxy | {format_metric(candidate_summary.get('intrinsic_perplexity'))} | {format_metric(baseline_summary.get('intrinsic_perplexity'))} | {format_metric(comparison.get('delta_intrinsic_perplexity'))} |",
        "",
    ]
    return "\n".join(lines)


def write_model_card_artifacts(
    output_dir: Path,
    cfg: dict[str, Any],
    candidate_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, str]:
    artifacts_dir = output_dir / "model_cards"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    candidate_card_path = artifacts_dir / "candidate_model_card.md"
    baseline_card_path = artifacts_dir / "baseline_model_card.md"
    comparison_report_path = artifacts_dir / "comparison_report.md"

    candidate_card_path.write_text(
        build_model_card_markdown("candidate", cfg, candidate_summary, comparison),
        encoding="utf-8",
    )
    baseline_card_path.write_text(
        build_model_card_markdown("baseline", cfg, baseline_summary, comparison),
        encoding="utf-8",
    )
    comparison_report_path.write_text(
        build_comparison_report_markdown(cfg, candidate_summary, baseline_summary, comparison),
        encoding="utf-8",
    )

    return {
        "candidate_model_card": str(candidate_card_path),
        "baseline_model_card": str(baseline_card_path),
        "comparison_report": str(comparison_report_path),
    }


def write_single_model_card_artifact(output_dir: Path, cfg: dict[str, Any], model_summary: dict[str, Any]) -> dict[str, str]:
    artifacts_dir = output_dir / "model_cards"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_card_path = artifacts_dir / "model_card.md"
    model_card_path.write_text(build_single_model_card_markdown(cfg, model_summary), encoding="utf-8")
    return {"model_card": str(model_card_path)}


def build_comparison(candidate_summary: dict[str, Any], baseline_summary: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    delta = round(float(candidate_summary["overall_score"]) - float(baseline_summary["overall_score"]), 3)
    minimum_delta = float(cfg.get("minimum_delta_vs_baseline", 0.0))

    comparison = {
        "delta_vs_baseline": delta,
        "minimum_delta_vs_baseline": minimum_delta,
        "delta_public_benchmark_score": maybe_round(
            (
                candidate_summary.get("public_benchmark_score")
                - baseline_summary.get("public_benchmark_score")
            )
            if candidate_summary.get("public_benchmark_score") is not None
            and baseline_summary.get("public_benchmark_score") is not None
            else None
        ),
        "delta_domain_benchmark_score": maybe_round(
            (
                candidate_summary.get("domain_benchmark_score")
                - baseline_summary.get("domain_benchmark_score")
            )
            if candidate_summary.get("domain_benchmark_score") is not None
            and baseline_summary.get("domain_benchmark_score") is not None
            else None
        ),
        "delta_intrinsic_perplexity": maybe_round(
            (
                candidate_summary.get("intrinsic_perplexity")
                - baseline_summary.get("intrinsic_perplexity")
            )
            if candidate_summary.get("intrinsic_perplexity") is not None
            and baseline_summary.get("intrinsic_perplexity") is not None
            else None
        ),
    }
    comparison["winner"] = "candidate" if delta > 0 else "baseline" if delta < 0 else "tie"
    comparison["result"] = "pass" if delta >= minimum_delta else "fail"
    return comparison


def resolve_single_model_inference_endpoint(cfg: dict[str, Any]) -> str:
    return str(
        cfg.get("inference_endpoint")
        or cfg.get("candidate_inference_endpoint")
        or os.getenv("QWEN_INFERENCE_URL")
        or os.getenv("QWEN_CANDIDATE_INFERENCE_URL")
        or DEFAULT_INFERENCE_ENDPOINT
    )


def resolve_path_from_eval_config(cfg: dict[str, Any], path_value: str) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    config_dir = resolve_eval_config_dir(cfg)
    if config_dir is not None:
        candidates.append(config_dir / path)
    candidates.append(Path.cwd() / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_model_source_from_uri(model_uri: str) -> str:
    if model_uri.startswith("hf://"):
        return model_uri[len("hf://") :]
    if model_uri.startswith("file://"):
        return model_uri[len("file://") :]
    return model_uri


def resolve_local_torch_device() -> str:
    import torch

    requested = os.getenv("QWEN_DEVICE", "auto").strip().lower()
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_cpt_model_and_tokenizer(model_uri: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_source = resolve_model_source_from_uri(model_uri)
    resolved_device = resolve_local_torch_device()
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_source)
    model.to(resolved_device)
    model.eval()
    return tokenizer, model, model_source, resolved_device


def build_packed_cpt_examples(texts: list[str], tokenizer, block_size: int):
    import torch

    eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    pad_token_id = tokenizer.pad_token_id or eos_token_id
    token_buffer: list[int] = []

    for text in texts:
        token_ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
        token_buffer.extend(token_ids)
        if eos_token_id is not None:
            token_buffer.append(eos_token_id)

    if len(token_buffer) < 2:
        raise ValueError("Not enough tokens to build a CPT evaluation dataset")

    examples: list[dict[str, Any]] = []
    for start in range(0, len(token_buffer), block_size):
        chunk = token_buffer[start : start + block_size]
        if len(chunk) < 2:
            continue
        attention_mask = [1] * len(chunk)
        labels = list(chunk)
        if len(chunk) < block_size:
            padding = [pad_token_id] * (block_size - len(chunk))
            chunk = chunk + padding
            attention_mask = attention_mask + [0] * len(padding)
            labels = labels + [-100] * len(padding)
        examples.append(
            {
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        )

    if not examples:
        raise ValueError("No CPT evaluation examples were created after packing the corpus")

    return examples, len(token_buffer)


def iterate_batches(examples: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(examples), batch_size):
        yield examples[start : start + batch_size]


def evaluate_cpt_split(
    *,
    model,
    tokenizer,
    texts: list[str],
    block_size: int,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    import torch

    examples, token_count = build_packed_cpt_examples(texts, tokenizer, block_size)
    total_loss = 0.0
    total_active_tokens = 0

    with torch.no_grad():
        for batch in iterate_batches(examples, batch_size):
            input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
            attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(device)
            labels = torch.stack([item["labels"] for item in batch]).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            active_tokens = int((labels != -100).sum().item())
            total_loss += float(outputs.loss) * active_tokens
            total_active_tokens += active_tokens

    if total_active_tokens <= 0:
        raise ValueError("CPT evaluation produced no active tokens")

    loss = total_loss / total_active_tokens
    return {
        "loss": round(loss, 4),
        "perplexity": round(math.exp(min(loss, 20.0)), 4),
        "token_count": token_count,
        "packed_example_count": len(examples),
    }


def build_completion_seed_samples(
    *,
    texts: list[str],
    tokenizer,
    sample_count: int,
    prompt_tokens: int,
    reference_tokens: int,
) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for index, text in enumerate(texts):
        token_ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
        if len(token_ids) < prompt_tokens + max(reference_tokens, 8):
            continue
        prompt_ids = token_ids[:prompt_tokens]
        reference_ids = token_ids[prompt_tokens : prompt_tokens + reference_tokens]
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True).strip()
        reference_completion = tokenizer.decode(reference_ids, skip_special_tokens=True).strip()
        if not prompt or not reference_completion:
            continue
        samples.append(
            {
                "id": f"completion-{index + 1:03d}",
                "prompt": prompt,
                "reference_completion": reference_completion,
            }
        )
        if len(samples) >= sample_count:
            break
    return samples


def compute_reference_token_recall(generated_text: str, reference_text: str) -> tuple[float, list[str]]:
    reference_tokens = {token for token in normalize_tokens(reference_text)}
    generated_tokens = {token for token in normalize_tokens(generated_text)}
    if not reference_tokens:
        return 0.0, []
    matched_tokens = sorted(reference_tokens & generated_tokens)
    return round(len(matched_tokens) / len(reference_tokens), 3), matched_tokens


def generate_cpt_completion_samples(
    *,
    model,
    tokenizer,
    texts: list[str],
    device: str,
    sample_count: int,
    prompt_tokens: int,
    reference_tokens: int,
    max_new_tokens: int,
    temperature: float,
) -> list[dict[str, Any]]:
    import torch

    seed_samples = build_completion_seed_samples(
        texts=texts,
        tokenizer=tokenizer,
        sample_count=sample_count,
        prompt_tokens=prompt_tokens,
        reference_tokens=reference_tokens,
    )
    if not seed_samples:
        return []

    prompt_texts = [sample["prompt"] for sample in seed_samples]
    encoded = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        generated = model.generate(**encoded, **generation_kwargs)

    completion_samples: list[dict[str, Any]] = []
    for index, seed_sample in enumerate(seed_samples):
        prompt_length = int(prompt_lengths[index])
        generated_completion = tokenizer.decode(
            generated[index][prompt_length:],
            skip_special_tokens=True,
        ).strip()
        token_recall, matched_tokens = compute_reference_token_recall(
            generated_completion,
            seed_sample["reference_completion"],
        )
        completion_samples.append(
            {
                **seed_sample,
                "generated_completion": generated_completion,
                "reference_token_recall": token_recall,
                "matched_reference_tokens": matched_tokens,
            }
        )
    return completion_samples


def build_single_model_cpt_card_markdown(cfg: dict[str, Any], scorecard: dict[str, Any]) -> str:
    lines = [
        "# CPT Model Card",
        "",
        f"- Project: `{cfg.get('project', 'default')}`",
        f"- Run ID: `{cfg.get('run_id', 'local-eval')}`",
        f"- Model URI: `{scorecard['model_uri']}`",
        f"- Served Model ID: `{scorecard.get('served_model_id', 'unknown')}`",
        f"- Resolved Device: `{scorecard.get('resolved_device', 'unknown')}`",
        f"- Dataset Name: `{scorecard.get('dataset_name', 'unknown')}`",
        f"- Dataset Version: `{scorecard.get('dataset_version', 'unknown')}`",
        f"- Dataset Metadata Path: `{scorecard.get('dataset_metadata_path', 'unknown')}`",
        "",
        "## CPT Metrics",
        "",
        f"- Validation Loss: `{format_metric(scorecard.get('validation_loss'))}`",
        f"- Validation Perplexity: `{format_metric(scorecard.get('validation_perplexity'))}`",
        f"- Test Loss: `{format_metric(scorecard.get('test_loss'))}`",
        f"- Test Perplexity: `{format_metric(scorecard.get('test_perplexity'))}`",
        f"- Completion Reference Token Recall: `{format_metric(scorecard.get('completion_reference_token_recall'))}`",
        "",
        "## Completion Samples",
        "",
    ]

    completion_samples = scorecard.get("completion_samples", [])
    if not completion_samples:
        lines.append("No completion samples were generated.")
    else:
        for sample in completion_samples:
            lines.extend(
                [
                    f"### {sample['id']}",
                    f"- Prompt: {sample['prompt']}",
                    f"- Reference continuation: {sample['reference_completion']}",
                    f"- Generated continuation: {sample['generated_completion']}",
                    f"- Reference token recall: `{format_metric(sample.get('reference_token_recall'))}`",
                    "",
                ]
            )

    return "\n".join(lines).strip() + "\n"


def write_single_model_cpt_card_artifact(output_dir: Path, cfg: dict[str, Any], scorecard: dict[str, Any]) -> dict[str, str]:
    artifacts_dir = output_dir / "model_cards"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_card_path = artifacts_dir / "model_card.md"
    model_card_path.write_text(build_single_model_cpt_card_markdown(cfg, scorecard), encoding="utf-8")
    return {"model_card": str(model_card_path)}


def run_single_model_cpt_evaluation(
    cfg: dict[str, Any],
    model_uri: str,
    scorecard_path: Path,
) -> str:
    from components.trainer.run_pretraining import load_dataset_manifest

    dataset_metadata_path = cfg.get("dataset_metadata_path")
    if not dataset_metadata_path:
        raise ValueError("CPT evaluation requires 'dataset_metadata_path' in the eval config")

    resolved_dataset_metadata_path = resolve_path_from_eval_config(cfg, str(dataset_metadata_path))
    dataset_manifest = load_dataset_manifest(str(resolved_dataset_metadata_path))
    block_size = int(cfg.get("block_size", 128))
    eval_batch_size = max(int(cfg.get("eval_batch_size", 1)), 1)
    completion_split = str(cfg.get("completion_split", "test"))
    if completion_split == "eval":
        completion_split = "validation"
    if completion_split not in {"validation", "test", "train"}:
        raise ValueError("completion_split must be one of: train, validation, eval, test")

    completion_sample_count = max(int(cfg.get("completion_sample_count", 5)), 1)
    completion_prompt_tokens = max(int(cfg.get("completion_prompt_tokens", 48)), 8)
    completion_reference_tokens = max(int(cfg.get("completion_reference_tokens", 64)), 8)
    completion_max_new_tokens = max(int(cfg.get("max_new_tokens", completion_reference_tokens)), 1)
    temperature = float(cfg.get("temperature", 0.0))
    publish_model_cards = bool(cfg.get("publish_model_cards", True))

    tokenizer, model, model_source, resolved_device = load_cpt_model_and_tokenizer(model_uri)
    validation_metrics = evaluate_cpt_split(
        model=model,
        tokenizer=tokenizer,
        texts=dataset_manifest["resolved_splits"]["validation"]["texts"],
        block_size=block_size,
        batch_size=eval_batch_size,
        device=resolved_device,
    )
    test_metrics = evaluate_cpt_split(
        model=model,
        tokenizer=tokenizer,
        texts=dataset_manifest["resolved_splits"]["test"]["texts"],
        block_size=block_size,
        batch_size=eval_batch_size,
        device=resolved_device,
    )
    completion_samples = generate_cpt_completion_samples(
        model=model,
        tokenizer=tokenizer,
        texts=dataset_manifest["resolved_splits"][completion_split]["texts"],
        device=resolved_device,
        sample_count=completion_sample_count,
        prompt_tokens=completion_prompt_tokens,
        reference_tokens=completion_reference_tokens,
        max_new_tokens=completion_max_new_tokens,
        temperature=temperature,
    )
    completion_reference_token_recall = average(
        [float(sample["reference_token_recall"]) for sample in completion_samples]
    )

    scorecard = {
        "model_uri": model_uri,
        "evaluation_backend": "cpt_model",
        "evaluation_mode": "single_model",
        "served_model_id": model_source,
        "resolved_device": resolved_device,
        "dataset_name": dataset_manifest.get("dataset_name", "unknown"),
        "dataset_version": dataset_manifest.get("dataset_version", cfg.get("dataset_version", "unknown")),
        "dataset_metadata_path": str(resolved_dataset_metadata_path),
        "block_size": block_size,
        "eval_batch_size": eval_batch_size,
        "validation_loss": validation_metrics["loss"],
        "validation_perplexity": validation_metrics["perplexity"],
        "validation_token_count": validation_metrics["token_count"],
        "test_loss": test_metrics["loss"],
        "test_perplexity": test_metrics["perplexity"],
        "test_token_count": test_metrics["token_count"],
        "completion_split": completion_split,
        "completion_sample_count": len(completion_samples),
        "completion_prompt_tokens": completion_prompt_tokens,
        "completion_reference_tokens": completion_reference_tokens,
        "completion_reference_token_recall": completion_reference_token_recall,
        "completion_samples": completion_samples,
    }

    tags = {
        "dataset_version": str(scorecard["dataset_version"]),
        "config_version": str(cfg.get("config_version", "v1")),
        "pipeline_type": "evaluation",
        "run_id": str(cfg.get("run_id", "local-eval")),
        "evaluation_backend": "cpt_model",
        "evaluation_mode": "single_model",
    }
    experiment_name = f"llmops-evaluation-{cfg.get('project', 'default')}"

    with start_run(experiment_name, tags):
        mlflow.log_params(
            {
                "model_uri": model_uri,
                "project": cfg.get("project", "default"),
                "evaluation_backend": "cpt_model",
                "evaluation_mode": "single_model",
                "served_model_id": model_source,
                "resolved_device": resolved_device,
                "dataset_name": scorecard["dataset_name"],
                "dataset_version": scorecard["dataset_version"],
                "dataset_metadata_path": scorecard["dataset_metadata_path"],
                "block_size": block_size,
                "eval_batch_size": eval_batch_size,
                "completion_split": completion_split,
                "completion_sample_count": len(completion_samples),
            }
        )
        mlflow.log_metric("validation_loss", float(scorecard["validation_loss"]))
        mlflow.log_metric("validation_perplexity", float(scorecard["validation_perplexity"]))
        mlflow.log_metric("test_loss", float(scorecard["test_loss"]))
        mlflow.log_metric("test_perplexity", float(scorecard["test_perplexity"]))
        mlflow.log_metric(
            "completion_reference_token_recall",
            float(scorecard["completion_reference_token_recall"]),
        )

        model_card_artifacts: dict[str, str] = {}
        if publish_model_cards:
            model_card_artifacts = write_single_model_cpt_card_artifact(
                output_dir=scorecard_path.parent,
                cfg=cfg,
                scorecard=scorecard,
            )
            for artifact_path in model_card_artifacts.values():
                mlflow.log_artifact(artifact_path, artifact_path="model_cards")
        scorecard["model_card_artifacts"] = model_card_artifacts

        scorecard_path_str = str(scorecard_path)
        write_json(scorecard_path_str, scorecard)
        mlflow.log_artifact(scorecard_path_str, artifact_path="scorecards")

    return str(scorecard_path)


def run_qwen_service_evaluation(
    cfg: dict[str, Any],
    candidate_model_uri: str,
    baseline_model_uri: str,
    scorecard_path: Path,
) -> str:
    candidate_inference_endpoint = resolve_candidate_inference_endpoint(cfg)
    baseline_inference_endpoint = resolve_baseline_inference_endpoint(cfg)
    timeout_seconds = int(cfg.get("request_timeout_seconds", 120))
    system_prompt = str(cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
    max_new_tokens = int(cfg.get("max_new_tokens", 192))
    temperature = float(cfg.get("temperature", 0.2))
    inference_batch_size = max(int(cfg.get("inference_batch_size", 1)), 1)
    publish_model_cards = bool(cfg.get("publish_model_cards", True))

    candidate_summary = evaluate_model(
        cfg=cfg,
        model_uri=candidate_model_uri,
        inference_endpoint=candidate_inference_endpoint,
        timeout_seconds=timeout_seconds,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    if baseline_inference_endpoint:
        baseline_summary = evaluate_model(
            cfg=cfg,
            model_uri=baseline_model_uri,
            inference_endpoint=baseline_inference_endpoint,
            timeout_seconds=timeout_seconds,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        baseline_summary["evaluation_mode"] = "endpoint"
    else:
        baseline_summary = build_fallback_baseline(cfg, baseline_model_uri)

    comparison = build_comparison(candidate_summary, baseline_summary, cfg)

    tags = {
        "dataset_version": str(cfg.get("dataset_version", "unknown")),
        "config_version": str(cfg.get("config_version", "v1")),
        "pipeline_type": "evaluation",
        "run_id": str(cfg.get("run_id", "local-eval")),
        "evaluation_backend": "qwen_service",
        "comparison_mode": baseline_summary.get("evaluation_mode", "endpoint"),
        "winner": comparison["winner"],
    }
    experiment_name = f"llmops-evaluation-{cfg.get('project', 'default')}"

    scorecard = {
        "candidate_model_uri": candidate_model_uri,
        "baseline_model_uri": baseline_model_uri,
        "evaluation_backend": "qwen_service",
        "evaluation_mode": "compare",
        "comparison_mode": baseline_summary.get("evaluation_mode", "endpoint"),
        "inference_batch_size": inference_batch_size,
        "candidate_served_model_id": candidate_summary["served_model_id"],
        "baseline_served_model_id": baseline_summary.get("served_model_id"),
        "candidate_inference_endpoint": candidate_summary["inference_endpoint"],
        "baseline_inference_endpoint": baseline_summary.get("inference_endpoint"),
        "intrinsic_perplexity": candidate_summary["intrinsic_perplexity"],
        "intrinsic_perplexity_is_proxy": True,
        "public_benchmark_score": candidate_summary["public_benchmark_score"],
        "domain_benchmark_score": candidate_summary["domain_benchmark_score"],
        "overall_score": candidate_summary["overall_score"],
        "baseline_intrinsic_perplexity": baseline_summary.get("intrinsic_perplexity"),
        "baseline_public_benchmark_score": baseline_summary.get("public_benchmark_score"),
        "baseline_domain_benchmark_score": baseline_summary.get("domain_benchmark_score"),
        "baseline_overall_score": baseline_summary["overall_score"],
        "delta_vs_baseline": comparison["delta_vs_baseline"],
        "minimum_delta_vs_baseline": comparison["minimum_delta_vs_baseline"],
        "winner": comparison["winner"],
        "result": comparison["result"],
        "candidate": candidate_summary,
        "baseline": baseline_summary,
        "comparison": comparison,
    }
    scorecard.update(golden_set_metadata(cfg))

    with start_run(experiment_name, tags):
        log_params = {
            "candidate_model_uri": candidate_model_uri,
            "baseline_model_uri": baseline_model_uri,
            "project": cfg.get("project", "default"),
            "evaluation_backend": "qwen_service",
            "evaluation_mode": "compare",
            "candidate_served_model_id": candidate_summary["served_model_id"],
            "baseline_served_model_id": baseline_summary.get("served_model_id", "unknown"),
            "candidate_inference_endpoint": candidate_summary["inference_endpoint"],
            "baseline_inference_endpoint": baseline_summary.get("inference_endpoint", "config_score"),
            "comparison_mode": baseline_summary.get("evaluation_mode", "endpoint"),
            "inference_batch_size": inference_batch_size,
        }
        log_params.update(golden_set_metadata(cfg))
        mlflow.log_params(log_params)
        mlflow.log_metric("candidate_intrinsic_perplexity", candidate_summary["intrinsic_perplexity"])
        mlflow.log_metric("candidate_public_benchmark_score", candidate_summary["public_benchmark_score"])
        mlflow.log_metric("candidate_domain_benchmark_score", candidate_summary["domain_benchmark_score"])
        mlflow.log_metric("candidate_overall_score", candidate_summary["overall_score"])
        if baseline_summary.get("intrinsic_perplexity") is not None:
            mlflow.log_metric("baseline_intrinsic_perplexity", float(baseline_summary["intrinsic_perplexity"]))
        if baseline_summary.get("public_benchmark_score") is not None:
            mlflow.log_metric("baseline_public_benchmark_score", float(baseline_summary["public_benchmark_score"]))
        if baseline_summary.get("domain_benchmark_score") is not None:
            mlflow.log_metric("baseline_domain_benchmark_score", float(baseline_summary["domain_benchmark_score"]))
        mlflow.log_metric("baseline_overall_score", float(baseline_summary["overall_score"]))
        mlflow.log_metric("delta_vs_baseline", comparison["delta_vs_baseline"])
        mlflow.log_metric("candidate_benchmark_prompt_count", len(candidate_summary["benchmark_results"]))
        mlflow.log_metric("baseline_benchmark_prompt_count", len(baseline_summary.get("benchmark_results", [])))

        if comparison.get("delta_public_benchmark_score") is not None:
            mlflow.log_metric("delta_public_benchmark_score", float(comparison["delta_public_benchmark_score"]))
        if comparison.get("delta_domain_benchmark_score") is not None:
            mlflow.log_metric("delta_domain_benchmark_score", float(comparison["delta_domain_benchmark_score"]))
        if comparison.get("delta_intrinsic_perplexity") is not None:
            mlflow.log_metric("delta_intrinsic_perplexity", float(comparison["delta_intrinsic_perplexity"]))

        model_card_artifacts: dict[str, str] = {}
        if publish_model_cards:
            model_card_artifacts = write_model_card_artifacts(
                output_dir=scorecard_path.parent,
                cfg=cfg,
                candidate_summary=candidate_summary,
                baseline_summary=baseline_summary,
                comparison=comparison,
            )
            for artifact_path in model_card_artifacts.values():
                mlflow.log_artifact(artifact_path, artifact_path="model_cards")
        scorecard["model_card_artifacts"] = model_card_artifacts

        scorecard_path_str = str(scorecard_path)
        write_json(scorecard_path_str, scorecard)
        mlflow.log_artifact(scorecard_path_str, artifact_path="scorecards")

    return str(scorecard_path)


def run_single_model_qwen_service_evaluation(
    cfg: dict[str, Any],
    model_uri: str,
    scorecard_path: Path,
) -> str:
    inference_endpoint = resolve_single_model_inference_endpoint(cfg)
    timeout_seconds = int(cfg.get("request_timeout_seconds", 120))
    system_prompt = str(cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
    max_new_tokens = int(cfg.get("max_new_tokens", 192))
    temperature = float(cfg.get("temperature", 0.2))
    inference_batch_size = max(int(cfg.get("inference_batch_size", 1)), 1)
    publish_model_cards = bool(cfg.get("publish_model_cards", True))

    model_summary = evaluate_model(
        cfg=cfg,
        model_uri=model_uri,
        inference_endpoint=inference_endpoint,
        timeout_seconds=timeout_seconds,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    tags = {
        "dataset_version": str(cfg.get("dataset_version", "unknown")),
        "config_version": str(cfg.get("config_version", "v1")),
        "pipeline_type": "evaluation",
        "run_id": str(cfg.get("run_id", "local-eval")),
        "evaluation_backend": "qwen_service",
        "evaluation_mode": "single_model",
    }
    experiment_name = f"llmops-evaluation-{cfg.get('project', 'default')}"

    scorecard = {
        "model_uri": model_uri,
        "evaluation_backend": "qwen_service",
        "evaluation_mode": "single_model",
        "served_model_id": model_summary["served_model_id"],
        "inference_endpoint": model_summary["inference_endpoint"],
        "inference_batch_size": inference_batch_size,
        "intrinsic_perplexity": model_summary["intrinsic_perplexity"],
        "intrinsic_perplexity_is_proxy": True,
        "public_benchmark_score": model_summary["public_benchmark_score"],
        "domain_benchmark_score": model_summary["domain_benchmark_score"],
        "overall_score": model_summary["overall_score"],
        "benchmark_results": model_summary["benchmark_results"],
    }
    scorecard.update(golden_set_metadata(cfg))

    with start_run(experiment_name, tags):
        log_params = {
            "model_uri": model_uri,
            "project": cfg.get("project", "default"),
            "evaluation_backend": "qwen_service",
            "evaluation_mode": "single_model",
            "served_model_id": model_summary["served_model_id"],
            "inference_endpoint": model_summary["inference_endpoint"],
            "inference_batch_size": inference_batch_size,
        }
        log_params.update(golden_set_metadata(cfg))
        mlflow.log_params(log_params)
        mlflow.log_metric("intrinsic_perplexity", model_summary["intrinsic_perplexity"])
        mlflow.log_metric("public_benchmark_score", model_summary["public_benchmark_score"])
        mlflow.log_metric("domain_benchmark_score", model_summary["domain_benchmark_score"])
        mlflow.log_metric("overall_score", model_summary["overall_score"])
        mlflow.log_metric("benchmark_prompt_count", len(model_summary["benchmark_results"]))

        model_card_artifacts: dict[str, str] = {}
        if publish_model_cards:
            model_card_artifacts = write_single_model_card_artifact(
                output_dir=scorecard_path.parent,
                cfg=cfg,
                model_summary=model_summary,
            )
            for artifact_path in model_card_artifacts.values():
                mlflow.log_artifact(artifact_path, artifact_path="model_cards")
        scorecard["model_card_artifacts"] = model_card_artifacts

        scorecard_path_str = str(scorecard_path)
        write_json(scorecard_path_str, scorecard)
        mlflow.log_artifact(scorecard_path_str, artifact_path="scorecards")

    return str(scorecard_path)


def run(
    eval_config_path: str,
    candidate_model_uri: str,
    baseline_model_uri: str,
    output_dir: str | None = None,
    scorecard_output_path: str | None = None,
) -> str:
    cfg = read_json(eval_config_path)
    cfg["_eval_config_path"] = str(Path(eval_config_path).resolve())
    backend = str(cfg.get("evaluation_backend", "qwen_service"))

    if backend == "simulation":
        return simulate_evaluation.run(
            eval_config_path=eval_config_path,
            candidate_model_uri=candidate_model_uri,
            baseline_model_uri=baseline_model_uri,
            output_dir=output_dir,
            scorecard_output_path=scorecard_output_path,
        )
    if backend != "qwen_service":
        raise ValueError(f"Unsupported evaluation_backend: {backend}")

    resolved_output_dir, scorecard_path = resolve_output_paths(output_dir, scorecard_output_path)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return run_qwen_service_evaluation(cfg, candidate_model_uri, baseline_model_uri, scorecard_path)


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
