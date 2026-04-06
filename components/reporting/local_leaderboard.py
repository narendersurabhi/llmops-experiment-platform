from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class LeaderboardRow:
    run_name: str
    evaluation_backend: str
    dataset_name: str
    dataset_version: str
    model_uri: str
    model_source: str
    adaptation_method: str
    trainable_parameter_ratio: float | None
    lora_target_modules: str
    validation_perplexity: float | None
    test_perplexity: float | None
    completion_reference_token_recall: float | None
    overall_score: float | None
    public_benchmark_score: float | None
    domain_benchmark_score: float | None
    scorecard_path: str
    training_summary_path: str


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def maybe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def normalize_model_uri_to_local_path(model_uri: str) -> Path | None:
    if not model_uri:
        return None
    if model_uri.startswith("file://"):
        parsed = urlparse(model_uri)
        raw_path = parsed.path or model_uri[len("file://") :]
        return Path(unquote(raw_path)).resolve()
    return None


def discover_files(root: Path, filename: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob(f"*/{filename}"))


def build_training_index(train_root: Path) -> dict[Path, dict]:
    index: dict[Path, dict] = {}
    for summary_path in discover_files(train_root, "training_summary.json"):
        payload = read_json(summary_path)
        final_model_dir = payload.get("final_model_dir")
        if not final_model_dir:
            continue
        index[Path(str(final_model_dir)).resolve()] = {
            "summary_path": summary_path.resolve(),
            "adaptation_method": str(payload.get("adaptation_method", "full")),
            "trainable_parameter_ratio": maybe_float(payload.get("trainable_parameter_ratio")),
            "lora_target_modules": ",".join(payload.get("lora_config", {}).get("target_modules", [])),
        }
    return index


def build_leaderboard_rows(train_root: Path, eval_root: Path) -> list[LeaderboardRow]:
    training_index = build_training_index(train_root)
    rows: list[LeaderboardRow] = []

    for scorecard_path in discover_files(eval_root, "scorecard.json"):
        payload = read_json(scorecard_path)
        model_uri = str(payload.get("model_uri", ""))
        local_model_path = normalize_model_uri_to_local_path(model_uri)
        training_payload = training_index.get(local_model_path or Path("/nonexistent"), {})

        rows.append(
            LeaderboardRow(
                run_name=scorecard_path.parent.name,
                evaluation_backend=str(payload.get("evaluation_backend", "unknown")),
                dataset_name=str(payload.get("dataset_name", "unknown")),
                dataset_version=str(payload.get("dataset_version", "unknown")),
                model_uri=model_uri,
                model_source="local" if local_model_path else "remote",
                adaptation_method=str(training_payload.get("adaptation_method", "base")),
                trainable_parameter_ratio=training_payload.get("trainable_parameter_ratio"),
                lora_target_modules=str(training_payload.get("lora_target_modules", "")),
                validation_perplexity=maybe_float(payload.get("validation_perplexity")),
                test_perplexity=maybe_float(payload.get("test_perplexity")),
                completion_reference_token_recall=maybe_float(payload.get("completion_reference_token_recall")),
                overall_score=maybe_float(payload.get("overall_score")),
                public_benchmark_score=maybe_float(payload.get("public_benchmark_score")),
                domain_benchmark_score=maybe_float(payload.get("domain_benchmark_score")),
                scorecard_path=str(scorecard_path.resolve()),
                training_summary_path=str(training_payload.get("summary_path", "")),
            )
        )

    return rows


def format_number(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def sort_rows(rows: list[LeaderboardRow], backend: str) -> list[LeaderboardRow]:
    selected = [row for row in rows if row.evaluation_backend == backend]
    if backend == "cpt_model":
        return sorted(
            selected,
            key=lambda row: (
                row.dataset_version,
                float("inf") if row.test_perplexity is None else row.test_perplexity,
                row.run_name,
            ),
        )
    return sorted(
        selected,
        key=lambda row: (
            row.dataset_version,
            float("-inf") if row.overall_score is None else -row.overall_score,
            row.run_name,
        ),
    )


def render_markdown(rows: list[LeaderboardRow], repo_root: Path) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Local Experiment Leaderboard",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Total eval runs: `{len(rows)}`",
        "",
    ]

    cpt_rows = sort_rows(rows, "cpt_model")
    prompt_rows = [row for row in rows if row.evaluation_backend != "cpt_model"]
    prompt_rows = sort_rows(prompt_rows, prompt_rows[0].evaluation_backend) if prompt_rows else []

    lines.extend(
        [
            "## CPT Runs",
            "",
            "| Run | Dataset | Adaptation | LoRA Modules | Val PPL | Test PPL | Completion Recall | Training | Scorecard |",
            "|---|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in cpt_rows:
        lines.append(
            "| {run} | {dataset} | {adaptation} | {modules} | {val_ppl} | {test_ppl} | {recall} | {training} | {scorecard} |".format(
                run=row.run_name,
                dataset=row.dataset_version,
                adaptation=row.adaptation_method,
                modules=row.lora_target_modules or "-",
                val_ppl=format_number(row.validation_perplexity),
                test_ppl=format_number(row.test_perplexity),
                recall=format_number(row.completion_reference_token_recall, digits=3),
                training=(
                    maybe_relative(Path(row.training_summary_path), repo_root) if row.training_summary_path else "-"
                ),
                scorecard=maybe_relative(Path(row.scorecard_path), repo_root),
            )
        )
    if not cpt_rows:
        lines.append("| - | - | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Prompt/Golden Runs",
            "",
            "| Run | Dataset | Adaptation | Overall | Public | Domain | Scorecard |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in prompt_rows:
        lines.append(
            "| {run} | {dataset} | {adaptation} | {overall} | {public} | {domain} | {scorecard} |".format(
                run=row.run_name,
                dataset=row.dataset_version,
                adaptation=row.adaptation_method,
                overall=format_number(row.overall_score, digits=3),
                public=format_number(row.public_benchmark_score, digits=3),
                domain=format_number(row.domain_benchmark_score, digits=3),
                scorecard=maybe_relative(Path(row.scorecard_path), repo_root),
            )
        )
    if not prompt_rows:
        lines.append("| - | - | - | - | - | - | - |")

    return "\n".join(lines) + "\n"


def write_csv(rows: list[LeaderboardRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_name",
                "evaluation_backend",
                "dataset_name",
                "dataset_version",
                "model_uri",
                "model_source",
                "adaptation_method",
                "trainable_parameter_ratio",
                "lora_target_modules",
                "validation_perplexity",
                "test_perplexity",
                "completion_reference_token_recall",
                "overall_score",
                "public_benchmark_score",
                "domain_benchmark_score",
                "scorecard_path",
                "training_summary_path",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.run_name,
                    row.evaluation_backend,
                    row.dataset_name,
                    row.dataset_version,
                    row.model_uri,
                    row.model_source,
                    row.adaptation_method,
                    row.trainable_parameter_ratio,
                    row.lora_target_modules,
                    row.validation_perplexity,
                    row.test_perplexity,
                    row.completion_reference_token_recall,
                    row.overall_score,
                    row.public_benchmark_score,
                    row.domain_benchmark_score,
                    row.scorecard_path,
                    row.training_summary_path,
                ]
            )


def generate_leaderboard(
    *,
    repo_root: Path,
    train_root: Path,
    eval_root: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    rows = build_leaderboard_rows(train_root, eval_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "local_experiment_leaderboard.md"
    csv_path = output_dir / "local_experiment_leaderboard.csv"
    markdown_path.write_text(render_markdown(rows, repo_root), encoding="utf-8")
    write_csv(rows, csv_path)
    return markdown_path, csv_path
