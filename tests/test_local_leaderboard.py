from __future__ import annotations

import json
from pathlib import Path

from components.reporting.local_leaderboard import build_leaderboard_rows, generate_leaderboard


def test_build_leaderboard_rows_links_local_scorecards_to_training_summaries(tmp_path: Path):
    train_root = tmp_path / "local-train"
    eval_root = tmp_path / "local-eval"
    train_run = train_root / "healthcare_lora"
    eval_run = eval_root / "healthcare_lora_cpt"
    model_dir = train_run / "model"
    train_run.mkdir(parents=True)
    eval_run.mkdir(parents=True)
    model_dir.mkdir()

    (train_run / "training_summary.json").write_text(
        json.dumps(
            {
                "adaptation_method": "lora",
                "final_model_dir": str(model_dir.resolve()),
                "trainable_parameter_ratio": 0.001093,
                "lora_config": {"target_modules": ["q_proj", "v_proj"]},
            }
        ),
        encoding="utf-8",
    )
    (eval_run / "scorecard.json").write_text(
        json.dumps(
            {
                "model_uri": f"file://{model_dir.resolve()}",
                "evaluation_backend": "cpt_model",
                "dataset_name": "healthcare",
                "dataset_version": "v1",
                "validation_perplexity": 10.5,
                "test_perplexity": 9.9,
                "completion_reference_token_recall": 0.33,
            }
        ),
        encoding="utf-8",
    )

    rows = build_leaderboard_rows(train_root, eval_root)

    assert len(rows) == 1
    row = rows[0]
    assert row.model_source == "local"
    assert row.adaptation_method == "lora"
    assert row.lora_target_modules == "q_proj,v_proj"
    assert row.test_perplexity == 9.9


def test_generate_leaderboard_writes_markdown_and_csv(tmp_path: Path):
    train_root = tmp_path / "local-train"
    eval_root = tmp_path / "local-eval"
    output_dir = tmp_path / "reports"
    train_run = train_root / "base"
    eval_run = eval_root / "base_cpt"
    model_dir = train_run / "model"
    train_run.mkdir(parents=True)
    eval_run.mkdir(parents=True)
    model_dir.mkdir()

    (train_run / "training_summary.json").write_text(
        json.dumps(
            {
                "adaptation_method": "full",
                "final_model_dir": str(model_dir.resolve()),
                "trainable_parameter_ratio": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (eval_run / "scorecard.json").write_text(
        json.dumps(
            {
                "model_uri": f"file://{model_dir.resolve()}",
                "evaluation_backend": "cpt_model",
                "dataset_name": "healthcare",
                "dataset_version": "v1",
                "validation_perplexity": 11.2,
                "test_perplexity": 10.8,
                "completion_reference_token_recall": 0.29,
            }
        ),
        encoding="utf-8",
    )

    markdown_path, csv_path = generate_leaderboard(
        repo_root=tmp_path,
        train_root=train_root,
        eval_root=eval_root,
        output_dir=output_dir,
    )

    assert markdown_path.exists()
    assert csv_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    csv_payload = csv_path.read_text(encoding="utf-8")
    assert "Local Experiment Leaderboard" in markdown
    assert "base_cpt" in markdown
    assert "run_name,evaluation_backend,dataset_name" in csv_payload
