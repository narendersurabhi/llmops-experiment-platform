from pathlib import Path

import pytest

pytest.importorskip("mlflow")
from components.evaluator.simulate_evaluation import run


def test_simulate_evaluation(tmp_path: Path):
    cfg = tmp_path / "eval.json"
    out = tmp_path / "out"
    cfg.write_text('{"project":"p","config_version":"v1","run_id":"r1","dataset_version":"d1"}', encoding="utf-8")

    scorecard = run(str(cfg), "models:/cand/1", "models:/base/1", str(out))
    assert Path(scorecard).exists()
