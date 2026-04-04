from pathlib import Path

import pytest

pytest.importorskip("mlflow")
from components.trainer.simulate_training import run


def test_simulate_training(tmp_path: Path):
    cfg = tmp_path / "run.json"
    ds = tmp_path / "dataset.json"
    out = tmp_path / "out"

    cfg.write_text('{"project":"p","config_version":"v1","run_id":"r1","epochs":2}', encoding="utf-8")
    ds.write_text('{"dataset_name":"d","dataset_version":"dv1","num_samples":10}', encoding="utf-8")

    summary_path = run(str(cfg), str(ds), str(out))
    assert Path(summary_path).exists()
    assert (out / "checkpoints" / "checkpoint_epoch_1.ckpt").exists()
