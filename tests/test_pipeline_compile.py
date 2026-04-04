from pathlib import Path

import pytest

kfp = pytest.importorskip("kfp")
from kfp import compiler

from pipelines.training.training_pipeline import training_pipeline
from pipelines.evaluation.evaluation_pipeline import evaluation_pipeline


def test_compile_training_pipeline(tmp_path: Path):
    out = tmp_path / "training.yaml"
    compiler.Compiler().compile(training_pipeline, str(out))
    assert out.exists()


def test_compile_eval_pipeline(tmp_path: Path):
    out = tmp_path / "evaluation.yaml"
    compiler.Compiler().compile(evaluation_pipeline, str(out))
    assert out.exists()
