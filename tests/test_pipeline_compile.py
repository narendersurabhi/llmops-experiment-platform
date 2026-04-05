from pathlib import Path

import pytest

kfp = pytest.importorskip("kfp")
from kfp import compiler

from pipelines.training.training_pipeline import BASE_IMAGE as TRAINING_BASE_IMAGE
from pipelines.training.training_pipeline import TRAINER_IMAGE, training_pipeline
from pipelines.evaluation.evaluation_pipeline import BASE_IMAGE as EVAL_BASE_IMAGE
from pipelines.evaluation.evaluation_pipeline import (
    EVALUATOR_IMAGE,
    QWEN_BASELINE_INFERENCE_URL,
    QWEN_CANDIDATE_INFERENCE_URL,
    QWEN_INFERENCE_URL,
    evaluation_pipeline,
)
from pipelines.evaluation.single_model_evaluation_pipeline import (
    BASE_IMAGE as SINGLE_EVAL_BASE_IMAGE,
    single_model_evaluation_pipeline,
)


def test_compile_training_pipeline(tmp_path: Path):
    out = tmp_path / "training.yaml"
    compiler.Compiler().compile(training_pipeline, str(out))
    assert out.exists()
    compiled = out.read_text(encoding="utf-8")
    assert TRAINING_BASE_IMAGE in compiled
    assert TRAINER_IMAGE in compiled
    assert "components.trainer.run_pretraining" in compiled


def test_compile_eval_pipeline(tmp_path: Path):
    out = tmp_path / "evaluation.yaml"
    compiler.Compiler().compile(evaluation_pipeline, str(out))
    assert out.exists()
    compiled = out.read_text(encoding="utf-8")
    assert EVAL_BASE_IMAGE in compiled
    assert EVALUATOR_IMAGE in compiled
    assert "components.evaluator.run_evaluation" in compiled
    assert QWEN_INFERENCE_URL in compiled
    assert QWEN_CANDIDATE_INFERENCE_URL in compiled
    assert QWEN_BASELINE_INFERENCE_URL in compiled


def test_compile_single_model_eval_pipeline(tmp_path: Path):
    out = tmp_path / "single-evaluation.yaml"
    compiler.Compiler().compile(single_model_evaluation_pipeline, str(out))
    assert out.exists()
    compiled = out.read_text(encoding="utf-8")
    assert SINGLE_EVAL_BASE_IMAGE in compiled
    assert EVALUATOR_IMAGE in compiled
    assert "components.evaluator.run_single_model_evaluation" in compiled
    assert QWEN_INFERENCE_URL in compiled
