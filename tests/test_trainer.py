import json
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
from components.trainer.run_pretraining import (
    build_training_arguments,
    load_dataset_manifest,
    patch_accelerator_class_unwrap_model_compat,
    resolve_trainer_device,
    run,
)


def test_continued_pretraining_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "run.json"
    ds = tmp_path / "metadata.json"
    out = tmp_path / "out"
    train_split = tmp_path / "train.jsonl"
    validation_split = tmp_path / "validation.jsonl"
    test_split = tmp_path / "test.jsonl"

    cfg.write_text(
        (
            '{"project":"p","config_version":"v2","run_id":"r1","training_backend":"continued_pretraining",'
            '"base_model_id":"sshleifer/tiny-gpt2","tokenizer_name":"sshleifer/tiny-gpt2","epochs":1}'
        ),
        encoding="utf-8",
    )
    train_split.write_text('{"text":"train sample one"}\n{"text":"train sample two"}\n', encoding="utf-8")
    validation_split.write_text('{"text":"validation sample one"}\n', encoding="utf-8")
    test_split.write_text('{"text":"test sample one"}\n', encoding="utf-8")
    ds.write_text(
        json.dumps(
            {
                "dataset_name": "d",
                "dataset_version": "dv2",
                "format": "jsonl",
                "text_field": "text",
                "splits": {
                    "train": str(train_split),
                    "validation": str(validation_split),
                    "test": str(test_split),
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_execute(cfg_payload, dataset_manifest, output_dir):
        checkpoint_dir = output_dir / "checkpoints" / "checkpoint-1"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return {
            "training_backend": "continued_pretraining",
            "base_model_id": cfg_payload["base_model_id"],
            "resolved_device": "mps",
            "best_checkpoint": str(checkpoint_dir),
            "train_metrics": {"train_loss": 1.23, "tokens_per_second_estimate": 456.0},
            "validation_metrics": {"validation_loss": 1.11, "validation_perplexity": 3.03},
            "test_metrics": {"test_loss": 1.22, "test_perplexity": 3.39},
        }

    monkeypatch.setattr("components.trainer.run_pretraining.execute_hf_continued_pretraining", fake_execute)

    summary_path = run(str(cfg), str(ds), str(out))
    payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    assert Path(summary_path).exists()
    assert payload["training_backend"] == "continued_pretraining"
    assert payload["resolved_device"] == "mps"
    assert payload["best_checkpoint"].endswith("checkpoint-1")


def test_build_training_arguments_uses_eval_strategy_when_available():
    captured = {}

    class DummyTrainingArguments:
        def __init__(self, *, eval_strategy=None, **kwargs):
            captured["eval_strategy"] = eval_strategy
            captured["kwargs"] = kwargs

    build_training_arguments({}, Path("/tmp/out"), 4, DummyTrainingArguments)
    assert captured["eval_strategy"] == "epoch"


def test_build_training_arguments_uses_evaluation_strategy_for_older_signatures():
    captured = {}

    class DummyTrainingArguments:
        def __init__(self, *, evaluation_strategy=None, **kwargs):
            captured["evaluation_strategy"] = evaluation_strategy
            captured["kwargs"] = kwargs

    build_training_arguments({}, Path("/tmp/out"), 4, DummyTrainingArguments)
    assert captured["evaluation_strategy"] == "epoch"


def test_patch_accelerator_class_unwrap_model_compat_adds_keep_torch_compile():
    captured = {}

    class DummyAccelerator:
        def unwrap_model(self, model, keep_fp32_wrapper=True):
            captured["model"] = model
            captured["keep_fp32_wrapper"] = keep_fp32_wrapper
            return ("ok", model)

    patch_accelerator_class_unwrap_model_compat(DummyAccelerator)
    result = DummyAccelerator().unwrap_model("model", keep_fp32_wrapper=False, keep_torch_compile=False)

    assert result == ("ok", "model")
    assert captured["model"] == "model"
    assert captured["keep_fp32_wrapper"] is False


def test_resolve_trainer_device_prefers_trainer_args_device():
    class DummyTrainerArgs:
        device = "mps"

    class DummyTrainer:
        args = DummyTrainerArgs()

    class DummyModel:
        def parameters(self):
            raise AssertionError("model.parameters should not be used when trainer.args.device is present")

    assert resolve_trainer_device(DummyTrainer(), DummyModel()) == "mps"


def test_load_dataset_manifest_accepts_eval_split_alias(tmp_path: Path):
    train_split = tmp_path / "train.jsonl"
    eval_split = tmp_path / "eval.jsonl"
    test_split = tmp_path / "test.jsonl"
    manifest_path = tmp_path / "metadata.json"

    train_split.write_text('{"text":"train sample"}\n', encoding="utf-8")
    eval_split.write_text('{"text":"eval sample"}\n', encoding="utf-8")
    test_split.write_text('{"text":"test sample"}\n', encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": "d",
                "dataset_version": "v1",
                "format": "jsonl",
                "text_field": "text",
                "splits": {
                    "train": "train.jsonl",
                    "eval": "eval.jsonl",
                    "test": "test.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_dataset_manifest(str(manifest_path))
    assert manifest["splits"]["validation"] == "eval.jsonl"
    assert manifest["resolved_splits"]["validation"]["document_count"] == 1
