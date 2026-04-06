import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
from components.evaluator.run_evaluation import BatchEndpointUnavailable, load_cpt_model_and_tokenizer, run
from components.evaluator.run_single_model_evaluation import run as run_single_model


def test_qwen_backed_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "eval.json"
    out = tmp_path / "out"
    cfg.write_text(
        (
            '{"project":"p","config_version":"v3","run_id":"r1","dataset_version":"d1",'
            '"evaluation_backend":"qwen_service",'
            '"candidate_inference_endpoint":"http://candidate",'
            '"baseline_inference_endpoint":"http://baseline",'
            '"publish_model_cards":true,'
            '"benchmark_prompts":['
            '{"id":"b1","suite":"public","prompt":"Explain Kubernetes","expected_keywords":["kubernetes","container"]},'
            '{"id":"b2","suite":"domain","prompt":"Explain MLflow","expected_keywords":["mlflow","artifacts"]}'
            ']}'
        ),
        encoding="utf-8",
    )

    def fake_generation(**kwargs):
        prompt = kwargs["prompt"].lower()
        endpoint = kwargs["endpoint"]
        if endpoint == "http://candidate":
            if "kubernetes" in prompt:
                return {
                    "model_id": "candidate-model",
                    "text": "Kubernetes orchestrates container workloads across a cluster and manages rolling updates safely.",
                }
            return {
                "model_id": "candidate-model",
                "text": "MLflow tracks metrics, artifacts, experiment metadata, and model lineage for machine learning runs.",
            }
        if "kubernetes" in prompt:
            return {"model_id": "baseline-model", "text": "Kubernetes runs apps in containers."}
        return {"model_id": "baseline-model", "text": "MLflow stores model run data."}

    monkeypatch.setattr("components.evaluator.run_evaluation.request_generation", fake_generation)
    scorecard = run(str(cfg), "models:/cand/1", "models:/base/1", str(out))
    assert Path(scorecard).exists()
    payload = Path(scorecard).read_text(encoding="utf-8")
    assert "qwen_service" in payload
    assert "candidate-model" in payload
    assert "baseline-model" in payload
    assert '"winner": "candidate"' in payload
    assert (out / "model_cards" / "candidate_model_card.md").exists()
    assert (out / "model_cards" / "baseline_model_card.md").exists()
    assert (out / "model_cards" / "comparison_report.md").exists()


def test_single_model_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "single-eval.json"
    out = tmp_path / "single-out"
    cfg.write_text(
        (
            '{"project":"p","config_version":"v1","run_id":"r1","dataset_version":"d1",'
            '"evaluation_mode":"single_model",'
            '"evaluation_backend":"qwen_service",'
            '"inference_endpoint":"http://single",'
            '"publish_model_cards":true,'
            '"benchmark_prompts":['
            '{"id":"b1","suite":"public","prompt":"Explain Kubernetes","expected_keywords":["kubernetes","container"]},'
            '{"id":"b2","suite":"domain","prompt":"Explain MLflow","expected_keywords":["mlflow","artifacts"]}'
            ']}'
        ),
        encoding="utf-8",
    )

    def fake_generation(**kwargs):
        prompt = kwargs["prompt"].lower()
        if "kubernetes" in prompt:
            return {
                "model_id": "single-model",
                "text": "Kubernetes orchestrates container workloads across a cluster and manages rolling updates safely.",
            }
        return {
            "model_id": "single-model",
            "text": "MLflow tracks metrics, artifacts, experiment metadata, and model lineage for machine learning runs.",
        }

    monkeypatch.setattr("components.evaluator.run_evaluation.request_generation", fake_generation)
    scorecard = run_single_model(str(cfg), "models:/single/1", str(out))
    assert Path(scorecard).exists()
    payload = Path(scorecard).read_text(encoding="utf-8")
    assert '"evaluation_mode": "single_model"' in payload
    assert '"model_uri": "models:/single/1"' in payload
    assert '"served_model_id": "single-model"' in payload
    assert '"benchmark_results"' in payload
    assert '"baseline_model_uri"' not in payload
    assert '"comparison"' not in payload
    assert (out / "model_cards" / "model_card.md").exists()
    assert not (out / "model_cards" / "comparison_report.md").exists()


def test_single_model_evaluation_uses_golden_set_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "single-eval.json"
    golden = tmp_path / "golden.json"
    out = tmp_path / "single-out"

    cfg.write_text(
        (
            '{"project":"p","config_version":"v2","run_id":"r2","dataset_version":"d2",'
            '"evaluation_mode":"single_model",'
            '"evaluation_backend":"qwen_service",'
            '"inference_endpoint":"http://single",'
            '"publish_model_cards":true,'
            '"golden_set_path":"golden.json"}'
        ),
        encoding="utf-8",
    )
    golden.write_text(
        (
            '{"dataset_name":"qwen-golden","dataset_version":"golden-v1","benchmark_prompts":['
            '{"id":"b1","suite":"public","prompt":"Explain Kubernetes","reference_answer":"Kubernetes orchestrates containers.","expected_keywords":["kubernetes","container"]},'
            '{"id":"b2","suite":"domain","prompt":"Explain MLflow","reference_answer":"MLflow tracks metrics and artifacts.","expected_keywords":["mlflow","artifacts"]}'
            ']}'
        ),
        encoding="utf-8",
    )

    def fake_generation(**kwargs):
        prompt = kwargs["prompt"].lower()
        if "kubernetes" in prompt:
            return {
                "model_id": "single-model",
                "text": "Kubernetes orchestrates container workloads across a cluster and manages rolling updates safely.",
            }
        return {
            "model_id": "single-model",
            "text": "MLflow tracks metrics, artifacts, experiment metadata, and model lineage for machine learning runs.",
        }

    monkeypatch.setattr("components.evaluator.run_evaluation.request_generation", fake_generation)
    scorecard = run_single_model(str(cfg), "models:/single/2", str(out))
    payload = Path(scorecard).read_text(encoding="utf-8")
    model_card = (out / "model_cards" / "model_card.md").read_text(encoding="utf-8")

    assert '"golden_set_name": "qwen-golden"' in payload
    assert '"golden_set_version": "golden-v1"' in payload
    assert '"golden_set_path":' in payload
    assert "## Golden Set" in model_card
    assert "qwen-golden" in model_card


def test_single_model_evaluation_uses_batched_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "single-eval-batch.json"
    out = tmp_path / "single-out-batch"
    cfg.write_text(
        (
            '{"project":"p","config_version":"v3","run_id":"r-batch","dataset_version":"d3",'
            '"evaluation_mode":"single_model",'
            '"evaluation_backend":"qwen_service",'
            '"inference_endpoint":"http://single",'
            '"inference_batch_size":2,'
            '"publish_model_cards":true,'
            '"benchmark_prompts":['
            '{"id":"b1","suite":"public","prompt":"Explain Kubernetes","expected_keywords":["kubernetes","container"]},'
            '{"id":"b2","suite":"domain","prompt":"Explain MLflow","expected_keywords":["mlflow","artifacts"]},'
            '{"id":"b3","suite":"public","prompt":"Explain MinIO","expected_keywords":["minio","storage"]},'
            '{"id":"b4","suite":"domain","prompt":"Explain checkpoints","expected_keywords":["checkpoint","model"]}'
            ']}'
        ),
        encoding="utf-8",
    )

    batch_sizes: list[int] = []

    def fake_generation_batch(*, endpoint, requests, timeout_seconds):
        assert endpoint == "http://single"
        assert timeout_seconds == 120
        batch_sizes.append(len(requests))
        results = []
        for item in requests:
            prompt = item["prompt"].lower()
            if "kubernetes" in prompt:
                text = "Kubernetes orchestrates container workloads across a cluster."
            elif "mlflow" in prompt:
                text = "MLflow tracks metrics and artifacts for machine learning experiments."
            elif "minio" in prompt:
                text = "MinIO provides S3-compatible object storage for artifacts."
            else:
                text = "A checkpoint stores intermediate model weights during training."
            results.append({"text": text})
        return {"model_id": "batched-model", "results": results}

    def fail_single_generation(**kwargs):
        raise AssertionError(f"Sequential inference should not be used: {kwargs}")

    monkeypatch.setattr("components.evaluator.run_evaluation.request_generation_batch", fake_generation_batch)
    monkeypatch.setattr("components.evaluator.run_evaluation.request_generation", fail_single_generation)

    scorecard = run_single_model(str(cfg), "models:/single/batch", str(out))
    payload = Path(scorecard).read_text(encoding="utf-8")

    assert batch_sizes == [2, 2]
    assert '"inference_batch_size": 2' in payload
    assert '"served_model_id": "batched-model"' in payload


def test_single_model_evaluation_falls_back_when_batch_endpoint_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = tmp_path / "single-eval-batch-fallback.json"
    out = tmp_path / "single-out-batch-fallback"
    cfg.write_text(
        (
            '{"project":"p","config_version":"v4","run_id":"r-batch-fallback","dataset_version":"d4",'
            '"evaluation_mode":"single_model",'
            '"evaluation_backend":"qwen_service",'
            '"inference_endpoint":"http://single",'
            '"inference_batch_size":2,'
            '"publish_model_cards":true,'
            '"benchmark_prompts":['
            '{"id":"b1","suite":"public","prompt":"Explain Kubernetes","expected_keywords":["kubernetes","container"]},'
            '{"id":"b2","suite":"domain","prompt":"Explain MLflow","expected_keywords":["mlflow","artifacts"]}'
            ']}'
        ),
        encoding="utf-8",
    )

    single_calls: list[str] = []

    def fail_batch_generation(**kwargs):
        raise BatchEndpointUnavailable("no batch endpoint")

    def fake_single_generation(**kwargs):
        single_calls.append(kwargs["prompt"])
        prompt = kwargs["prompt"].lower()
        if "kubernetes" in prompt:
            return {"model_id": "fallback-model", "text": "Kubernetes orchestrates container workloads."}
        return {"model_id": "fallback-model", "text": "MLflow tracks metrics and artifacts."}

    monkeypatch.setattr("components.evaluator.run_evaluation.request_generation_batch", fail_batch_generation)
    monkeypatch.setattr("components.evaluator.run_evaluation.request_generation", fake_single_generation)

    scorecard = run_single_model(str(cfg), "models:/single/fallback", str(out))
    payload = Path(scorecard).read_text(encoding="utf-8")

    assert len(single_calls) == 2
    assert '"served_model_id": "fallback-model"' in payload


def test_load_cpt_model_and_tokenizer_supports_local_lora_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    adapter_dir = tmp_path / "adapter-model"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    calls: list[tuple[str, str]] = []

    class DummyTokenizer:
        pad_token_id = None
        eos_token = "</s>"
        padding_side = "right"

    class DummyModel:
        def __init__(self):
            self.device = None
            self.eval_called = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    def fake_tokenizer_from_pretrained(source: str):
        calls.append(("tokenizer", source))
        return DummyTokenizer()

    def fake_model_from_pretrained(source: str):
        calls.append(("base_model", source))
        return DummyModel()

    class DummyPeftConfig:
        base_model_name_or_path = "Qwen/Qwen2.5-0.5B"

        @staticmethod
        def from_pretrained(source: str):
            calls.append(("peft_config", source))
            return DummyPeftConfig()

    class DummyPeftModel:
        @staticmethod
        def from_pretrained(model, source: str):
            calls.append(("peft_model", source))
            return model

    transformers_stub = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=fake_tokenizer_from_pretrained),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=fake_model_from_pretrained),
    )
    peft_stub = types.SimpleNamespace(PeftConfig=DummyPeftConfig, PeftModel=DummyPeftModel)

    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    monkeypatch.setitem(sys.modules, "peft", peft_stub)
    monkeypatch.setattr("components.evaluator.run_evaluation.resolve_local_torch_device", lambda: "cpu")

    tokenizer, model, model_source, resolved_device = load_cpt_model_and_tokenizer(f"file://{adapter_dir}")

    assert tokenizer.pad_token_id is None
    assert tokenizer.padding_side == "left"
    assert model_source == str(adapter_dir)
    assert resolved_device == "cpu"
    assert model.device == "cpu"
    assert model.eval_called is True
    assert calls == [
        ("tokenizer", str(adapter_dir)),
        ("peft_config", str(adapter_dir)),
        ("base_model", "Qwen/Qwen2.5-0.5B"),
        ("peft_model", str(adapter_dir)),
    ]


def test_single_model_cpt_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "single-cpt-eval.json"
    dataset_manifest = tmp_path / "dataset.json"
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    test_path = tmp_path / "test.jsonl"
    out = tmp_path / "single-cpt-out"

    train_path.write_text(
        '\n'.join(
            [
                '{"id":"train-1","text":"Transformers learn from large token sequences to predict the next token."}',
                '{"id":"train-2","text":"Continued pretraining adapts a base model to a new domain corpus."}',
            ]
        )
        + '\n',
        encoding="utf-8",
    )
    eval_path.write_text(
        '{"id":"eval-1","text":"Evaluation loss on held-out text should decrease when adaptation helps."}\n',
        encoding="utf-8",
    )
    test_path.write_text(
        '{"id":"test-1","text":"Test perplexity is a primary metric for continued pretraining quality."}\n',
        encoding="utf-8",
    )
    dataset_manifest.write_text(
        (
            '{"dataset_name":"cpt-public","dataset_version":"cpt-v1","format":"jsonl","text_field":"text",'
            f'"splits":{{"train":"{train_path.name}","eval":"{eval_path.name}","test":"{test_path.name}"}}}}'
        ),
        encoding="utf-8",
    )
    cfg.write_text(
        (
            '{"project":"p","config_version":"v-cpt","run_id":"r-cpt","dataset_version":"cpt-v1",'
            '"evaluation_mode":"single_model",'
            '"evaluation_backend":"cpt_model",'
            f'"dataset_metadata_path":"{dataset_manifest.name}",'
            '"publish_model_cards":true,'
            '"block_size":64,'
            '"eval_batch_size":1,'
            '"completion_split":"test",'
            '"completion_sample_count":2,'
            '"completion_prompt_tokens":8,'
            '"completion_reference_tokens":12,'
            '"max_new_tokens":12,'
            '"temperature":0.0}'
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "components.evaluator.run_evaluation.load_cpt_model_and_tokenizer",
        lambda model_uri: ("tokenizer", "model", "Qwen/Qwen2.5-0.5B", "cpu"),
    )
    split_results = [
        {"loss": 2.3456, "perplexity": 10.4401, "token_count": 123, "packed_example_count": 2},
        {"loss": 2.1234, "perplexity": 8.3602, "token_count": 111, "packed_example_count": 2},
    ]

    def fake_evaluate_cpt_split(**kwargs):
        return split_results.pop(0)

    monkeypatch.setattr("components.evaluator.run_evaluation.evaluate_cpt_split", fake_evaluate_cpt_split)
    monkeypatch.setattr(
        "components.evaluator.run_evaluation.generate_cpt_completion_samples",
        lambda **kwargs: [
            {
                "id": "completion-001",
                "prompt": "Test perplexity is",
                "reference_completion": "a primary metric for held-out text quality.",
                "generated_completion": "a primary signal for domain adaptation quality.",
                "reference_token_recall": 0.625,
                "matched_reference_tokens": ["a", "primary", "for", "quality"],
            }
        ],
    )

    scorecard = run_single_model(str(cfg), "file:///tmp/qwen-model", str(out))
    payload = Path(scorecard).read_text(encoding="utf-8")
    model_card = (out / "model_cards" / "model_card.md").read_text(encoding="utf-8")

    assert '"evaluation_backend": "cpt_model"' in payload
    assert '"validation_perplexity": 10.4401' in payload
    assert '"test_perplexity": 8.3602' in payload
    assert '"completion_reference_token_recall": 0.625' in payload
    assert "## CPT Metrics" in model_card
    assert "Completion Samples" in model_card
