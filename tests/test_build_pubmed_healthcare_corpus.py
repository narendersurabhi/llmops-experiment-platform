from __future__ import annotations

import json

import pytest

from scripts.data import build_pubmed_healthcare_corpus as builder


def test_resolve_split_sizes_uses_split_count_defaults_and_explicit_overrides():
    assert builder.resolve_split_sizes(split_count=500, train_count=2000, eval_count=None, test_count=None) == (
        2000,
        500,
        500,
    )


def test_resolve_split_sizes_requires_full_explicit_counts_without_split_count():
    with pytest.raises(ValueError):
        builder.resolve_split_sizes(split_count=None, train_count=2000, eval_count=500, test_count=None)


def test_build_dataset_writes_relative_metadata_paths(tmp_path, monkeypatch):
    articles = [
        {
            "pmid": f"pmid-{index}",
            "pmcid": f"PMC{index}",
            "title": f"Healthcare article {index}",
            "abstract": "A" * 700,
            "journal": "Test Journal",
            "publication_date": "2026-04-06",
        }
        for index in range(6)
    ]

    monkeypatch.setattr(builder, "fetch_esearch_ids", lambda **_: [article["pmid"] for article in articles])
    monkeypatch.setattr(builder, "fetch_pubmed_articles", lambda pmids, batch_size: articles[: len(pmids)])

    output_dir = tmp_path / "cpt" / "healthcare_expanded"
    metadata_path = tmp_path / "metadata.healthcare.expanded.json"
    builder.build_dataset(
        query="healthcare",
        from_date="2025/01/01",
        to_date="2026/04/06",
        train_count=2,
        eval_count=2,
        test_count=2,
        retmax=10,
        batch_size=2,
        output_dir=output_dir,
        metadata_path=metadata_path,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["dataset_name"] == "qwen2-5-0-5b-healthcare-pubmed-cpt-6"
    assert metadata["dataset_version"] == "pubmed-healthcare-20250101-20260406-2-2-2-v1"
    assert metadata["splits"]["train"] == "cpt/healthcare_expanded/train.jsonl"
    assert metadata["splits"]["eval"] == "cpt/healthcare_expanded/eval.jsonl"
    assert metadata["splits"]["test"] == "cpt/healthcare_expanded/test.jsonl"
    assert sum(1 for _ in output_dir.joinpath("train.jsonl").open(encoding="utf-8")) == 2
    assert sum(1 for _ in output_dir.joinpath("eval.jsonl").open(encoding="utf-8")) == 2
    assert sum(1 for _ in output_dir.joinpath("test.jsonl").open(encoding="utf-8")) == 2
