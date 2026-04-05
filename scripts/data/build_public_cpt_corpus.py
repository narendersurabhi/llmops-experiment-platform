#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset" / "cpt" / "qwen2_5_0_5b_public"
METADATA_PATH = REPO_ROOT / "dataset" / "metadata.qwen2_5_0_5b.public.json"
GOLDEN_PATH = REPO_ROOT / "dataset" / "golden" / "qwen2_5_0_5b_public.golden.json"
EVAL_CONFIG_PATH = REPO_ROOT / "configs" / "evaluation" / "qwen2_5_0_5b.public.single_model.sample.json"

FINEWEB_SOURCE = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_CATEGORIES = ["cs.CL", "cs.LG", "cs.AI", "stat.ML"]
TRAIN_COUNT = 2400
EVAL_COUNT = 300
TEST_COUNT = 300
GOLDEN_COUNT = 300
FINEWEB_DOCS_REQUIRED = 2700
ARXIV_DOCS_REQUIRED = 600
MIN_TEXT_CHARS = 600
MAX_TEXT_CHARS = 4000
ARXIV_WAIT_SECONDS = 3.5
TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{2,}")
STOPWORDS = {
    "about", "after", "again", "algorithm", "analysis", "approach", "between", "dataset", "experiments",
    "from", "have", "into", "language", "large", "learning", "model", "models", "paper", "results",
    "show", "that", "their", "these", "this", "using", "with", "what", "when", "where", "which",
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    ensure_parent(path)
    lines = [json.dumps(record, ensure_ascii=True) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clip_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    last_period = clipped.rfind(". ")
    if last_period > 200:
        clipped = clipped[: last_period + 1]
    return clipped.strip()


def build_fineweb_records() -> list[dict]:
    from datasets import load_dataset

    records: list[dict] = []
    seen_texts: set[str] = set()
    dataset = load_dataset(FINEWEB_SOURCE, name=FINEWEB_CONFIG, split="train", streaming=True)

    for sample in dataset:
        text = clip_text(str(sample.get("text", "")))
        if len(text) < MIN_TEXT_CHARS:
            continue
        if text in seen_texts:
            continue
        seen_texts.add(text)
        records.append(
            {
                "text": text,
                "source": "fineweb-edu",
                "language": "en",
                "domain": "general",
            }
        )
        if len(records) >= FINEWEB_DOCS_REQUIRED:
            break

    if len(records) < FINEWEB_DOCS_REQUIRED:
        raise RuntimeError(f"Only collected {len(records)} FineWeb-Edu documents; expected {FINEWEB_DOCS_REQUIRED}")
    return records


def fetch_arxiv_batch(start: int, batch_size: int) -> list[dict]:
    query = urlencode(
        {
            "search_query": " OR ".join(f"cat:{category}" for category in ARXIV_CATEGORIES),
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    with urlopen(f"{ARXIV_API_URL}?{query}", timeout=60) as response:
        xml_payload = response.read().decode("utf-8")

    root = ET.fromstring(xml_payload)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    entries: list[dict] = []
    for entry in root.findall("atom:entry", namespace):
        entry_id = normalize_whitespace(entry.findtext("atom:id", default="", namespaces=namespace))
        title = normalize_whitespace(entry.findtext("atom:title", default="", namespaces=namespace))
        summary = normalize_whitespace(entry.findtext("atom:summary", default="", namespaces=namespace))
        if not entry_id or not title or not summary:
            continue
        categories = [elem.attrib.get("term", "") for elem in entry.findall("atom:category", namespace)]
        entries.append(
            {
                "id": entry_id,
                "title": title,
                "summary": summary,
                "categories": categories,
            }
        )
    return entries


def build_arxiv_records() -> list[dict]:
    records: list[dict] = []
    seen_ids: set[str] = set()
    start = 0
    batch_size = 100
    while len(records) < ARXIV_DOCS_REQUIRED:
        batch = fetch_arxiv_batch(start, batch_size)
        if not batch:
            break
        for item in batch:
            if item["id"] in seen_ids:
                continue
            seen_ids.add(item["id"])
            text = clip_text(f"{item['title']}. {item['summary']}")
            if len(text) < MIN_TEXT_CHARS:
                continue
            records.append(
                {
                    "text": text,
                    "source": "arxiv",
                    "language": "en",
                    "domain": "llm-cpt",
                    "paper_id": item["id"],
                    "title": item["title"],
                    "summary": item["summary"],
                    "categories": item["categories"],
                }
            )
            if len(records) >= ARXIV_DOCS_REQUIRED:
                break
        start += batch_size
        time.sleep(ARXIV_WAIT_SECONDS)

    if len(records) < ARXIV_DOCS_REQUIRED:
        raise RuntimeError(f"Only collected {len(records)} arXiv records; expected {ARXIV_DOCS_REQUIRED}")
    return records


def build_split_records(fineweb_records: list[dict], arxiv_records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    train_records: list[dict] = []
    eval_records: list[dict] = []
    test_records: list[dict] = []

    train_pool = fineweb_records[:2100] + arxiv_records[:300]
    eval_pool = fineweb_records[2100:2400]
    test_pool = fineweb_records[2400:2700]

    for idx, record in enumerate(train_pool):
        train_records.append(
            {
                "id": f"train-{idx + 1:04d}",
                "text": record["text"],
                "source": record["source"],
                "language": record["language"],
                "domain": record["domain"],
                "split": "train",
            }
        )
    for idx, record in enumerate(eval_pool):
        eval_records.append(
            {
                "id": f"eval-{idx + 1:04d}",
                "text": record["text"],
                "source": record["source"],
                "language": record["language"],
                "domain": record["domain"],
                "split": "eval",
            }
        )
    for idx, record in enumerate(test_pool):
        test_records.append(
            {
                "id": f"test-{idx + 1:04d}",
                "text": record["text"],
                "source": record["source"],
                "language": record["language"],
                "domain": record["domain"],
                "split": "test",
            }
        )

    if len(train_records) != TRAIN_COUNT or len(eval_records) != EVAL_COUNT or len(test_records) != TEST_COUNT:
        raise RuntimeError("Public-source split construction produced incorrect counts")
    return train_records, eval_records, test_records


def extract_keywords(title: str, summary: str, max_keywords: int = 5) -> list[str]:
    tokens = TOKEN_RE.findall(f"{title.lower()} {summary.lower()}")
    seen: list[str] = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token not in seen:
            seen.append(token)
        if len(seen) >= max_keywords:
            break
    return seen or ["language", "model", "training"]


def build_golden_prompt(item: dict, index: int) -> dict:
    keywords = extract_keywords(item["title"], item["summary"])
    prompt_styles = [
        f"Summarize the main idea of the paper titled '{item['title']}' in three sentences.",
        f"Explain the core contribution of the paper titled '{item['title']}'.",
        f"What problem is addressed in the paper titled '{item['title']}', and why does it matter?",
        f"Describe the key takeaway from the paper titled '{item['title']}' for an LLM engineer.",
    ]
    prompt = prompt_styles[index % len(prompt_styles)]
    return {
        "id": f"golden-arxiv-{index + 1:03d}",
        "suite": "domain",
        "prompt": prompt,
        "reference_answer": item["summary"],
        "expected_keywords": keywords,
        "minimum_response_chars": 180,
        "source_title": item["title"],
        "source_paper_id": item["paper_id"],
    }


def build_golden_set(arxiv_records: list[dict]) -> dict:
    held_out = arxiv_records[300:600]
    prompts = [build_golden_prompt(item, idx) for idx, item in enumerate(held_out[:GOLDEN_COUNT])]
    if len(prompts) != GOLDEN_COUNT:
        raise RuntimeError("Golden set construction produced incorrect prompt count")
    return {
        "dataset_name": "qwen2-5-0-5b-public-cpt-golden-300",
        "dataset_version": "public-golden-v1",
        "description": "Held-out golden set derived from arXiv abstracts in cs.CL, cs.LG, cs.AI, and stat.ML.",
        "benchmark_prompts": prompts,
    }


def build_metadata() -> dict:
    return {
        "dataset_name": "qwen2-5-0-5b-public-cpt-3000",
        "dataset_version": "public-cpt-3000-v1",
        "format": "jsonl",
        "text_field": "text",
        "source": "fineweb-edu+arxiv",
        "language": "en",
        "splits": {
            "train": "cpt/qwen2_5_0_5b_public/train.jsonl",
            "eval": "cpt/qwen2_5_0_5b_public/eval.jsonl",
            "test": "cpt/qwen2_5_0_5b_public/test.jsonl",
        },
    }


def build_eval_config() -> dict:
    return {
        "project": "llmops-local",
        "config_version": "eval-single-public-golden-v1",
        "run_id": "eval-qwen2-5-0-5b-public-golden-300",
        "dataset_version": "public-golden-v1",
        "evaluation_mode": "single_model",
        "evaluation_backend": "qwen_service",
        "inference_endpoint": "http://qwen-inference.llmops.svc.cluster.local:8000/v1/generate",
        "judge_model_id": "single-model-run",
        "publish_model_cards": True,
        "golden_set_path": "dataset/golden/qwen2_5_0_5b_public.golden.json",
        "inference_batch_size": 4,
        "max_new_tokens": 192,
        "temperature": 0.0,
        "request_timeout_seconds": 120,
    }


def main() -> None:
    fineweb_records = build_fineweb_records()
    arxiv_records = build_arxiv_records()
    train_records, eval_records, test_records = build_split_records(fineweb_records, arxiv_records)

    write_jsonl(DATASET_DIR / "train.jsonl", train_records)
    write_jsonl(DATASET_DIR / "eval.jsonl", eval_records)
    write_jsonl(DATASET_DIR / "test.jsonl", test_records)
    write_json(METADATA_PATH, build_metadata())
    write_json(GOLDEN_PATH, build_golden_set(arxiv_records))
    write_json(EVAL_CONFIG_PATH, build_eval_config())

    print(f"Wrote train/eval/test to {DATASET_DIR}")
    print(f"Wrote dataset manifest to {METADATA_PATH}")
    print(f"Wrote golden set to {GOLDEN_PATH}")
    print(f"Wrote eval config to {EVAL_CONFIG_PATH}")


if __name__ == "__main__":
    main()
