#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonschema


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(base_path: Path, candidate: str) -> Path:
    resolved = Path(candidate)
    if resolved.is_absolute():
        return resolved
    return (base_path.parent / resolved).resolve()


def load_jsonl_count(path: Path, text_field: str) -> tuple[int, int]:
    document_count = 0
    non_empty_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        document_count += 1
        record = json.loads(line)
        if str(record.get(text_field, "")).strip():
            non_empty_count += 1
    return document_count, non_empty_count


def normalize_splits(splits: dict) -> dict:
    normalized = dict(splits)
    if "validation" not in normalized and "eval" in normalized:
        normalized["validation"] = normalized["eval"]
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local CPT dataset metadata and golden-set files.")
    parser.add_argument("--dataset-metadata-path", required=True)
    parser.add_argument("--golden-set-path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dataset_metadata_path = Path(args.dataset_metadata_path).resolve()
    dataset_schema = read_json(repo_root / "schemas" / "dataset.schema.json")
    golden_schema = read_json(repo_root / "schemas" / "golden.schema.json")

    dataset_manifest = read_json(dataset_metadata_path)
    jsonschema.validate(dataset_manifest, dataset_schema)

    splits = normalize_splits(dataset_manifest["splits"])
    text_field = str(dataset_manifest.get("text_field", "text"))

    print(f"Dataset manifest: {dataset_metadata_path}")
    print(f"Dataset name: {dataset_manifest['dataset_name']}")
    print(f"Dataset version: {dataset_manifest['dataset_version']}")
    print(f"Text field: {text_field}")

    for split_name in ("train", "validation", "test"):
        split_path = resolve_path(dataset_metadata_path, str(splits[split_name]))
        if not split_path.exists():
            raise FileNotFoundError(f"Missing {split_name} split: {split_path}")
        total, non_empty = load_jsonl_count(split_path, text_field)
        print(f"{split_name}: {split_path} ({total} rows, {non_empty} non-empty {text_field} values)")

    golden_set_path = Path(args.golden_set_path).resolve() if args.golden_set_path else None
    if golden_set_path is not None:
        golden_payload = read_json(golden_set_path)
        jsonschema.validate(golden_payload, golden_schema)
        prompts = golden_payload["benchmark_prompts"]
        print(f"Golden set: {golden_set_path}")
        print(f"Golden set name: {golden_payload['dataset_name']}")
        print(f"Golden set version: {golden_payload['dataset_version']}")
        print(f"Golden prompts: {len(prompts)}")


if __name__ == "__main__":
    main()
