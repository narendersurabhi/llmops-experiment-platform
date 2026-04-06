#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from components.reporting.local_leaderboard import generate_leaderboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a local experiment leaderboard from training and evaluation artifacts.")
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--train-root", default=str(repo_root / "artifacts" / "local-train"))
    parser.add_argument("--eval-root", default=str(repo_root / "artifacts" / "local-eval"))
    parser.add_argument("--output-dir", default=str(repo_root / "artifacts" / "reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    markdown_path, csv_path = generate_leaderboard(
        repo_root=repo_root,
        train_root=Path(args.train_root).resolve(),
        eval_root=Path(args.eval_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
    )
    print(f"Leaderboard markdown: {markdown_path}")
    print(f"Leaderboard csv: {csv_path}")


if __name__ == "__main__":
    main()
