#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x ".venv/bin/python" ]]; then
        PYTHON_BIN=".venv/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        PYTHON_BIN="python3"
    fi
fi

"$PYTHON_BIN" pipelines/training/training_pipeline.py
"$PYTHON_BIN" pipelines/evaluation/evaluation_pipeline.py
"$PYTHON_BIN" pipelines/evaluation/single_model_evaluation_pipeline.py
