#!/usr/bin/env bash
set -euo pipefail
python pipelines/training/training_pipeline.py
python pipelines/evaluation/evaluation_pipeline.py
