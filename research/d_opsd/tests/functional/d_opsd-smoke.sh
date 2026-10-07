#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail

DOPSD_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$DOPSD_REPO_ROOT"
: "${DOPSD_MODEL_PATH:?Set DOPSD_MODEL_PATH to a Nemotron Labs Diffusion checkpoint}"
mkdir -p "${DOPSD_ARTIFACT_ROOT:-$DOPSD_REPO_ROOT/results}"
DOPSD_OUTPUT_DIR=$(mktemp -d "${DOPSD_ARTIFACT_ROOT:-$DOPSD_REPO_ROOT/results}/d_opsd_smoke.XXXXXX")
uv run --all-packages --extra mcore python research/d_opsd/run_d_opsd.py \
  --config research/d_opsd/configs/recipes/d_opsd-sudoku6x6-1n8g-megatron-smoke.yaml \
  policy.model_name="$DOPSD_MODEL_PATH" logger.log_dir="$DOPSD_OUTPUT_DIR" "$@"
