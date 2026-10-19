#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="$(realpath "$(dirname "$0")/../../../research/just_grpo/configs/recipes/just_grpo-gsm8k-llada8b-canvas256-2n8g-megatron-inference-smoke-diffaux.yaml")"
source "$(dirname "$0")/common.env"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/research/block_diffusion:${PROJECT_ROOT}/research/just_grpo:${PYTHONPATH:-}"
uv run --extra mcore research/just_grpo/run_just_grpo.py --config "${CONFIG_PATH}" \
  "logger.log_dir=${LOG_DIR}" "checkpointing.checkpoint_dir=${CKPT_DIR}"
