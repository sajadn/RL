#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Run GRPO for Nemotron-Diffusion AR experiments on an already allocated node.
#
# The YAML config is the source of truth for algorithm defaults. This wrapper
# only handles runtime setup plus run-specific filesystem/model overrides.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

: "${CONFIG:?CONFIG must point to the experiment YAML.}"
: "${RUN_NAME:?RUN_NAME is required for the run directory and default W&B name.}"

RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl}"
RUNDIR="${RUNDIR:-${RUN_ROOT}/${RUN_NAME}}"

EXTRA_GRPO_OVERRIDES="${EXTRA_GRPO_OVERRIDES:-}"

cd "${REPO_DIR}"
eval "$(python3 tools/nemotron_diffusion/extract_runtime_env.py "${CONFIG}")"

: "${CHECKPOINT_DIR:?runtime_env.launcher.checkpoint_dir must be set in ${CONFIG}.}"
: "${LOG_DIR:?runtime_env.launcher.log_dir must be set in ${CONFIG}.}"
: "${RESET_CHECKPOINTS:?runtime_env.launcher.reset_checkpoints must be set in ${CONFIG}.}"
: "${RESET_LOG_DIR:?runtime_env.launcher.reset_log_dir must be set in ${CONFIG}.}"

mkdir -p "${RUNDIR}"
if [[ "${RESET_CHECKPOINTS}" == "1" ]]; then
  rm -rf "${CHECKPOINT_DIR}"
fi
if [[ "${RESET_LOG_DIR}" == "1" ]]; then
  rm -rf "${LOG_DIR}"
fi

# Important: do not expose the Megatron bridge patch to the Ray driver.
# Pass it only to Megatron policy workers via policy.megatron_cfg.env_vars.
unset PYTHONPATH
if [[ -n "${SGLANG_SOURCE_PATH:-}" ]]; then
  export PYTHONPATH="${SGLANG_SOURCE_PATH}"
fi

GRPO_ARGS=(
  --config "${CONFIG}"
)

if [[ -n "${MEGATRON_PATCH_DIR:-}" ]]; then
  GRPO_ARGS+=("policy.megatron_cfg.env_vars={PYTHONPATH:${MEGATRON_PATCH_DIR}}")
fi

if [[ -n "${EXTRA_GRPO_OVERRIDES}" ]]; then
  read -r -a EXTRA_GRPO_ARGS <<< "${EXTRA_GRPO_OVERRIDES}"
  GRPO_ARGS+=("${EXTRA_GRPO_ARGS[@]}")
fi

uv run --extra mcore --with soundfile==0.13.1 --with onnx==1.19.1 python examples/run_grpo.py \
  "${GRPO_ARGS[@]}" \
  2>&1 | tee "${RUNDIR}/run.log"
