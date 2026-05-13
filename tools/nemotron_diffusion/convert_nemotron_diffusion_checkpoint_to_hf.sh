#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Export a Nemotron Diffusion Megatron checkpoint to Hugging Face format.

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  STEP_DIR=/path/to/checkpoints/step_N OUT=/path/to/hf_out \
    tools/nemotron_diffusion/convert_nemotron_diffusion_checkpoint_to_hf.sh

Required:
  STEP_DIR  Checkpoint step directory containing config.yaml and policy/weights/iter_0000000.
  OUT       Output Hugging Face checkpoint directory.

Optional:
  REPO_DIR                      NeMo-RL repo directory.
  BASE_MODEL                    HF model used as the Megatron-Bridge conversion template.
                                Defaults to policy.model_name from STEP_DIR/config.yaml.
  MEGATRON_PATCH_DIR            PYTHONPATH prefix containing Nemotron Diffusion Megatron-Bridge.
  NEMOTRON_UV_PROJECT_ENVIRONMENT
  NEMOTRON_UV_CACHE_DIR
  NEMOTRON_HF_HOME
  MEGATRON_CONFIG_LOCK_DIR
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STEP_DIR="${STEP_DIR:?STEP_DIR is required. See --help.}"
OUT="${OUT:?OUT is required. See --help.}"

HF_MODULES_DIR="${HF_MODULES_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/modules}"
MEGATRON_PATCH_DIR="${MEGATRON_PATCH_DIR:-/home/snorouzi/code/Megatron-Bridge/src:${HF_MODULES_DIR}}"
MEGATRON_CONFIG_LOCK_DIR="${MEGATRON_CONFIG_LOCK_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/megatron_config_locks}"
UV_PROJECT_ENVIRONMENT="${NEMOTRON_UV_PROJECT_ENVIRONMENT:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_envs/nemotron_diffusion_checkpoint_to_hf}"
UV_CACHE_DIR="${NEMOTRON_UV_CACHE_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/uv_cache}"
HF_HOME="${NEMOTRON_HF_HOME:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home}"

MEGATRON_CKPT_PATH="${MEGATRON_CKPT_PATH:-${STEP_DIR}/policy/weights/iter_0000000}"
CONFIG_PATH="${CONFIG_PATH:-${STEP_DIR}/config.yaml}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -d "${MEGATRON_CKPT_PATH}" ]]; then
  echo "Missing Megatron checkpoint directory: ${MEGATRON_CKPT_PATH}" >&2
  exit 1
fi

export UV_PROJECT_ENVIRONMENT
export UV_CACHE_DIR
export HF_HOME
export MEGATRON_CONFIG_LOCK_DIR
export HF_TOKEN="${HF_TOKEN:-$(cat /home/snorouzi/hf_token.txt 2>/dev/null || true)}"
export PYTHONPATH="${MEGATRON_PATCH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_DIR}"

if [[ -z "${BASE_MODEL:-}" ]]; then
  BASE_MODEL="$(uv run python - "${CONFIG_PATH}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r") as f:
    config = yaml.safe_load(f)
print(config["policy"]["model_name"])
PY
)"
fi

rm -rf "${OUT}"

uv run --extra mcore --with onnx==1.19.1 python examples/converters/convert_megatron_to_hf.py \
  --config "${CONFIG_PATH}" \
  --hf-model-name "${BASE_MODEL}" \
  --megatron-ckpt-path "${MEGATRON_CKPT_PATH}" \
  --hf-ckpt-path "${OUT}"

uv run python - "${OUT}/config.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
print(
    {
        key: config.get(key)
        for key in (
            "model_type",
            "architectures",
            "tie_word_embeddings",
            "vocab_size",
            "hidden_size",
        )
    }
)
PY
