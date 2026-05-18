#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Run GRPO for the Nemotron-Diffusion Ministral-3B AR checkpoint with
# vLLM generation and Megatron policy training. This variant assumes you are
# already on an interactive node/container with GPUs visible.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

RUN_NAME="${RUN_NAME:-grpo_nemotron_ar_megatron_policy_math50}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl}"
RUNDIR="${RUNDIR:-${RUN_ROOT}/${RUN_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUNDIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-/tmp/${RUN_NAME}}"
RESET_CHECKPOINTS="${RESET_CHECKPOINTS:-1}"
RESET_LOG_DIR="${RESET_LOG_DIR:-1}"

CONFIG="${CONFIG:-examples/configs/grpo_math_nemotron_diffusion_3b_ar_megatron.yaml}"
POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/Nemotron-Diffusion-Exp-Ministral-3B-vllm-ar}"
DATASET_NAME="${DATASET_NAME:-OpenMathInstruct-2}"
DATASET_SPLIT="${DATASET_SPLIT:-train_1M}"
MAX_STEPS="${MAX_STEPS:-50}"
MAX_EPOCHS="${MAX_EPOCHS:-2}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-8}"
GENERATIONS_PER_PROMPT="${GENERATIONS_PER_PROMPT:-8}"
SAVE_PERIOD="${SAVE_PERIOD:-10}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-64}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-8}"
LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE:-64}"
MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-640}"
HF_CONFIG_SEQ_LENGTH="${HF_CONFIG_SEQ_LENGTH:-${MAX_TOTAL_SEQUENCE_LENGTH}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SPLIT_VALIDATION_SIZE="${SPLIT_VALIDATION_SIZE:-0.01}"
VAL_PERIOD="${VAL_PERIOD:-10}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
VAL_AT_START="${VAL_AT_START:-false}"
VAL_AT_END="${VAL_AT_END:-true}"
MEGATRON_PATCH_DIR="${MEGATRON_PATCH_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_runtime_patches}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-diffusion_rl}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/home/snorouzi/wandb_api_key.txt}"
TENSORBOARD_ENABLED="${TENSORBOARD_ENABLED:-false}"
EXTRA_GRPO_OVERRIDES="${EXTRA_GRPO_OVERRIDES:-}"
EXTRA_GRPO_ARGS=()
if [[ -n "${EXTRA_GRPO_OVERRIDES}" ]]; then
  read -r -a EXTRA_GRPO_ARGS <<< "${EXTRA_GRPO_OVERRIDES}"
fi

mkdir -p "${RUNDIR}"
if [[ "${RESET_CHECKPOINTS}" == "1" ]]; then
  rm -rf "${CHECKPOINT_DIR}"
fi
if [[ "${RESET_LOG_DIR}" == "1" ]]; then
  rm -rf "${LOG_DIR}"
fi

cd "${REPO_DIR}"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_envs/diffusion_RL_RL_mcore}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/uv_cache}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemo_rl_megatron_ckpts}"
export MEGATRON_CONFIG_LOCK_DIR="${MEGATRON_CONFIG_LOCK_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/megatron_config_locks}"
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/transformers}"

if [[ -z "${HF_TOKEN:-}" && -f /home/snorouzi/hf_token.txt ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d "\n\r" < /home/snorouzi/hf_token.txt)"
fi
export HF_HUB_TOKEN="${HF_HUB_TOKEN:-${HF_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_API_KEY_FILE}" ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d "\n\r" < "${WANDB_API_KEY_FILE}")"
fi

export HOME="${CONTAINER_HOME:-/lustre/fsw/portfolios/coreai/users/snorouzi/container_home}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-1}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-120}"
export NRL_REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.3}"

# Important: do not expose the Megatron bridge patch to the Ray driver.
# Pass it only to Megatron policy workers via policy.megatron_cfg.env_vars.
unset PYTHONPATH
if [[ -n "${SGLANG_SOURCE_PATH:-}" ]]; then
  export PYTHONPATH="${SGLANG_SOURCE_PATH}"
fi

uv run --extra mcore --with soundfile==0.13.1 --with onnx==1.19.1 python examples/run_grpo.py \
  --config "${CONFIG}" \
  "policy.model_name=${POLICY_MODEL_NAME}" \
  "policy.tokenizer.name=${POLICY_MODEL_NAME}" \
  "data.train.dataset_name=${DATASET_NAME}" \
  "+data.train.split=${DATASET_SPLIT}" \
  "data.train.split_validation_size=${SPLIT_VALIDATION_SIZE}" \
  "grpo.max_num_steps=${MAX_STEPS}" \
  "grpo.max_num_epochs=${MAX_EPOCHS}" \
  "grpo.num_prompts_per_step=${PROMPTS_PER_STEP}" \
  "grpo.num_generations_per_prompt=${GENERATIONS_PER_PROMPT}" \
  "grpo.val_at_start=${VAL_AT_START}" \
  "grpo.val_at_end=${VAL_AT_END}" \
  "grpo.val_period=${VAL_PERIOD}" \
  "grpo.max_val_samples=${MAX_VAL_SAMPLES}" \
  "grpo.val_batch_size=${VAL_BATCH_SIZE}" \
  checkpointing.enabled=true \
  "checkpointing.checkpoint_dir=${CHECKPOINT_DIR}" \
  "checkpointing.save_period=${SAVE_PERIOD}" \
  checkpointing.save_optimizer=true \
  checkpointing.save_consolidated=false \
  "logger.log_dir=${LOG_DIR}" \
  "logger.wandb_enabled=${WANDB_ENABLED}" \
  "logger.wandb.project=${WANDB_PROJECT}" \
  "logger.wandb.name=${WANDB_RUN_NAME}" \
  "logger.tensorboard_enabled=${TENSORBOARD_ENABLED}" \
  "policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE}" \
  "policy.train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE}" \
  "policy.logprob_batch_size=${LOGPROB_BATCH_SIZE}" \
  "policy.max_total_sequence_length=${MAX_TOTAL_SEQUENCE_LENGTH}" \
  "+policy.hf_config_overrides.seq_length=${HF_CONFIG_SEQ_LENGTH}" \
  "policy.generation.max_new_tokens=${MAX_NEW_TOKENS}" \
  "policy.generation.vllm_cfg.max_model_len=${MAX_TOTAL_SEQUENCE_LENGTH}" \
  policy.generation.vllm_cfg.enforce_eager=true \
  "policy.megatron_cfg.env_vars={PYTHONPATH:${MEGATRON_PATCH_DIR}}" \
  "data.max_input_seq_length=${MAX_TOTAL_SEQUENCE_LENGTH}" \
  "${EXTRA_GRPO_ARGS[@]}" \
  2>&1 | tee "${RUNDIR}/run.log"
