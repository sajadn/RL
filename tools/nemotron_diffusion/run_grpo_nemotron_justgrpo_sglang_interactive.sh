#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Run GRPO for the Nemotron-Diffusion Ministral-3B Instruct checkpoint with
# SGLang AR rollout generation and JustGRPO leftmost-reveal Megatron training.
# This script assumes it is already running on a GPU node/container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

RUN_NAME="${RUN_NAME:-grpo_nemotron_justgrpo_sglang_smoke}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl}"
RUNDIR="${RUNDIR:-${RUN_ROOT}/${RUN_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUNDIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-/tmp/${RUN_NAME}}"

CONFIG="${CONFIG:-examples/configs/grpo_math_nemotron_diffusion_3b_instruct_justgrpo_megatron_sglang.yaml}"
POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/Nemotron-Diffusion-Exp-Ministral-3B-Instruct-vllm-ar}"
DATASET_NAME="${DATASET_NAME:-OpenMathInstruct-2}"
DATASET_SPLIT="${DATASET_SPLIT:-train_1M}"
MAX_STEPS="${MAX_STEPS:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-2}"
GENERATIONS_PER_PROMPT="${GENERATIONS_PER_PROMPT:-2}"
SAVE_PERIOD="${SAVE_PERIOD:-1000}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-4}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE:-64}"
MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-640}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
HF_CONFIG_SEQ_LENGTH="${HF_CONFIG_SEQ_LENGTH:-${MAX_TOTAL_SEQUENCE_LENGTH}}"
SPLIT_VALIDATION_SIZE="${SPLIT_VALIDATION_SIZE:-0.01}"
VAL_PERIOD="${VAL_PERIOD:-1000}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
VAL_AT_START="${VAL_AT_START:-false}"
VAL_AT_END="${VAL_AT_END:-false}"
KL_PENALTY="${KL_PENALTY:-0.01}"
JUSTGRPO_REVEAL_BATCH_SIZE="${JUSTGRPO_REVEAL_BATCH_SIZE:-${LOGPROB_BATCH_SIZE}}"
JUSTGRPO_TRAIN_REVEAL_BATCH_SIZE="${JUSTGRPO_TRAIN_REVEAL_BATCH_SIZE:-${TRAIN_MICRO_BATCH_SIZE}}"
MEGATRON_TP_SIZE="${MEGATRON_TP_SIZE:-1}"
MEGATRON_SEQUENCE_PARALLEL="${MEGATRON_SEQUENCE_PARALLEL:-false}"
MEGATRON_CONVERTER_TYPE="${MEGATRON_CONVERTER_TYPE:-Ministral3ForCausalLM}"
MEGATRON_FORCE_RECONVERT="${MEGATRON_FORCE_RECONVERT:-false}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-32}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-}"
SGLANG_GPUS_PER_SERVER="${SGLANG_GPUS_PER_SERVER:-1}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-false}"
CLUSTER_NUM_NODES="${CLUSTER_NUM_NODES:-${NODES:-${SLURM_NNODES:-1}}}"
CLUSTER_GPUS_PER_NODE="${CLUSTER_GPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"
NEMOTRON_DIFFUSION_DISABLE_COMPILED_FLEX_ATTENTION="${NEMOTRON_DIFFUSION_DISABLE_COMPILED_FLEX_ATTENTION:-0}"
MEGATRON_PATCH_DIR="${MEGATRON_PATCH_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_runtime_patches}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-diffusion_rl}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/home/snorouzi/wandb_api_key.txt}"
TENSORBOARD_ENABLED="${TENSORBOARD_ENABLED:-false}"

mkdir -p "${RUNDIR}"

cd "${REPO_DIR}"

if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" || "${UV_PROJECT_ENVIRONMENT}" == /opt/* ]]; then
  export UV_PROJECT_ENVIRONMENT="/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_envs/diffusion_RL_JustGRPO_mcore"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/uv_cache}"
export RUSTUP_HOME="${RUSTUP_HOME:-/lustre/fsw/portfolios/coreai/users/snorouzi/rustup}"
export CARGO_HOME="${CARGO_HOME:-/lustre/fsw/portfolios/coreai/users/snorouzi/cargo}"
export PROTOC="${PROTOC:-/lustre/fsw/portfolios/coreai/users/snorouzi/protoc/bin/protoc}"
export PATH="${CARGO_HOME}/bin:$(dirname "${PROTOC}"):${PATH}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemo_rl_ray_venvs_justgrpo}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export UV_NO_BINARY_PACKAGE="${UV_NO_BINARY_PACKAGE:-}"
export NEMO_RL_SGLANG_KERNEL_SOURCE="${NEMO_RL_SGLANG_KERNEL_SOURCE:-/lustre/fsw/portfolios/coreai/users/snorouzi/wheels/sglang_kernel_torch210_cu129_py313/sglang_kernel-0.4.1-cp310-abi3-linux_x86_64.whl}"
export NEMO_RL_SGLANG_FLASHINFER_SPECS="${NEMO_RL_SGLANG_FLASHINFER_SPECS:-flashinfer_python==0.6.7.post3 flashinfer_cubin==0.6.7.post3}"
export NEMO_RL_SGLANG_SOURCE="${NEMO_RL_SGLANG_SOURCE:-}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemo_rl_megatron_ckpts_justgrpo}"
export MEGATRON_CONFIG_LOCK_DIR="${MEGATRON_CONFIG_LOCK_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/megatron_config_locks}"
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/transformers}"

HF_MODULES_DIR="${HF_HOME}/modules"
if [[ -d "${HF_MODULES_DIR}" && ":${MEGATRON_PATCH_DIR}:" != *":${HF_MODULES_DIR}:"* ]]; then
  MEGATRON_PATCH_DIR="${MEGATRON_PATCH_DIR}:${HF_MODULES_DIR}"
fi

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

# Avoid reusing a stale Ray session from shared node-local /tmp.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export TMPDIR="${TMPDIR:-/tmp/snorouzi_nemorl_${SLURM_JOB_ID}}"
  export RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR}}"
  mkdir -p "${TMPDIR}"
fi

# Important: do not expose the Megatron bridge patch to the Ray driver.
# Pass it only to Megatron policy workers via policy.megatron_cfg.env_vars.
unset PYTHONPATH

EXTRA_OVERRIDES=()
if [[ -n "${SGLANG_DLLM_ALGORITHM:-}" ]]; then
  EXTRA_OVERRIDES+=("policy.generation.sglang_cfg.dllm_algorithm=${SGLANG_DLLM_ALGORITHM}")
fi
if [[ -n "${SGLANG_DLLM_ALGORITHM_CONFIG:-}" ]]; then
  EXTRA_OVERRIDES+=("policy.generation.sglang_cfg.dllm_algorithm_config=${SGLANG_DLLM_ALGORITHM_CONFIG}")
fi
if [[ -n "${SGLANG_MAX_TOTAL_TOKENS}" ]]; then
  EXTRA_OVERRIDES+=("policy.generation.sglang_cfg.max_total_tokens=${SGLANG_MAX_TOTAL_TOKENS}")
fi

uv run --reinstall-package nemo-rl --extra mcore --with soundfile==0.13.1 --with onnx==1.19.1 python examples/run_grpo.py \
  --config "${CONFIG}" \
  "cluster.num_nodes=${CLUSTER_NUM_NODES}" \
  "cluster.gpus_per_node=${CLUSTER_GPUS_PER_NODE}" \
  "policy.model_name=${POLICY_MODEL_NAME}" \
  "policy.tokenizer.name=${POLICY_MODEL_NAME}" \
  "+policy.hf_config_overrides.seq_length=${HF_CONFIG_SEQ_LENGTH}" \
  "policy.generation.sglang_cfg.model_path=${POLICY_MODEL_NAME}" \
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
  "loss_fn.reference_policy_kl_penalty=${KL_PENALTY}" \
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
  "policy.generation.max_new_tokens=${MAX_NEW_TOKENS}" \
  "policy.generation.sglang_cfg.context_length=${MAX_TOTAL_SEQUENCE_LENGTH}" \
  "policy.generation.sglang_cfg.gpus_per_server=${SGLANG_GPUS_PER_SERVER}" \
  "policy.generation.sglang_cfg.max_running_requests=${SGLANG_MAX_RUNNING_REQUESTS}" \
  "policy.generation.sglang_cfg.mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}" \
  "policy.generation.sglang_cfg.disable_cuda_graph=${SGLANG_DISABLE_CUDA_GRAPH}" \
  "${EXTRA_OVERRIDES[@]}" \
  "policy.megatron_cfg.tensor_model_parallel_size=${MEGATRON_TP_SIZE}" \
  "policy.megatron_cfg.sequence_parallel=${MEGATRON_SEQUENCE_PARALLEL}" \
  "policy.megatron_cfg.converter_type=${MEGATRON_CONVERTER_TYPE}" \
  "policy.megatron_cfg.force_reconvert_from_hf=${MEGATRON_FORCE_RECONVERT}" \
  "policy.logprob_estimation.reveal_batch_size=${JUSTGRPO_REVEAL_BATCH_SIZE}" \
  "policy.logprob_estimation.train_reveal_batch_size=${JUSTGRPO_TRAIN_REVEAL_BATCH_SIZE}" \
  "policy.megatron_cfg.env_vars={PYTHONPATH:${MEGATRON_PATCH_DIR},NEMOTRON_DIFFUSION_DISABLE_COMPILED_FLEX_ATTENTION:\"${NEMOTRON_DIFFUSION_DISABLE_COMPILED_FLEX_ATTENTION}\"}" \
  "data.max_input_seq_length=${MAX_TOTAL_SEQUENCE_LENGTH}" \
  2>&1 | tee "${RUNDIR}/run.log"
