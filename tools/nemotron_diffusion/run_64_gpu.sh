#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/snorouzi/diffusion_RL/JustGRPO-worktree}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl}"
RUN_NAME="${RUN_NAME:-justgrpo_leftmost_sglang_hf_instruct_64gpu_$(date +%Y%m%d_%H%M%S)}"
RUNDIR="${RUNDIR:-${RUN_ROOT}/${RUN_NAME}}"
mkdir -p "${RUNDIR}"

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh}"
MOUNTS="${MOUNTS:-/home/snorouzi:/home/snorouzi,/lustre:/lustre}"
RAY_CLI="${RAY_CLI:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_envs/diffusion_RL_JustGRPO_mcore/bin/ray}"
UV_ENV="${UV_ENV:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_envs/diffusion_RL_JustGRPO_mcore}"

POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/hub/models--nvidia--Nemotron-Diffusion-Exp-Ministral-3B-Instruct/snapshots/d7c52fbc82c29932c18a02478da6e93921daad34}"
SGLANG_DLLM_ALGORITHM_CONFIG="${SGLANG_DLLM_ALGORITHM_CONFIG:-${REPO_DIR}/tools/nemotron_diffusion/justgrpo_leftmost_dllm.yaml}"
MEGATRON_PATCH_DIR="${MEGATRON_PATCH_DIR:-/home/snorouzi/code/Megatron-Bridge/src:/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/modules}"

COMMAND="cd ${REPO_DIR} && env \
REPO_DIR=${REPO_DIR} \
RUN_NAME=${RUN_NAME} \
RUN_ROOT=${RUN_ROOT} \
RUNDIR=${RUNDIR} \
UV_PROJECT_ENVIRONMENT=${UV_ENV} \
NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemo_rl_ray_venvs_justgrpo} \
UV_CACHE_DIR=${UV_CACHE_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/uv_cache} \
CLUSTER_NUM_NODES=${CLUSTER_NUM_NODES:-8} \
CLUSTER_GPUS_PER_NODE=${CLUSTER_GPUS_PER_NODE:-8} \
POLICY_MODEL_NAME=${POLICY_MODEL_NAME} \
MAX_STEPS=${MAX_STEPS:-5000} \
MAX_EPOCHS=${MAX_EPOCHS:-1} \
PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-8} \
GENERATIONS_PER_PROMPT=${GENERATIONS_PER_PROMPT:-8} \
SAVE_PERIOD=${SAVE_PERIOD:-50} \
VAL_PERIOD=${VAL_PERIOD:-50} \
VAL_AT_START=${VAL_AT_START:-false} \
VAL_AT_END=${VAL_AT_END:-true} \
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-128} \
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128} \
TRAIN_GLOBAL_BATCH_SIZE=${TRAIN_GLOBAL_BATCH_SIZE:-64} \
TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1} \
LOGPROB_BATCH_SIZE=${LOGPROB_BATCH_SIZE:-64} \
KL_PENALTY=${KL_PENALTY:-0.01} \
JUSTGRPO_REVEAL_BATCH_SIZE=${JUSTGRPO_REVEAL_BATCH_SIZE:-64} \
JUSTGRPO_TRAIN_REVEAL_BATCH_SIZE=${JUSTGRPO_TRAIN_REVEAL_BATCH_SIZE:-1} \
MAX_TOTAL_SEQUENCE_LENGTH=${MAX_TOTAL_SEQUENCE_LENGTH:-640} \
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256} \
HF_CONFIG_SEQ_LENGTH=${HF_CONFIG_SEQ_LENGTH:-640} \
SGLANG_GPUS_PER_SERVER=${SGLANG_GPUS_PER_SERVER:-1} \
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-1} \
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.6} \
SGLANG_MAX_TOTAL_TOKENS=${SGLANG_MAX_TOTAL_TOKENS:-2048} \
SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH:-true} \
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-} \
SGLANG_DLLM_ALGORITHM=${SGLANG_DLLM_ALGORITHM:-FastDiffuser} \
SGLANG_DLLM_ALGORITHM_CONFIG=${SGLANG_DLLM_ALGORITHM_CONFIG} \
MEGATRON_TP_SIZE=${MEGATRON_TP_SIZE:-1} \
MEGATRON_SEQUENCE_PARALLEL=${MEGATRON_SEQUENCE_PARALLEL:-false} \
MEGATRON_CONVERTER_TYPE=${MEGATRON_CONVERTER_TYPE:-MinistralDiffEncoderModel} \
MEGATRON_FORCE_RECONVERT=${MEGATRON_FORCE_RECONVERT:-true} \
MEGATRON_PATCH_DIR=${MEGATRON_PATCH_DIR} \
NEMOTRON_DIFFUSION_DISABLE_COMPILED_FLEX_ATTENTION=${NEMOTRON_DIFFUSION_DISABLE_COMPILED_FLEX_ATTENTION:-0} \
WANDB_ENABLED=${WANDB_ENABLED:-true} \
WANDB_PROJECT=${WANDB_PROJECT:-diffusion_rl} \
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${RUN_NAME}} \
NEMO_RL_SGLANG_STARTUP_TIMEOUT=${NEMO_RL_SGLANG_STARTUP_TIMEOUT:-1200} \
bash tools/nemotron_diffusion/run_grpo_nemotron_justgrpo_sglang_interactive.sh"

env \
  CONTAINER="${CONTAINER}" \
  MOUNTS="${MOUNTS}" \
  BASE_LOG_DIR="${RUNDIR}" \
  GPUS_PER_NODE=8 \
  RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-120}" \
  RAY_CLI="${RAY_CLI}" \
  RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-300}" \
  COMMAND="${COMMAND}" \
  sbatch --nodes=8 \
    --account="${SLURM_ACCOUNT:-nvr_lpr_llm}" \
    --partition="${SLURM_PARTITION:-batch}" \
    --time="${SLURM_TIME:-04:00:00}" \
    --gres=gpu:8 \
    --job-name="${SLURM_JOB_NAME:-jg64_leftmost}" \
    --output="${RUNDIR}/slurm-%j.out" \
    ray.sub

printf "RUN_NAME=%s\nRUNDIR=%s\n" "${RUN_NAME}" "${RUNDIR}"
