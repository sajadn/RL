#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Submit GRPO for the Nemotron-Diffusion Ministral-3B checkpoint with SGLang
# FastDiffuser rollout generation and JustGRPO leftmost-reveal Megatron training.

set -euo pipefail

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-batch}"
TIME="${TIME:-04:00:00}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CLUSTER_NUM_NODES="${CLUSTER_NUM_NODES:-${NODES}}"
CLUSTER_GPUS_PER_NODE="${CLUSTER_GPUS_PER_NODE:-${GPUS_PER_NODE}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
JOB_NAME="${JOB_NAME:-grpo_nd_3b_jg_sglang}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${HOME}:/home/snorouzi,/lustre:/lustre}"
REPO_DIR="${REPO_DIR:-/home/snorouzi/diffusion_RL/JustGRPO-worktree}"
RUN_NAME="${RUN_NAME:-grpo_nemotron_justgrpo_sglang_5k}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl}"
RUNDIR="${RUNDIR:-${RUN_ROOT}/${RUN_NAME}}"
DEPENDENCY="${DEPENDENCY:-}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-64}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-8}"
LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE:-64}"

mkdir -p "${RUNDIR}"

SBATCH_ARGS=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${TIME}"
  --nodes="${NODES}"
  --gres="gpu:${GPUS_PER_NODE}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --job-name="${JOB_NAME}"
  --output="${RUNDIR}/slurm-%j.out"
  --open-mode=append
)

if [[ -n "${DEPENDENCY}" ]]; then
  SBATCH_ARGS+=(--dependency="${DEPENDENCY}")
fi

sbatch "${SBATCH_ARGS[@]}" <<SBATCH
#!/usr/bin/env bash
set -euo pipefail

export REPO_DIR="${REPO_DIR}"
export CLUSTER_NUM_NODES="${CLUSTER_NUM_NODES}"
export CLUSTER_GPUS_PER_NODE="${CLUSTER_GPUS_PER_NODE}"
export CONFIG="${CONFIG:-examples/configs/grpo_math_nemotron_diffusion_3b_instruct_justgrpo_megatron_sglang.yaml}"
export RUN_NAME="${RUN_NAME}"
export RUN_ROOT="${RUN_ROOT}"
export RUNDIR="${RUNDIR}"
export POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/Nemotron-Diffusion-Exp-Ministral-3B}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export MAX_EPOCHS="${MAX_EPOCHS:-1}"
export PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-8}"
export GENERATIONS_PER_PROMPT="${GENERATIONS_PER_PROMPT:-8}"
export SAVE_PERIOD="${SAVE_PERIOD:-50}"
export VAL_PERIOD="${VAL_PERIOD:-50}"
export TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE}"
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE}"
export LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE}"
export MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-640}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export HF_CONFIG_SEQ_LENGTH="${HF_CONFIG_SEQ_LENGTH:-${MAX_TOTAL_SEQUENCE_LENGTH:-640}}"
export MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-128}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
export VAL_AT_START="${VAL_AT_START:-false}"
export VAL_AT_END="${VAL_AT_END:-true}"
export JUSTGRPO_REVEAL_BATCH_SIZE="${JUSTGRPO_REVEAL_BATCH_SIZE:-64}"
export JUSTGRPO_TRAIN_REVEAL_BATCH_SIZE="${JUSTGRPO_TRAIN_REVEAL_BATCH_SIZE:-8}"
export MEGATRON_TP_SIZE="${MEGATRON_TP_SIZE:-1}"
export MEGATRON_SEQUENCE_PARALLEL="${MEGATRON_SEQUENCE_PARALLEL:-false}"
export MEGATRON_CONVERTER_TYPE="${MEGATRON_CONVERTER_TYPE:-MinistralDiffEncoderModel}"
export MEGATRON_FORCE_RECONVERT="${MEGATRON_FORCE_RECONVERT:-false}"
export SGLANG_GPUS_PER_SERVER="${SGLANG_GPUS_PER_SERVER:-1}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-1}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"
export SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-2048}"
export SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-true}"
export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-diffusion_rl}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TENSORBOARD_ENABLED="${TENSORBOARD_ENABLED:-false}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemo_rl_ray_venvs}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export UV_NO_BINARY_PACKAGE="${UV_NO_BINARY_PACKAGE:-}"
export NEMO_RL_SGLANG_SOURCE="${NEMO_RL_SGLANG_SOURCE:-}"
export NEMO_RL_SGLANG_KERNEL_SOURCE="${NEMO_RL_SGLANG_KERNEL_SOURCE:-/lustre/fsw/portfolios/coreai/users/snorouzi/wheels/sglang_kernel_torch210_cu129_py313/sglang_kernel-0.4.1-cp310-abi3-linux_x86_64.whl}"
export NEMO_RL_SGLANG_FLASHINFER_SPECS="${NEMO_RL_SGLANG_FLASHINFER_SPECS:-flashinfer_python==0.6.7.post3 flashinfer_cubin==0.6.7.post3}"
export SGLANG_DLLM_ALGORITHM="${SGLANG_DLLM_ALGORITHM:-FastDiffuser}"
export SGLANG_DLLM_ALGORITHM_CONFIG="${SGLANG_DLLM_ALGORITHM_CONFIG:-${REPO_DIR}/tools/nemotron_diffusion/justgrpo_leftmost_dllm.yaml}"
export MEGATRON_PATCH_DIR="${MEGATRON_PATCH_DIR:-/home/snorouzi/code/Megatron-Bridge/src}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemo_rl_megatron_ckpts_justgrpo}"
export NEMOTRON_DIFFUSION_DISABLE_COMPILED_FLEX_ATTENTION="${NEMOTRON_DIFFUSION_DISABLE_COMPILED_FLEX_ATTENTION:-0}"
export SGLANG_PORT_OFFSET="${SGLANG_PORT_OFFSET:-0}"
export NEMO_RL_SGLANG_STARTUP_TIMEOUT="${NEMO_RL_SGLANG_STARTUP_TIMEOUT:-1200}"

srun --kill-on-bad-exit=1 \\
  --container-image="${CONTAINER_IMAGE}" \\
  --container-mounts="${CONTAINER_MOUNTS}" \\
  --container-workdir="${REPO_DIR}" \\
  bash -lc 'cd "\${REPO_DIR}" && tools/nemotron_diffusion/run_grpo_nemotron_justgrpo_sglang_interactive.sh'
SBATCH
