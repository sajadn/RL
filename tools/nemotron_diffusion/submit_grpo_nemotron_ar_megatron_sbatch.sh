#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Submit GRPO for the Nemotron-Diffusion Ministral-3B AR checkpoint with vLLM
# generation and Megatron policy training.

set -euo pipefail

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-batch}"
TIME="${TIME:-04:00:00}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
JOB_NAME="${JOB_NAME:-grpo_nd_3b_ar}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${HOME}:/home/snorouzi,/lustre:/lustre}"
REPO_DIR="${REPO_DIR:-/home/snorouzi/diffusion_RL/RL}"
RUN_NAME="${RUN_NAME:-grpo_nemotron_ar_megatron_policy_5k}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl}"
RUNDIR="${RUNDIR:-${RUN_ROOT}/${RUN_NAME}}"
DEPENDENCY="${DEPENDENCY:-}"

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
export CONFIG="${CONFIG:-examples/configs/grpo_math_nemotron_diffusion_3b_ar_megatron.yaml}"
export RUN_NAME="${RUN_NAME}"
export RUN_ROOT="${RUN_ROOT}"
export RUNDIR="${RUNDIR}"
export POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/Nemotron-Diffusion-Exp-Ministral-3B-vllm-ar}"
export RESET_CHECKPOINTS="${RESET_CHECKPOINTS:-1}"
export RESET_LOG_DIR="${RESET_LOG_DIR:-1}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export MAX_EPOCHS="${MAX_EPOCHS:-2}"
export PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-8}"
export GENERATIONS_PER_PROMPT="${GENERATIONS_PER_PROMPT:-8}"
export SAVE_PERIOD="${SAVE_PERIOD:-1000}"
export VAL_PERIOD="${VAL_PERIOD:-200}"
export TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-64}"
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-8}"
export LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE:-64}"
export MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-640}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-diffusion_rl}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TENSORBOARD_ENABLED="${TENSORBOARD_ENABLED:-false}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemo_rl_ray_venvs}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export UV_NO_BINARY_PACKAGE="${UV_NO_BINARY_PACKAGE:-}"
export NEMO_RL_SGLANG_KERNEL_SOURCE="${NEMO_RL_SGLANG_KERNEL_SOURCE:-/lustre/fsw/portfolios/coreai/users/snorouzi/wheels/sglang_kernel_torch210_cu129_py313/sglang_kernel-0.4.1-cp310-abi3-linux_x86_64.whl}"
export NEMO_RL_SGLANG_FLASHINFER_SPECS="${NEMO_RL_SGLANG_FLASHINFER_SPECS:-flashinfer_python==0.6.7.post3 flashinfer_cubin==0.6.7.post3}"

srun --kill-on-bad-exit=1 \\
  --container-image="${CONTAINER_IMAGE}" \\
  --container-mounts="${CONTAINER_MOUNTS}" \\
  --container-workdir="${REPO_DIR}" \\
  bash -lc 'cd "\${REPO_DIR}" && tools/nemotron_diffusion/run_grpo_nemotron_ar_megatron_interactive.sh'
SBATCH
