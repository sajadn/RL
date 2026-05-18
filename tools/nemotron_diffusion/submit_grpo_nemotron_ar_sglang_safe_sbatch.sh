#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Submit GRPO for the Nemotron-Diffusion checkpoint using SGLang dLLM AR
# generation and Megatron policy training. The generated config is staged under
# the run directory on /lustre so it is visible inside the Slurm container.

set -euo pipefail

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-batch}"
TIME="${TIME:-04:00:00}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
JOB_NAME="${JOB_NAME:-grpo_ar_sglang}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${HOME}:/home/snorouzi,/lustre:/lustre}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl}"
RUN_NAME="${RUN_NAME:-grpo_ar_sglang_dllm_$(date +%Y%m%d_%H%M%S)}"
RUNDIR="${RUNDIR:-${RUN_ROOT}/${RUN_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUNDIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-${RUNDIR}/logs}"
CONFIG_DIR="${CONFIG_DIR:-${RUNDIR}/configs}"
CONFIG="${CONFIG:-${CONFIG_DIR}/ar_sglang.yaml}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_DIR}/examples/configs/grpo_math_nemotron_diffusion_3b_instruct_ar_megatron_sglang.yaml}"
DEPENDENCY="${DEPENDENCY:-}"

POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/Nemotron-Diffusion-Exp-Ministral-3B}"
MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-640}"
HF_CONFIG_SEQ_LENGTH="${HF_CONFIG_SEQ_LENGTH:-${MAX_TOTAL_SEQUENCE_LENGTH}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.4}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-32}"
SGLANG_SKIP_SERVER_WARMUP="${SGLANG_SKIP_SERVER_WARMUP:-true}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-true}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"
SGLANG_ALLOW_AUTO_TRUNCATE="${SGLANG_ALLOW_AUTO_TRUNCATE:-true}"
SGLANG_ENABLE_MEMORY_SAVER="${SGLANG_ENABLE_MEMORY_SAVER:-false}"
SGLANG_DLLM_ALGORITHM="${SGLANG_DLLM_ALGORITHM:-AR}"
SGLANG_DLLM_ALGORITHM_CONFIG="${SGLANG_DLLM_ALGORITHM_CONFIG:-}"
if [[ -z "${SGLANG_DLLM_ALGORITHM}" || "${SGLANG_DLLM_ALGORITHM}" == "null" || "${SGLANG_DLLM_ALGORITHM}" == "None" ]]; then
  SGLANG_DLLM_ALGORITHM_YAML="null"
else
  SGLANG_DLLM_ALGORITHM_YAML="\"${SGLANG_DLLM_ALGORITHM}\""
fi

MAX_STEPS="${MAX_STEPS:-5000}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-8}"
GENERATIONS_PER_PROMPT="${GENERATIONS_PER_PROMPT:-8}"
SAVE_PERIOD="${SAVE_PERIOD:-100}"
VAL_PERIOD="${VAL_PERIOD:-100}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-64}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-8}"
SEQUENCE_PACKING_ENABLED="${SEQUENCE_PACKING_ENABLED:-false}"
if [[ "${SEQUENCE_PACKING_ENABLED}" == "true" ]]; then
  LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE:-64}"
else
  LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE:-${TRAIN_MICRO_BATCH_SIZE}}"
fi
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-diffusion_rl}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
TENSORBOARD_ENABLED="${TENSORBOARD_ENABLED:-false}"
EXTRA_GRPO_OVERRIDES="${EXTRA_GRPO_OVERRIDES:-}"
RESET_CHECKPOINTS="${RESET_CHECKPOINTS:-1}"
RESET_LOG_DIR="${RESET_LOG_DIR:-1}"
MEGATRON_PATCH_DIR="${MEGATRON_PATCH_DIR:-/home/snorouzi/code/Megatron-Bridge/src:/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/modules}"
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${RUNDIR}/mcore_ckpts}"
MEGATRON_CONVERTER_TYPE="${MEGATRON_CONVERTER_TYPE:-MinistralDiffEncoderModel}"
MEGATRON_DIFFUSION_ATTENTION_MODE="${MEGATRON_DIFFUSION_ATTENTION_MODE:-inference_causal}"
SGLANG_SOURCE_PATH="${SGLANG_SOURCE_PATH:-/home/snorouzi/code/sglang-nemotron-dllm-lora-eval/python}"
NEMO_RL_SGLANG_SOURCE="${NEMO_RL_SGLANG_SOURCE:-}"
NEMO_RL_SGLANG_KERNEL_SOURCE="${NEMO_RL_SGLANG_KERNEL_SOURCE:-/lustre/fsw/portfolios/coreai/users/snorouzi/wheels/sglang_kernel_torch210_cu129_py313/sglang_kernel-0.4.1-cp310-abi3-linux_x86_64.whl}"
NEMO_RL_SGLANG_FLASHINFER_SPECS="${NEMO_RL_SGLANG_FLASHINFER_SPECS:-flashinfer_python==0.6.7.post3 flashinfer_cubin==0.6.7.post3}"
SBATCH_TMPDIR="${TMPDIR:-}"
if [[ -z "${SBATCH_TMPDIR}" ]]; then
  SBATCH_TMPDIR='/tmp/nrl_${SLURM_JOB_ID}'
fi
SBATCH_RAY_TMPDIR="${RAY_TMPDIR:-}"
if [[ -z "${SBATCH_RAY_TMPDIR}" ]]; then
  SBATCH_RAY_TMPDIR='/tmp/ray_${SLURM_JOB_ID}'
fi

mkdir -p "${CONFIG_DIR}" "${RUNDIR}"

cat > "${CONFIG}" <<YAML
defaults: "${BASE_CONFIG}"

checkpointing:
  checkpoint_dir: "${CHECKPOINT_DIR}"

policy:
  model_name: "${POLICY_MODEL_NAME}"
  tokenizer:
    name: "${POLICY_MODEL_NAME}"
  sequence_packing:
    enabled: ${SEQUENCE_PACKING_ENABLED}
  generation:
    backend: "sglang"
    max_new_tokens: ${MAX_NEW_TOKENS}
    temperature: ${TEMPERATURE}
    top_p: ${TOP_P}
    top_k: null
    stop_token_ids: null
    stop_strings: null
    sglang_cfg:
      model_path: "${POLICY_MODEL_NAME}"
      gpus_per_server: 1
      dtype: \${policy.precision}
      context_length: \${policy.max_total_sequence_length}
      allow_auto_truncate: ${SGLANG_ALLOW_AUTO_TRUNCATE}
      enable_memory_saver: ${SGLANG_ENABLE_MEMORY_SAVER}
      dp_size: 1
      pp_size: 1
      ep_size: 1
      max_running_requests: ${SGLANG_MAX_RUNNING_REQUESTS}
      mem_fraction_static: ${SGLANG_MEM_FRACTION_STATIC}
      skip_server_warmup: ${SGLANG_SKIP_SERVER_WARMUP}
      disable_cuda_graph: ${SGLANG_DISABLE_CUDA_GRAPH}
      disable_piecewise_cuda_graph: ${SGLANG_DISABLE_CUDA_GRAPH}
      attention_backend: "${SGLANG_ATTENTION_BACKEND}"
      dllm_algorithm: ${SGLANG_DLLM_ALGORITHM_YAML}
      dllm_algorithm_config: ${SGLANG_DLLM_ALGORITHM_CONFIG:-null}
  megatron_cfg:
    converter_type: "${MEGATRON_CONVERTER_TYPE}"
    diffusion_attention_mode: "${MEGATRON_DIFFUSION_ATTENTION_MODE}"

logger:
  log_dir: "${LOG_DIR}"
  wandb:
    project: "${WANDB_PROJECT}"
    name: "${WANDB_RUN_NAME}"
YAML

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
export CONFIG="${CONFIG}"
export RUN_NAME="${RUN_NAME}"
export RUN_ROOT="${RUN_ROOT}"
export RUNDIR="${RUNDIR}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR}"
export LOG_DIR="${LOG_DIR}"
export POLICY_MODEL_NAME="${POLICY_MODEL_NAME}"
export RESET_CHECKPOINTS="${RESET_CHECKPOINTS}"
export RESET_LOG_DIR="${RESET_LOG_DIR}"
export MAX_STEPS="${MAX_STEPS}"
export MAX_EPOCHS="${MAX_EPOCHS}"
export PROMPTS_PER_STEP="${PROMPTS_PER_STEP}"
export GENERATIONS_PER_PROMPT="${GENERATIONS_PER_PROMPT}"
export SAVE_PERIOD="${SAVE_PERIOD}"
export VAL_PERIOD="${VAL_PERIOD}"
export TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE}"
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE}"
export LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE}"
export MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH}"
export HF_CONFIG_SEQ_LENGTH="${HF_CONFIG_SEQ_LENGTH}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS}"
export WANDB_ENABLED="${WANDB_ENABLED}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME}"
export TENSORBOARD_ENABLED="${TENSORBOARD_ENABLED}"
export EXTRA_GRPO_OVERRIDES="${EXTRA_GRPO_OVERRIDES}"
export MEGATRON_PATCH_DIR="${MEGATRON_PATCH_DIR}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR}"
export MEGATRON_CONVERTER_TYPE="${MEGATRON_CONVERTER_TYPE}"
export MEGATRON_DIFFUSION_ATTENTION_MODE="${MEGATRON_DIFFUSION_ATTENTION_MODE}"
export SGLANG_SOURCE_PATH="${SGLANG_SOURCE_PATH}"
export SGLANG_PORT_OFFSET="${SGLANG_PORT_OFFSET:-0}"
export NEMO_RL_SGLANG_PORT_MIN="${NEMO_RL_SGLANG_PORT_MIN:-20000}"
export NEMO_RL_SGLANG_PORT_MAX="${NEMO_RL_SGLANG_PORT_MAX:-60000}"
export NEMO_RL_SGLANG_RANK_PORT_STRIDE="${NEMO_RL_SGLANG_RANK_PORT_STRIDE:-128}"
export NEMO_RL_SGLANG_PORT_SCAN_ATTEMPTS="${NEMO_RL_SGLANG_PORT_SCAN_ATTEMPTS:-128}"
export NEMO_RL_SGLANG_PORT_LAUNCH_ATTEMPTS="${NEMO_RL_SGLANG_PORT_LAUNCH_ATTEMPTS:-4}"
export NEMO_RL_SGLANG_STARTUP_TIMEOUT="${NEMO_RL_SGLANG_STARTUP_TIMEOUT:-1200}"
export NEMO_RL_SGLANG_STARTUP_TIMEOUT_PER_ATTEMPT="${NEMO_RL_SGLANG_STARTUP_TIMEOUT_PER_ATTEMPT:-}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemo_rl_ray_venvs}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_envs/diffusion_RL_RL_mcore}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/uv_cache}"
export TMPDIR="${SBATCH_TMPDIR}"
export RAY_TMPDIR="${SBATCH_RAY_TMPDIR}"
export RUSTUP_HOME="${RUSTUP_HOME:-}"
export CARGO_HOME="${CARGO_HOME:-}"
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-stable-x86_64-unknown-linux-gnu}"
export PROTOC="${PROTOC:-}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export UV_NO_BINARY_PACKAGE="${UV_NO_BINARY_PACKAGE:-}"
export NEMO_RL_SGLANG_SOURCE="${NEMO_RL_SGLANG_SOURCE}"
export NEMO_RL_SGLANG_KERNEL_SOURCE="${NEMO_RL_SGLANG_KERNEL_SOURCE}"
export NEMO_RL_SGLANG_FLASHINFER_SPECS="${NEMO_RL_SGLANG_FLASHINFER_SPECS}"

if [[ -n "${NEMO_RL_SGLANG_SOURCE}" && -d "${NEMO_RL_SGLANG_SOURCE}" ]]; then
  find "${NEMO_RL_SGLANG_SOURCE}" -type d -name __pycache__ -prune -exec rm -rf {} +
elif [[ -n "${SGLANG_SOURCE_PATH}" && -d "${SGLANG_SOURCE_PATH}" ]]; then
  find "${SGLANG_SOURCE_PATH}" -type d -name __pycache__ -prune -exec rm -rf {} +
fi

srun --kill-on-bad-exit=1 \\
  --container-image="${CONTAINER_IMAGE}" \\
  --container-mounts="${CONTAINER_MOUNTS}" \\
  --container-workdir="${REPO_DIR}" \\
  bash -lc 'cd "\${REPO_DIR}" && tools/nemotron_diffusion/run_grpo_nemotron_ar_megatron_interactive.sh'
SBATCH

echo "CONFIG=${CONFIG}"
