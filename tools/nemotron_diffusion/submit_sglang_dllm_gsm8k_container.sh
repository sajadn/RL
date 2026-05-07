#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Submit a container-backed SGLang dLLM GSM8K evaluation job.
#
# Bare Slurm jobs on DFW may not expose the CUDA toolkit, nvcc, or ninja in the
# same way the SGLang/FlashInfer stack expects. This wrapper runs the evaluator
# inside the known-good eval container and puts the SGLang virtualenv plus CUDA
# toolkit on PATH.
#
# Example:
#   MODEL=/path/to/Nemotron-Diffusion-Exp-Ministral-3B \
#   OUTDIR=/lustre/.../eval_results/sglang_dllm_3b_fastdiffuser_b32_s32 \
#   BLOCK_SIZE=32 MAX_STEPS=32 THRESHOLD=0.9 \
#   tools/nemotron_diffusion/submit_sglang_dllm_gsm8k_container.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/snorouzi/diffusion_RL/RL}"
SGLANG_REPO="${SGLANG_REPO:-/home/snorouzi/code/sglang-nemotron-dllm-lora-eval}"
VENV="${VENV:-/lustre/fsw/portfolios/coreai/users/snorouzi/sglang_nemotron_torch291_cu129_uvpy312_venv}"
EVAL_SCRIPT="${EVAL_SCRIPT:-tools/nemotron_diffusion/run_sglang_linearspec_gsm8k.sh}"

MODEL="${MODEL:?MODEL must point to a Hugging Face checkpoint directory}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
OUTDIR="${OUTDIR:?OUTDIR must point to an output directory}"

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-batch_short}"
TIME="${TIME:-02:00:00}"
GPUS="${GPUS:-1}"
JOB_NAME="${JOB_NAME:-sglang_dllm_gsm8k}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/nvr/users/lwhalen/docker/lxaw_ministral_sft_eval/nemo-rl-lxaw-sft-eval.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/home/snorouzi:/home/snorouzi,/lustre:/lustre}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-${REPO_DIR}}"
CUDA_HOME_IN_CONTAINER="${CUDA_HOME_IN_CONTAINER:-/usr/local/cuda}"

DLLM_ALGORITHM="${DLLM_ALGORITHM:-FastDiffuser}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-64}"
THRESHOLD="${THRESHOLD:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
CONCURRENT="${CONCURRENT:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS-1 2 4 8 16 32 64 128}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
DISABLE_PIECEWISE_CUDA_GRAPH="${DISABLE_PIECEWISE_CUDA_GRAPH:-1}"
INCLUDE_STATS_FILE="${INCLUDE_STATS_FILE:-1}"
PORT="${PORT:-31001}"
HOST="${HOST:-127.0.0.1}"
DTYPE="${DTYPE:-bfloat16}"
LORA_PATH="${LORA_PATH:-}"
EVAL_DRY_RUN="${EVAL_DRY_RUN:-0}"
SUBMIT_DRY_RUN="${SUBMIT_DRY_RUN:-0}"

mkdir -p "${OUTDIR}"
RUNNER="${OUTDIR}/run_inside_container.sh"
SLURM_OUT="${OUTDIR}/slurm-%j.out"

cat >"${RUNNER}" <<EOF2
#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_DIR}"

export SGLANG_REPO="${SGLANG_REPO}"
export VENV="${VENV}"
export PATH="${VENV}/bin:${CUDA_HOME_IN_CONTAINER}/bin:\$PATH"
export CUDA_HOME="${CUDA_HOME_IN_CONTAINER}"
export CUDA_PATH="${CUDA_HOME_IN_CONTAINER}"
export LD_LIBRARY_PATH="${CUDA_HOME_IN_CONTAINER}/lib64:\${LD_LIBRARY_PATH:-}"

printf "CONTAINER_WRAPPER CUDA_HOME=%s VENV=%s SGLANG_REPO=%s\n" "\${CUDA_HOME}" "\${VENV}" "\${SGLANG_REPO}"
printf "CONTAINER_WRAPPER nvcc=%s ninja=%s\n" "\$(command -v nvcc || true)" "\$(command -v ninja || true)"

MODEL="${MODEL}" \\
TOKENIZER="${TOKENIZER}" \\
OUTDIR="${OUTDIR}" \\
SGLANG_REPO="${SGLANG_REPO}" \\
VENV="${VENV}" \\
DLLM_ALGORITHM="${DLLM_ALGORITHM}" \\
BLOCK_SIZE="${BLOCK_SIZE}" \\
MAX_STEPS="${MAX_STEPS}" \\
THRESHOLD="${THRESHOLD}" \\
MAX_TOKENS="${MAX_TOKENS}" \\
NUM_SAMPLES="${NUM_SAMPLES}" \\
CONCURRENT="${CONCURRENT}" \\
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \\
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \\
CUDA_GRAPH_BS="${CUDA_GRAPH_BS}" \\
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH}" \\
DISABLE_PIECEWISE_CUDA_GRAPH="${DISABLE_PIECEWISE_CUDA_GRAPH}" \\
INCLUDE_STATS_FILE="${INCLUDE_STATS_FILE}" \\
PORT="${PORT}" \\
HOST="${HOST}" \\
DTYPE="${DTYPE}" \\
LORA_PATH="${LORA_PATH}" \\
DRY_RUN="${EVAL_DRY_RUN}" \\
bash "${EVAL_SCRIPT}"
EOF2

chmod +x "${RUNNER}"

if [[ "${SUBMIT_DRY_RUN}" == "1" ]]; then
  echo "submit_dry_run=1"
  echo "outdir=${OUTDIR}"
  echo "runner=${RUNNER}"
  printf "sbatch_command:"
  printf " %q" sbatch --parsable -A "${ACCOUNT}" -p "${PARTITION}" \
    --time="${TIME}" --gpus="${GPUS}" --job-name="${JOB_NAME}" \
    --output="${SLURM_OUT}" \
    --wrap="srun --container-image=${CONTAINER_IMAGE} --container-mounts=${CONTAINER_MOUNTS} --container-workdir=${CONTAINER_WORKDIR} bash ${RUNNER}"
  printf "\n"
  exit 0
fi

JOB_ID=$(
  sbatch --parsable \
    -A "${ACCOUNT}" \
    -p "${PARTITION}" \
    --time="${TIME}" \
    --gpus="${GPUS}" \
    --job-name="${JOB_NAME}" \
    --output="${SLURM_OUT}" \
    --wrap="srun --container-image=${CONTAINER_IMAGE} --container-mounts=${CONTAINER_MOUNTS} --container-workdir=${CONTAINER_WORKDIR} bash ${RUNNER}"
)

echo "job_id=${JOB_ID}"
echo "outdir=${OUTDIR}"
echo "runner=${RUNNER}"
echo "slurm_out=${OUTDIR}/slurm-${JOB_ID}.out"

squeue -j "${JOB_ID}" -o "%.18i %.9P %.32j %.8T %.10M %.10l %.6D %R"
