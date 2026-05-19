#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-examples/configs/grpo_math_nemotron_diffusion_3b_instruct_vllm_ar_sglang.yaml}"
export RUN_NAME="${RUN_NAME:-ministral3_ar}"

if [[ "${1:-}" == "--sbatch" || "${SUBMIT:-0}" == "1" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"
  RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl}"
  RUNDIR="${RUNDIR:-${RUN_ROOT}/${RUN_NAME}}"
  ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
  PARTITION="${PARTITION:-batch}"
  TIME="${TIME:-04:00:00}"
  NODES="${NODES:-1}"
  GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
  CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
  JOB_NAME="${JOB_NAME:-${RUN_NAME}}"
  CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh}"
  CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${HOME}:/home/snorouzi,/lustre:/lustre}"
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
export CONFIG="${CONFIG}"
export RUN_NAME="${RUN_NAME}"
export RUN_ROOT="${RUN_ROOT}"
export RUNDIR="${RUNDIR}"
export EXTRA_GRPO_OVERRIDES="${EXTRA_GRPO_OVERRIDES:-}"

srun --kill-on-bad-exit=1 \\
  --container-image="${CONTAINER_IMAGE}" \\
  --container-mounts="${CONTAINER_MOUNTS}" \\
  --container-workdir="${REPO_DIR}" \\
  bash -lc 'cd "\${REPO_DIR}" && bash tools/nemotron_diffusion/run_grpo_nemotron_ar_megatron_interactive.sh'
SBATCH
  echo "CONFIG=${CONFIG}"
  echo "RUNDIR=${RUNDIR}"
  exit 0
fi

bash tools/nemotron_diffusion/run_grpo_nemotron_ar_megatron_interactive.sh
