#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
set -euo pipefail
: "${REPO_DIR:?}" "${CONFIG:?}" "${OUTPUT_DIR:?}"
# ray.sub clears SLURM_* before launching the driver; preserve allocation metadata.
export DOPSD_JOB_ID="${SLURM_JOB_ID}" DOPSD_NUM_NODES="${SLURM_NNODES}"
export POLICY_PYTHON=/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_driver_envs/diffusion_RL_RL_mb_3rdparty_sglagn_local_fork/bin/python
export MODEL=/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/Nemotron-Labs-Diffusion-3B-official
export NRL_VLLM_PY_EXECUTABLE=/lustre/fsw/portfolios/coreai/users/snorouzi/vllm_runtimes/nemotron_dllm_792ab07/bin/python-compat
export NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/just_grpo_upstream_worker_smoke_retry_20260923_0445/hf_cache/nemo_rl
export CONTAINER_IMAGE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh
export CUDA_COMPAT_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/vllm_runtimes/cuda_compat_13.1/lib.real
export POLICY_EXTRA_PYTHONPATH=/lustre/fsw/portfolios/coreai/users/snorouzi/just_grpo_megatron_overlay:/lustre/fsw/portfolios/coreai/users/snorouzi/Megatron-Bridge-asymmetric-rl/src:/lustre/fsw/portfolios/coreai/users/snorouzi/Megatron-Bridge-asymmetric-rl/3rdparty/Megatron-LM
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/research/d_opsd:${REPO_DIR}/research/block_diffusion:${POLICY_EXTRA_PYTHONPATH}"
export EXPANDABLE_SEGMENTS=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 TOKENIZERS_PARALLELISM=false
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export HF_HOME=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_MODULES_CACHE="${OUTPUT_DIR}/hf_modules" MEGATRON_CONFIG_LOCK_DIR="${OUTPUT_DIR}/megatron_config_locks"
export XDG_CACHE_HOME="${OUTPUT_DIR}/cache" MPLCONFIGDIR="${OUTPUT_DIR}/mpl"
export VLLM_CACHE_ROOT="${OUTPUT_DIR}/vllm_cache" VLLM_CONFIG_ROOT="${OUTPUT_DIR}/vllm_config" VLLM_NO_USAGE_STATS=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/d_opsd_${SLURM_JOB_ID}/inductor" TRITON_CACHE_DIR="/tmp/d_opsd_${SLURM_JOB_ID}/triton"
export RAY_TMPDIR="/tmp/d_opsd_${SLURM_JOB_ID}"
export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export WANDB_DIR="${OUTPUT_DIR}/wandb" WANDB_CACHE_DIR="${OUTPUT_DIR}/wandb_cache" WANDB_CONFIG_DIR="${OUTPUT_DIR}/wandb_config"
mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}"
if [[ -z "${WANDB_API_KEY:-}" && -f /home/snorouzi/wandb_api_key.txt ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '\n\r' < /home/snorouzi/wandb_api_key.txt)"
fi
if [[ "${SLURM_NNODES}" == 1 ]]; then
  srun --kill-on-bad-exit=1 \
    --output="${OUTPUT_DIR}/node-%t.log" --error="${OUTPUT_DIR}/node-%t.log" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts=/home/snorouzi:/home/snorouzi,/lustre:/lustre \
    --container-workdir="${REPO_DIR}" bash research/d_opsd/tools/driver.sh
else
  export CONTAINER="${CONTAINER_IMAGE}"
  ray_cli="${POLICY_PYTHON%/*}/ray"
  test -x "${ray_cli}"
  export MOUNTS="/home/snorouzi:/home/snorouzi,/lustre:/lustre,${ray_cli}:/opt/nemo_rl_venv/bin/ray"
  export CONTAINER_CWD="${REPO_DIR}" BASE_LOG_DIR="${OUTPUT_DIR}/ray"
  export GPUS_PER_NODE=8 CPUS_PER_WORKER="${SLURM_CPUS_PER_TASK}"
  printf -v COMMAND 'cd %q && bash research/d_opsd/tools/driver.sh' "${REPO_DIR}"
  export COMMAND
  bash "${REPO_DIR}/ray.sub"
fi
