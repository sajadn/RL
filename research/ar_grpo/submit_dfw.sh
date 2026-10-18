#!/usr/bin/env bash
# DFW launcher for the Nemotron AR SingleController experiment.
set -euo pipefail
: "${RUN_NAME:?}" "${ACCOUNT:?}" "${CONFIG:?}" "${NODES:?}"
export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RUNDIR="/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/$RUN_NAME"
export CHECKPOINT_DIR="$RUNDIR/checkpoints" LOG_DIR="$RUNDIR/logs"
export WANDB_RUN_NAME="$RUN_NAME"
export POLICY_PYTHON=/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_driver_envs/diffusion_RL_RL_mb_3rdparty_sglagn_local_fork/bin/python
export NRL_VLLM_PY_EXECUTABLE=/lustre/fsw/portfolios/coreai/users/snorouzi/vllm_runtimes/nemotron_dllm_792ab07/bin/python-compat
export GYM_SOURCE_DIR="${GYM_SOURCE_DIR:-$REPO_DIR/3rdparty/Gym-workspace/Gym}"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/research/ar_grpo:$REPO_DIR/research/block_diffusion:/lustre/fsw/portfolios/coreai/users/snorouzi/ar_sc_overlay_20261001:/lustre/fsw/portfolios/coreai/users/snorouzi/just_grpo_megatron_overlay:$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:$GYM_SOURCE_DIR"
export NRL_GYM_PY_EXECUTABLE="${NRL_GYM_PY_EXECUTABLE:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemo_rl_worker_venvs_mb_3rdparty_vllm_local_fork_fixed/nemo_rl.environments.nemo_gym.NemoGym/bin/python}"
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1 NRL_IGNORE_VERSION_MISMATCH=1
export NRL_REFIT_BUFFER_MEMORY_RATIO=0.02 EXPANDABLE_SEGMENTS=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/lustre/fsw/portfolios/coreai/users/snorouzi/vllm_runtimes/cuda_compat_13.1/lib.real${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home
export HF_HUB_OFFLINE=1
export HF_MODULES_CACHE="$RUNDIR/hf_modules"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemo_rl_megatron_ckpts_ar_sc_native_rope}"
export MEGATRON_CONFIG_LOCK_DIR="$RUNDIR/megatron_config_locks"
export NEMO_GYM_VENV_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/gym_venvs
export VLLM_CACHE_ROOT="$RUNDIR/vllm_cache" VLLM_CONFIG_ROOT="$RUNDIR/vllm_config"
export VLLM_NO_USAGE_STATS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [[ "$NCCL_DEBUG" == INFO || "$NCCL_DEBUG" == TRACE ]]; then
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-ALL}"
  export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-$RUNDIR/nccl/nccl.%h.%p.log}"
  mkdir -p "$(dirname "$NCCL_DEBUG_FILE")"
fi
export NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT=3600 NEMO_SKILLS_SANDBOX_BLAS_THREADS=1
if [[ "${ENABLE_SANDBOX:-1}" == 1 ]]; then
  export SANDBOX_CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_genai/nemo-rl/nemo-skills-sandbox.sqsh
fi
if [[ "${ENABLE_SANDBOX:-1}" == 1 ]]; then
  export SANDBOX_COMMAND=/start-with-nginx.sh
else
  unset SANDBOX_COMMAND SANDBOX_CONTAINER
fi
export SANDBOX_EXTRA_MOUNTS=/lustre/fsw/portfolios/coreai/users/snorouzi/ns_tools_lifecycle_0915/main.py:/app/main.py:ro,/lustre/fsw/portfolios/coreai/users/snorouzi/ns_tools_lifecycle_0915/start-with-nginx-fixed.sh:/start-with-nginx.sh:ro
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh}"
export MOUNTS="/home/snorouzi:/home/snorouzi,/lustre:/lustre,${POLICY_PYTHON%/*}/ray:/opt/nemo_rl_venv/bin/ray"
export BASE_LOG_DIR="$RUNDIR" GPUS_PER_NODE=8 CPUS_PER_WORKER=128
export COMMAND="cd $REPO_DIR && bash research/ar_grpo/driver.sh"
mkdir -p "$RUNDIR/audit"
exec 9>"$RUNDIR/audit/submit.lock"
flock -n 9
if [[ -s "$RUNDIR/audit/jobid" ]]; then cat "$RUNDIR/audit/jobid"; exit 0; fi
cd "$REPO_DIR"
sbatch --parsable --account="$ACCOUNT" --partition=batch --time="${TIME:-04:00:00}" --nodes="$NODES" --gpus-per-node=8 --ntasks-per-node=1 --cpus-per-task=128 --job-name="${JOB_NAME:-$RUN_NAME}" --output="$RUNDIR/slurm-%j.out" ray.sub | tee "$RUNDIR/audit/jobid"
