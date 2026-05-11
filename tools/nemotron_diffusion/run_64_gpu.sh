RUN_NAME=justgrpo_reveal_fastdiffuser_64gpu_kl005_5k_20260506_194652
  RUN_ROOT=/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl
  RUNDIR=${RUN_ROOT}/${RUN_NAME}

  env \
    CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh \
    MOUNTS=/home/snorouzi:/home/snorouzi,/lustre:/lustre \
    BASE_LOG_DIR="${RUNDIR}" \
    GPUS_PER_NODE=8 \
    RAY_LOG_SYNC_FREQUENCY=120 \
    RAY_CLI=/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_envs/diffusion_RL_JustGRPO_mcore/bin/ray \
    RAY_raylet_start_wait_time_s=300 \
    COMMAND="cd /home/snorouzi/diffusion_RL/JustGRPO-worktree && REPO_DIR=/home/snorouzi/diffusion_RL/JustGRPO-worktree RUN_NAME=${RUN_NAME} RUNDIR=${RUNDIR}
  RUN_ROOT=${RUN_ROOT} CLUSTER_NUM_NODES=8 CLUSTER_GPUS_PER_NODE=8 POLICY_MODEL_NAME=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/hub/models--nvidia--Nemotron-
  Diffusion-Exp-Ministral-3B-Instruct/snapshots/d7c52fbc82c29932c18a02478da6e93921daad34 MAX_STEPS=5000 MAX_EPOCHS=1
  PROMPTS_PER_STEP=8 GENERATIONS_PER_PROMPT=8 SAVE_PERIOD=50 VAL_PERIOD=50 VAL_AT_START=false VAL_AT_END=true MAX_VAL_SAMPLES=128 VAL_BATCH_SIZE=128
  TRAIN_GLOBAL_BATCH_SIZE=64 TRAIN_MICRO_BATCH_SIZE=1 LOGPROB_BATCH_SIZE=64 KL_PENALTY=0.05 JUSTGRPO_REVEAL_BATCH_SIZE=64 JUSTGRPO_TRAIN_REVEAL_BATCH_SIZE=1
  MAX_TOTAL_SEQUENCE_LENGTH=640 MAX_NEW_TOKENS=256 HF_CONFIG_SEQ_LENGTH=640 SGLANG_GPUS_PER_SERVER=1 SGLANG_MAX_RUNNING_REQUESTS=1 SGLANG_MEM_FRACTION_STATIC=0.6
  SGLANG_MAX_TOTAL_TOKENS=2048 SGLANG_DISABLE_CUDA_GRAPH=true SGLANG_DLLM_ALGORITHM=FastDiffuser SGLANG_DLLM_ALGORITHM_CONFIG=/home/snorouzi/diffusion_RL/JustGRPO-
  worktree/tools/nemotron_diffusion/justgrpo_leftmost_dllm.yaml MEGATRON_TP_SIZE=1 MEGATRON_SEQUENCE_PARALLEL=false MEGATRON_CONVERTER_TYPE=MinistralDiffEncoderModel
  MEGATRON_FORCE_RECONVERT=false MEGATRON_PATCH_DIR=/home/snorouzi/code/Megatron-Bridge/src:/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/modules
  NEMOTRON_DIFFUSION_DISABLE_COMPILED_FLEX_ATTENTION=0 WANDB_ENABLED=true WANDB_PROJECT=diffusion_rl WANDB_RUN_NAME=${RUN_NAME} NEMO_RL_SGLANG_STARTUP_TIMEOUT=${NEMO_RL_SGLANG_STARTUP_TIMEOUT:-1200} bash
  tools/nemotron_diffusion/run_grpo_nemotron_justgrpo_sglang_interactive.sh" \
    sbatch --nodes=8 \
      --account=nvr_lpr_llm \
      --partition=batch \
      --time=04:00:00 \
      --gres=gpu:8 \
      --job-name=jg64_kl005_5k \
      --output=${RUNDIR}/slurm-%j.out \
      ray.sub
