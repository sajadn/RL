RUN_NAME=grpo_nemotron_ar_megatron_sglang_3b_instruct_5k \
  CONFIG=examples/configs/grpo_math_nemotron_diffusion_3b_instruct_ar_megatron_sglang.yaml \
  POLICY_MODEL_NAME=/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/Nemotron-Diffusion-Exp-Ministral-3B-Instruct-vllm-ar \
  MAX_STEPS=5000 \
  MAX_EPOCHS=1 \
  tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh
