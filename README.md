# GRPO rollout reuse experiment figures

Evidence for one versus two policy updates per rollout, using upstream grpo_math_1B.yaml with a 128-rollout-step cap.

- Sync: one node, eight H100 GPUs, colocated generation.
- Async: one eight-H100 training node plus one eight-H100 generation node, importance sampling correction enabled, maximum trajectory age one rollout step.
- Model: Qwen/Qwen2.5-1.5B. Dataset: OpenMathInstruct-2 train_1M with the upstream validation split. Batch: 32 prompts x 16 responses, global batch 512. Reference KL coefficient: 0.01. Upstream optimizer, scheduler, packing, length limits, validation and checkpointing preserved.
- W&B group: https://wandb.ai/nemo-llm-service/diffusion_rl/groups/grpo-math-1B-upstream-8gpu-20260923

async-reward.png, async-update-time.png, and sync-update-time.png are the provided W&B screenshots. sync-reward.png was reconstructed from the actual synchronous training logs because the supplied second reward screenshot duplicated the asynchronous runs.

The JSON files contain all 128 per-step values parsed from training logs (timings rounded to 0.01 seconds and rewards to four decimals). Aggregate timing includes all recorded steps and their validation/checkpoint overhead. Curves use rollout step, not optimizer step or wall time. These are single-seed runs, not a convergence benchmark.

Experiment code: 8d390577a286ddfc2269d01de1de10919fc7f487 plus the drained async prefix-cache invalidation fix. Recipe snapshot: upstream 37f60e4959a8d2343dc2d6b824240bfb12fed88f. The PR was subsequently applied to newer upstream and unit-tested there.
