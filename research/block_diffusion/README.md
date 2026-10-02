# Block diffusion runtime

Shared NeMo-RL execution for Block JustGRPO, TraceGRPO, and causal AR GRPO on
Nemotron-Labs-Diffusion. This package owns runtime code; algorithm-specific
schedules and token/level sampling stay in their respective research packages.
It depends on `nemo-rl` and does not import either algorithm package.

- `block_diffusion/block_layout.py` and `denoising_schedule.py`: block geometry,
  schedule interface, and logprob aggregation.
- `block_diffusion/megatron_diffusion_policy.py` and `diffusion_processors.py`:
  shared asymmetric-attention execution, scoring, and optimizer lifecycle.
- `block_diffusion/generation/`: Megatron decoding, reference vLLM workers, and
  `MultiModeValidation`.
- `block_diffusion/training.py`: sync/async GRPO setup, validation lifecycle,
  algorithm callback dispatch, and cleanup.
- `block_diffusion/environments/`: shared Sudoku dataset, reward, and prompts.
  Math experiments use NeMo-RL's native datasets and environments.
- `configs/sudoku6x6_megatron.yaml`: algorithm-independent runtime/data settings.
- `configs/validation/ar_diffusion.yaml`: common named validation overrides.

Both JustGRPO and Trace depend directly on the `block-diffusion` workspace
package. With provisioned interpreters, add `research/block_diffusion` to
`PYTHONPATH` alongside the selected algorithm package and repository root.
Saved configs using `just_grpo.generation.reference_vllm.*` should switch to
`block_diffusion.generation.reference_vllm.*`; the algorithm policy-worker
paths remain `just_grpo.algorithms.block_just_grpo_policy_worker.*` and
`trace_grpo.policy_worker.*`.

## AR and diffusion validation

`policy.generation.vllm_val_dllm_variants` selects named validation engines.
The first mode owns unsuffixed metrics; every mode also reports suffixed
accuracy and length metrics. Each pass reloads the same validation data,
refits the current policy weights, and sleeps the engine before the next pass.
The rollout engines resume after validation. The GRPO controller controls
validation timing and drains async rollouts before the validation callback.
No extra generation nodes are needed for the named validation engines.

The causal AR policy adapter belongs to `../ar_grpo/ar_grpo/ar_policy_worker.py`;
the AR launcher reuses this package for controller setup and validation.
