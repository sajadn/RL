# TraceGRPO

`run_trace_grpo.py` runs recorded-trajectory GRPO through the same upstream
controller and shared Megatron diffusion worker. The Trace algorithm and schedule
live in `trace_grpo/algorithm.py`; its policy worker supplies only
configuration and schedule construction.

Use `trace_grpo.schedule.num_level_samples=1` for one sampled-level training
forward/backward per microbatch. Previous-policy and reference scoring each still
require their own forwards. For `k > 1`, sample up to `k` distinct recorded levels
per response, accumulate their gradients, and perform one optimizer update. Rows
with fewer levels contribute zero-loss views for the remaining draws, keeping
all DP ranks in lockstep. Draws are made once per optimizer step using per-row
seeds and are carried unchanged through sharding, previous/reference scoring,
and training.

`sampled_level_reduction: sum` uses the reference implementation's depth/k
weights; `mean` uses 1/k. The token loss is normalized by the weighted selected
token count. Replay uses per-response dense ranks of recorded block-relative
commit steps: blocks advance together, with earlier blocks supplied by the clean
half of the shared diffusion attention layout.

Generation must return reveal steps and retain the full terminal block. Committed
post-stop tokens remain available as replay context; uncommitted tail positions
remain MASK. Loss and logprob diagnostics include only selected tokens through
the first stop token, while environment text excludes the retained tail.
Whole-response sequence TIS and `seq_logprob_error_threshold` are rejected for
sampled Trace scoring. The driver data-plane path and multi-turn rollouts remain unsupported.
Checkpoint saving and continuation use the [shared upstream path](../just_grpo/README.md#checkpointing-and-continuation).

Recipes cover sync/async Megatron generation and the reference vLLM fork.
Megatron supports `leftmost` and `confidence_threshold`; the reference vLLM fork
also supports `random`, `entropy`, `entropy_budget`, and `low_confidence`. Supply its interpreter
with `--generation-python`. Stock upstream vLLM is not supported by these recipes.
For example, from the repository root in a provisioned environment:

```bash
uv run --all-packages --extra mcore python research/trace_grpo/run_trace_grpo.py \
  --model /path/to/Nemotron-Labs-Diffusion-3B \
  --output-dir /path/to/fresh/results \
  trace_grpo.schedule.num_level_samples=1
```

Reference implementation: `diffusion_RL/RL/nemo_rl/algorithms/trace_grpo_logprobs.py`
and `nemo_rl/models/policy/workers/trace_grpo_megatron_policy_worker.py` at
`2fad6d2cc8c4ec13bfaf5bf90bd64404268c6c81` (read from the DFW checkout).

## Package layout

- `trace_grpo/config.py`: Trace experiment parsing and validation.
- `trace_grpo/train.py`: Trace batch-preparation callback and shared-driver entrypoint.
- `trace_grpo/algorithm.py`: Trace configuration, level sampling, batch preparation, and replay schedule.
- `trace_grpo/policy_worker.py`: Trace schedule selection for the shared diffusion worker.
- `configs/recipes/`: Trace Megatron and vLLM recipes, synchronous and asynchronous.
- `tests/unit/`: Trace replay and configuration tests.

Shared diffusion execution stays in `../just_grpo/just_grpo/diffusion`; Trace imports
it through `just_grpo.diffusion`. The Python package depends on the `just-grpo`
workspace package. No diffusion execution code is duplicated.

The worker FQN is `trace_grpo.policy_worker.TraceGRPOPolicyWorker`.
Existing saved run configs need that FQN updated before reuse.
For provisioned interpreters, include both `research/just_grpo` and
`research/trace_grpo` on `PYTHONPATH`; the shared DFW sbatch script does this.

Run the Trace tests from the repository root with both research packages available:

```bash
uv run --package trace-grpo --group test python -m pytest \
  -c research/trace_grpo/pyproject.toml research/trace_grpo/tests/unit \
  --confcutdir=research
```

## AR and diffusion validation

Reference-vLLM recipes ending in `-dualval-long.yaml` evaluate each checkpoint in
both confidence-threshold diffusion and causal AR modes. Sync and async recipes
use `policy.generation.vllm_val_dllm_variants`; the reusable override is
`../just_grpo/configs/validation/ar_diffusion.yaml`. See the [shared validation behavior](../just_grpo/README.md#ar-and-diffusion-validation)
for metric names and engine lifecycle, and [VERIFICATION.md](VERIFICATION.md)
for recorded checks.
