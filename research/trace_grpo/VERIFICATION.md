# TraceGRPO verification

## TraceGRPO port (2026-09-28)

Based on the diffusion execution separation at `f14261e58`. Reference algorithm:
DFW `diffusion_RL/RL` at `2fad6d2cc8c4ec13bfaf5bf90bd64404268c6c81`.
The original port was smoke-tested at `347e23d8b`; these results predate the
subsequent package separation and asymmetric-attention integration.

- Research CPU suite: **152 passed, 3 optional-dependency skips**. Covers recorded
  score parity for leftmost/confidence decoding, 1/2/6 sampled views, identical
  scoring/training contexts, per-row draw stability under sharding, sum/mean
  weighting, dense-ranking gaps, terminal-block context, metadata transport
  through real sync/async rollout functions, and all eight recipe configurations.
- The new core sequence-diagnostic test passed with the actual function isolated
  from unrelated optional audio imports. Full core-test collection in the CPU
  environment was not run: the GRPO module's optional audio dataset dependencies
  are absent. GPU smokes exercised the complete GRPO controller imports and path.
- Ruff check/format and `git diff --check` passed.
- **Sync, num_level_samples=1:** Slurm **19488818**, COMPLETED `0:0`, 5m32s.
  Three optimizer updates; final generation KL `2.28e-9` (roundoff), reference KL
  `0.000111`. [W&B](https://wandb.ai/nemo-llm-service/diffusion_rl/runs/a8u5y31b).
- **Async, num_level_samples=2:** Slurm **19488819**, COMPLETED `0:0`, 6m24s.
  Three optimizer updates; final generation KL `0.000202`, reference KL
  `0.000132`. [W&B](https://wandb.ai/nemo-llm-service/diffusion_rl/runs/o1g5rxuq).

Both runs used the official 3B checkpoint, one node / two GPUs, confidence-threshold
Megatron decoding, retained terminal blocks, and initial/final Sudoku validation.
Both passed `../just_grpo/tests/functional/check_metrics.py` (finite losses/gradients/KL,
nonzero training gradient, positive LR/token counts, three updates and validation).
These are execution smokes with two validation puzzles, not learning-quality or
scaling evaluations. Each launch started an explicitly addressed Ray head.

Artifacts, exact submission scripts, configs, logs, and checker output:
`/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/trace_grpo_ports_smoke_20260928/{sync-k1,async-k2}`.

The reference-vLLM adapter and its reveal/response-length metadata have CPU contract
coverage. Trace generation/training through vLLM was **not** GPU-tested in this port.

## Package separation (2026-09-28)

The original package separation was validated in clean detached checkouts,
without the pending Step Distillation changes in the training worktree.
The results below describe that historical refactor.

### Validation

- Entropy-budget commit: **49 passed** (Trace and configuration suites).
- Shared CLI commit: **25 passed** (configuration and CLI suite).
- Package split: **160 passed** (combined Trace and JustGRPO CPU suites).
- The GPU adapter test module was excluded because Transformer Engine cannot
  load cuDNN on the login node. No GPU training job was submitted for this refactor.
- Trace wheel build and package/lock metadata checks passed during implementation.
- Shell syntax and `git diff --check` passed.

The final suite used the provisioned policy interpreter plus ephemeral pytest,
with both research packages, the repository, and the compatibility overlay on
PYTHONPATH:

```bash
uv run --no-cache --no-project --with pytest --python "$POLICY_PYTHON" \
  python -m pytest -c research/trace_grpo/pyproject.toml \
  research/trace_grpo/tests/unit research/just_grpo/tests/unit \
  --ignore=research/just_grpo/tests/unit/test_megatron_adapter.py \
  -q --confcutdir=research
```

## Shared multi-mode validation

The [shared validation record](../just_grpo/VERIFICATION.md#multi-mode-ardiffusion-validation-2026-09-29)
covers the sync/async AR and diffusion validation matrix, including Trace GPU smokes.
