# ForkGRPO verification (2026-10-19)

- ForkGRPO, TraceGRPO, shared Megatron generation, and logprob aggregation:
  **99 CPU tests passed**.
- The single recipe validates DeepScaleR training, AIME2024 validation, and the
  math environment. The launcher and functional smoke use that recipe.
- Replay tests cover recorded-generation logprob parity, fallback-only loss
  coverage and normalization, terminal context, empty responses, DP slicing,
  missing metadata, and sync/async fallback metadata transport.
- ForkGRPO directly uses `TraceGRPOSchedule`; no separate Fork schedule exists.
- Ruff check/format, shell syntax, and `git diff --check` passed.
- The Megatron split-state optimizer regression test was skipped because the
  CPU environment lacks `megatron.bridge`. An earlier focused check of the
  optimizer/scheduler guard with AdamW passed, including preservation of
  momentum, weight decay, and scheduler state on empty batches.
- Workspace dependency metadata is consistent. Full offline `uv lock --check`
  is unavailable because the host lacks the workspace-required Python 3.13.14;
  CPU tests used the existing Python 3.13.13 environment.
- No GPU training or evaluation was run. The two-GPU execution smoke is
  `tests/functional/deepscaler.sh`. vLLM training integration and paper-level
  learning reproduction remain unverified.

From the repository root, with `TEST_PYTHON` pointing to a provisioned test
interpreter:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=.:research/block_diffusion:research/fork_grpo:research/just_grpo:research/trace_grpo \
uv run --no-project --python "$TEST_PYTHON" python -m pytest \
  -c research/fork_grpo/pyproject.toml \
  research/fork_grpo/tests/unit research/trace_grpo/tests/unit \
  research/just_grpo/tests/unit/test_megatron_generation.py \
  research/just_grpo/tests/unit/test_diffusion_logprob_aggregator.py \
  -q --confcutdir=research
```
