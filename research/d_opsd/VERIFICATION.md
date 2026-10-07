# Verification

Checked on DFW on 2026-10-06.

- d-OPSD and TraceGRPO CPU suites: **89 passed**. This includes a shared-decoder
  generated trajectory, dense block-attention teacher/student forwards, exact
  reverse-KL and gradient oracles, per-block transition averaging, correctness
  filtering, teacher selection, EOS/tail replay, and worker lifecycle adapters.
- Ruff lint and format checks passed for the d-OPSD package and changed shared
  diffusion modules. `git diff --check` and functional-script shell syntax passed.
- Python syntax and new workspace lock-package metadata validated. No external
  dependencies or existing dependency pins were added or changed.
- DeepScaleR two-step GPU smoke **passed** on eight GPUs: DFW job `19836659`,
  Slurm `COMPLETED`, exit code `0:0`, elapsed 17m01s. Tested source commit:
  `3672ec95f2b873115f0ef9dbb3f658f3607941f7`.
- Both actual optimizer updates used all 16 reveal levels and the full 4096-token
  training context. Losses were 0.2860 and 0.2712; pre-clipping gradient norms
  were 17.3518 and 18.1236. Both AIME2024 validation modes (`diffusion_conf09`
  and `ar`) ran at initialization and after each update, including weight refits.
- Smoke limits: eight prompts, two candidate samples per prompt, correctness
  filtering disabled, eight validation samples per mode, and checkpoint saving
  disabled. This validates execution and repeated updates; it does not establish
  reasoning improvement, production throughput, or checkpoint/resume behavior.

GPU smoke recipe:
`research/d_opsd/configs/recipes/d_opsd-deepscaler-3b-1n8g-megatron-vllm-dualval-smoke.yaml`.
Logs:
`/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/d_opsd_deepscaler_3b_dualval_20261006/smoke4/node-0.log`.

The GPU run reused the provisioned policy environment, vLLM runtime at commit
`792ab07`, and `Megatron-Bridge-asymmetric-rl` at commit
`5d9b772c7c4787d11af8227dc5bca1b396ef34c1`. No dependency environment was rebuilt.

## Reproduce CPU checks in the provisioned environment

From the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
UV_CACHE_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/uv_cache \
UV_PROJECT_ENVIRONMENT=/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_test_venvs/justgrpo_unit \
PYTHONPATH=research/d_opsd:research/block_diffusion:research/trace_grpo:research/just_grpo:research/just_grpo/tests/unit:research/block_diffusion/tests/unit:. \
uv run --no-sync --frozen python -m pytest \
  -c research/d_opsd/pyproject.toml research/d_opsd/tests/unit \
  research/trace_grpo/tests/unit --confcutdir=research -q -o addopts=""
```

This existing CPU environment is Python 3.13.13; the repository requests
Python 3.13.14. No environment was rebuilt or upgraded for these checks.

## Existing workspace dependency conflict

A full `uv lock --check` was attempted with the available Python 3.13.14
interpreter and reduced uv concurrency. Resolution failed because the existing
linked Automodel checkout is version 0.3.0rc0 and requires
`transformers>=5.3.0,<5.4.0`, while this RL checkout already requires
`transformers>=5.5.0,<=5.12.1`. Both requirements predate d-OPSD. The initial
checkout already had a changed Automodel symlink. Those unrelated dependencies
were left unchanged. The lockfile edit registers only the new local d-OPSD
workspace package with its existing block-diffusion/nemo-rl/pytest dependencies.
