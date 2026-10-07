# Verification

CPU review checks updated on DFW on 2026-10-07.

- d-OPSD and TraceGRPO CPU suites: **94 passed**. This includes a shared-decoder
  generated trajectory, dense block-attention teacher/student forwards, exact
  reverse-KL and gradient oracles, per-block transition averaging, correctness
  filtering, teacher selection, EOS/tail replay, batched gradient equivalence,
  rejected-row compaction, uneven DP acceptance, and worker lifecycle adapters.
- Ruff lint and format checks passed for the d-OPSD package and changed shared
  diffusion modules. `git diff --check` and functional-script shell syntax passed.
- Python syntax and new workspace lock-package metadata validated. No external
  dependencies or existing dependency pins were added or changed.
- Before the batching review, DeepScaleR two-step GPU smoke **passed** on eight GPUs: DFW job `19836659`,
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

## Batching review GPU checks

The first batched check, job `19883640`, hit CUDA OOM with microbatches of two
and activation checkpointing disabled. The DeepScaleR recipe now enables
activation checkpointing for batched full-context student replay.

The checkpointed retry, job `19883792`, **passed** on eight GPUs:
Slurm `COMPLETED`, exit `0:0`, elapsed 00:18:16. Tested source commit:
`3f9754c9e6204064f3536f788d2297498c90aebe`.

- Both optimizer updates used microbatches of two, activation checkpointing,
  all 16 reveal levels, and a 4096-token training context.
- The student generated 128 responses per update: 16 prompts with eight
  candidates each. Correctness filtering was enabled. Accepted trajectories:
  12 and 12.
- Compacted replay executed 16 and 16
  microbatches per DP worker. Uncompacted replay of this 128-response batch
  with microbatch size one would require 256 per worker.
- Losses: 0.2509 and 0.2125. Pre-clipping gradient
  norms: 18.6389 and 16.0345.
- Both AIME2024 modes (`diffusion_conf09` and `ar`) completed at initialization
  and after each update, including policy weight refits.
- The five recipe-validation tests also passed after enabling checkpointing.

The retry used the one-node dual-validation smoke recipe with
`d_opsd.correct_only=true`, `grpo.num_prompts_per_step=16`, and
`grpo.num_generations_per_prompt=8`. It retained eight validation samples per
mode and disabled checkpoint saving. This checks execution and reduced replay
work; it does not establish reward improvement or production throughput.

Retry logs:
`/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/d_opsd_review_batched_20261007/smoke-checkpointed/node-0.log`.

## Teacher-phase performance review

The 2026-10-09 performance follow-up passed **30 d-OPSD CPU tests**, Ruff lint,
format checks, and reviewer follow-up with no remaining actionable issues.
The added checks cover one teacher weight swap per level, multi-microbatch
CPU target caching, equivalence of gradients and optimizer updates to the
original interleaved execution, and restoration before a student failure.

Teacher forwards remain bounded by `train_micro_batch_size=2`. Multiple
teacher microbatches share one reference-weight context per reveal level;
one level of full-vocabulary targets is cached on CPU, then streamed to GPU
one student microbatch at a time. Single-microbatch levels keep targets on GPU.
Replay uses the shared `DenoisingSchedule.return_schedule` iterator.

For 16 levels and M microbatches per level, full-model weight transfers drop
from 48*M to 48 per worker/update. CPU target transfers add overhead and host
memory scales with accepted rows in one level. A wall-time speedup has not
been measured.

Eight-GPU smoke job `19953438` was submitted with 32 prompts, eight candidates
per prompt, correctness filtering, microbatch size two, activation
checkpointing, and both AIME2024 validation modes. It is pending in Slurm as
of this commit; the multi-microbatch CUDA path has not yet been validated.
Run artifacts:
`/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/d_opsd_teacher_phase_20261009/`.

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
