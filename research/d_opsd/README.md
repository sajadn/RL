# d-OPSD

On-policy self-distillation for Nemotron Labs Diffusion, based on
[Learning from the Self-future: On-policy Self-distillation for dLLMs](https://arxiv.org/abs/2606.18195)
and its [reference implementation](https://github.com/xingzhejun/d-opsd-code).

This dedicated research package owns the algorithm, teacher-conditioned replay,
loss, worker adapter, entrypoint, configs, and tests. It uses `block_diffusion`
for response canvases, asymmetric attention, recorded reveal-state extraction,
generation, validation, and the native Megatron optimizer/checkpoint lifecycle.

## Method

1. Generate fresh student responses with recorded reveal steps and retain the
   terminal block. Generate eight candidates per prompt in a batch by default.
2. Keep the first candidate reaching `reward_threshold` for each prompt.
   All-failed groups contribute no loss. `correct_only: false` trains on all
   valid responses as an ablation. Rewards are verification outcomes; GRPO
   advantages do not enter the distillation loss.
3. Reconstruct every pre-transition masked state. The frozen initial checkpoint
   receives extra visible tokens from the final student-generated response.
   Visibility is nested: the teacher always has the student's context.
4. Select k still-masked positions per block by teacher probability confidence,
   with k equal to the recorded student's commit count for that transition.
   `selection_source: student` instead supervises the actual recorded commits.
5. Minimize full-vocabulary `KL(student || teacher)`, averaging over selected
   positions within each transition, transitions within each response, and
   accepted responses. Pointwise clipping caps individual vocabulary-entry
   contributions at 0.05; it does not clamp the summed token KL. Negative
   vocabulary contributions remain untouched, matching the reference code.

The teacher is the initial pretrained model, loaded through the worker's frozen
reference state. Student checkpoint resumes do not replace teacher weights.
Teacher targets are detached. Responses with zero accepted candidates globally
skip the optimizer and scheduler update, preserving Adam momentum and weights.

## Block-model adaptation

The paper uses LLaDA; this implementation uses Nemotron's shared block-attention
runtime. Each response block sees the prompt, previous clean blocks, and its own
noisy block. Future conditioning is therefore confined to the current block;
it does not expose later clean blocks through attention. Independent block
states can share one forward while retaining per-block transition averaging.

Teacher reveals are sampled reproducibly from committed, through-stop response
positions that remain masked in the student. The reveal count is capped to
leave at least k eligible masked positions for teacher selection. The final
transition may consequently receive no extra reveals. Committed terminal-block
tail tokens remain replay context; post-stop positions do not receive loss.

Batched pass@8 selects the first accepted candidate after all eight are generated;
it does not implement the paper's sequential early-stopping rollout optimization.
All actual reveal levels are supervised, including variable commit counts.
The fixed maximum-level loop keeps DP workers in lockstep; absent levels have
zero loss. Only one level/microbatch of teacher vocabulary logits is retained.

## Run

Initial support: synchronous, single-turn, unpacked BF16 full-parameter Megatron,
DP with TP=PP=CP=EP=1. Both in-process Megatron and the research reference vLLM
rollout worker are supported. LoRA, async rollouts, and the data plane are
rejected by config validation. The upstream GRPO controller supplies generation,
verification, logging, and checkpointing; the research worker replaces its loss
with d-OPSD and disables policy-ratio/reference-KL scoring.

From the repository root, in a provisioned environment:

```bash
uv run --all-packages --extra mcore python research/d_opsd/run_d_opsd.py \
  policy.model_name=/path/to/Nemotron-Labs-Diffusion-3B \
  logger.log_dir=/path/to/fresh/results
```

The default config inherits the shared Sudoku6x6 data/runtime settings and uses
`reward_threshold: 0.25`, matching the paper's Sudoku filter. Binary math
verification should use `reward_threshold: 1.0` with the native math dataset and
environment settings. No gold response tokens are used as teacher context.

For a small one-node functional run:

```bash
uv run --all-packages --extra mcore python research/d_opsd/run_d_opsd.py \
  --config research/d_opsd/configs/recipes/d_opsd-sudoku6x6-1n8g-megatron-smoke.yaml \
  policy.model_name=/path/to/Nemotron-Labs-Diffusion-3B
```

For a provisioned interpreter, put the repo root, `research/d_opsd`, and
`research/block_diffusion` on `PYTHONPATH`, including in remote worker environments.

## DeepScaleR with AR and diffusion validation

The recipe
`configs/recipes/d_opsd-deepscaler-3b-4n8g-megatron-vllm-dualval-long.yaml`
trains synchronously on DeepScaleR with 128 prompts and eight candidates per
prompt, LR 1e-6, and a 4096-token training budget. Math correctness filtering
uses reward threshold 1.0. AIME2024 validation repeats each problem 16 times,
using the shared `diffusion_conf09` and `ar` modes at the start, every ten
steps, and at the end. Diffusion validation owns the primary accuracy metric.

The DFW launcher is `tools/run.sbatch`. Supply `REPO_DIR`, `CONFIG`,
`OUTPUT_DIR`, `CHECKPOINT_ROOT`, and `WANDB_RUN_NAME` when submitting with
four nodes and eight GPUs per node. It reuses the provisioned policy interpreter,
asymmetric-attention Bridge, and diffusion vLLM runtime used by the existing
research runs; it does not rebuild their environments. Checkpoints save every
five steps, retaining the best two and latest two. The checkpoint deadline is
3h30m for a four-hour Slurm allocation.

The corresponding one-node `dualval-smoke.yaml` recipe runs two full-context
updates and both validation modes with eight validation samples and two toy
candidates per prompt. Production retains eight candidates per prompt. It disables
correctness filtering and checkpoint saving to exercise student backward even
when the sampled answers fail verification.

## Tests

```bash
uv run --package d-opsd --group test python -m pytest \
  -c research/d_opsd/pyproject.toml research/d_opsd/tests/unit --confcutdir=research
```

Tests cover reward filtering, nested reproducible masks, EOS/tail replay,
teacher confidence selection, exact step averaging and gradients, detached
teacher targets, pointwise clipping, empty losses, and config rejection.
A generated-trajectory CPU test exercises the shared decoder, teacher conditioning,
and a student update with an independent dense block-attention oracle. CPU worker
adapter tests cover frozen weights across updates, rejected batches, and failures.
The functional script is `tests/functional/d_opsd-smoke.sh`; run it inside a
provisioned GPU allocation. It uses a fresh two-step output directory and disables
external experiment tracking. Its `correct_only: false` ablation ensures backward
is exercised even if the tiny run produces no accepted answer. No cluster job is submitted by the script.

See [VERIFICATION.md](VERIFICATION.md) for executed checks and remaining GPU/environment validation.
