---
name: deepscaler-aime-experiment
description: Reproduce or extend the deepscaleR training + AIME 24/25/26 validation experiment for Nemotron diffusion models - dataset conversion to NeMo-Gym format, per-dataset validation metrics via agent aliases, sequence-length choice, TPF instrumentation, and the launch procedure. Use when setting up a math RL run on deepscaleR or AIME, when validation must report several datasets separately, when a run needs to survive session teardown, or when reusing this experiment's config as a template.
---

# deepscaleR + AIME experiment

Hybrid AR+diffusion GRPO on deepscaleR, validated every 20 steps on AIME
24/25/26 and a deepscaleR holdout, each reported separately, with diffusion
TPF logged alongside accuracy.

Reference run: `/linnanw/runs/diffusion_rl/deepscaler_aime_hybrid_v2`.
Config: `examples/configs/nemotron_labs_diffusion_3b_vllm_hybrid_ar_diffusion_megatron_deepscaler_aime.yaml`.
Launcher: `/linnanw/runs/launch_deepscaler_aime_hybrid_v2.sh`.

Decode follows [diffusion-eval-decode](../diffusion-eval-decode/SKILL.md) —
confidence-threshold at 0.9, never leftmost.

## 1. Data

Neither source ships in Gym format. Convert with
`_scratch/build_deepscaler_aime_gym.py` → `/linnanw/justGRPO/dataset/deepscaler_aime_gym/`:

| file | n | purpose |
|---|---|---|
| `train_deepscaler.jsonl` | 38,284 | training (95% split, seed 42) |
| `val_intraining.jsonl` | 340 | **in-training validation** (90 AIME + 250 deepscaleR) |
| `val_all.jsonl` | 2,105 | full offline eval (90 AIME + full 5% holdout) |
| `val_aime{24,25,26}.jsonl` | 30 each | per-year offline eval |
| `val_deepscaler.jsonl` | 2,015 | full 5% holdout |

Source quirks the converter handles:
- **aime24 has no `answer` column** — it must be extracted from `solution`,
  which is just `\boxed{N}` (30/30 extractable). aime25/26 have `answer`.
- deepscaleR `solution` is **empty in 81.7% of rows**; non-empty ones average
  412 tokens. Do not treat it as a CoT-length reference for the whole set.
- 42% of deepscaleR answers are LaTeX (`-\frac{2}{3}`), not integers.
  `math_with_judge` runs with `should_use_judge: false` → local `math_verify`,
  which handles them.
- 10 train prompts exceed 1024 tokens and are dropped.

Records are `math_with_judge_simple_agent` shape with a
`"\n\nMake sure to use \\boxed{} for your answer."` suffix.

## 2. Per-dataset validation metrics

Gym keys rollout metrics by `agent_ref["name"]`, so a validation file whose rows
all name one agent collapses to a single `accuracy`. Give each subset **its own
agent name**, all bound to the same `math_with_judge` resources server so the
verifier is identical and only the label differs:

`resources_servers/math_with_judge/configs/math_with_judge_eval_aliases.yaml`
defines `aime24_agent`, `aime25_agent`, `aime26_agent`, `deepscaler_val_agent`.
Load it alongside `math_with_judge.yaml`.

`grpo.py` accumulates `agent_rewards` across validation batches and emits
`{agent}/accuracy` + `{agent}/count`, printed as a "Per-dataset accuracy" block.

## 3. Sequence length: 2048

Not because prompts are long — deepscaleR is p50 91, AIME p50 117-151. Two
measured reasons:

- **AIME accuracy saturates near 2048.** Base-model sweep at 8192 max tokens
  (`_scratch/aime_length_sweep.py`): 1024 retains only 46-85% of achievable
  accuracy, 2048 retains 92-100%.
- **1024 trains a short-CoT prior.** The qamathcode runs at 1024 showed
  generation length collapsing 463 → 232 tokens as the model learned to fit the
  cap; a longer eval length afterwards cannot undo that.

Thinking mode was swept too and makes **no** difference (0.1139 off vs 0.1167 on,
noise at n=360) — leave it off, matching all prior runs.

Cost: the `[noisy | clean]` layout is ~2x the rollout length, so ~4096-token
training sequences, ~60-67 s/step vs ~49 s at 1024.
`activation_checkpointing: true` is mandatory. If it OOMs, drop
`train_micro_batch_size` 4 → 2 before touching `context_parallel_size`; global
batch is held at 128 by grad accumulation either way.

## 4. TPF in validation

`validate()` resets the diffusion counters, runs the pass, then logs
`diffusion_tpf`, `diffusion_nfe`, `diffusion_committed_tokens`:

```
• Diffusion TPF: 2.9601 (257062 tokens / 86841 denoising forwards)
```

Plumbed as: GPU counters in `MaskedDiffusionSampler` (no host sync on the hot
path) → `get_diffusion_tpf_stats()` module accessor → `VllmInternalWorkerExtension`
→ `get_diffusion_tpf_stats_async` → `VllmGeneration.get_diffusion_tpf_stats`.
**max** across tensor-parallel ranks (they replicate; summing inflates nfe by the
TP factor), **sum** across data-parallel workers (disjoint requests). Returns
zeros on the `ar_mode` rollout group.

Do **not** infer TPF from validation wall-clock. That estimate gave 2.13 against
a measured 2.96 (28% low) because forwards committing more tokens are not free.

## 5. Launch

```bash
cd /linnanw/runs
RUN_NAME=<name> STEPS=400 VAL_PERIOD=20 SAVE_PERIOD=20 KEEP_TOP_K=null MBS=4 \
  setsid nohup ./launch_deepscaler_aime_hybrid_v2.sh > /dev/null 2>&1 < /dev/null &
```

**`setsid` is required.** Launching through an agent-managed background shell
ties the job to that session's process group and it dies on session teardown —
this killed the run once at step 61, mid-rollout, with no error in the log.

`KEEP_TOP_K=null` keeps all 20 checkpoints (~43 GB each, ~860 GB). The inherited
default `keep_top_k: 2` silently reduced a previous run's 16 checkpoints to 3 and
made an early-step TPF curve impossible.

Resume is automatic from the latest checkpoint in `checkpoint_dir`.

## 6. Sanity checks

Step-0 validation should reproduce (base Nemotron-Labs-Diffusion-3B):

```
aime24_agent 0.1667   aime25_agent 0.1333   aime26_agent 0.0667
deepscaler_val_agent 0.4200        Diffusion TPF ~2.96
```

Verify before trusting a run: `'selection_policy': 'confidence_threshold'` in the
resolved config, `'max_model_len': 2048`, and the loss decomposition
(`Loss == PG + 0.1*CE`).

## 7. Greedy validation cannot resolve this task

Measured on this experiment's 250-problem deepscaleR holdout across six
checkpoints (steps 0-100):

```
always correct    54  (21.6%)
always wrong      84  (33.6%)
UNSTABLE         112  (44.8%)   <- flip between consecutive checkpoints
```

~25 problems are gained and ~25 lost every 20 steps; the net is +/-5. With 112
problems flipping ~50/50 the count's standard deviation is
`sqrt(112*0.25) ~= 5.3` problems = **+/-2.1 points, so +/-4.2 at 2 sigma**. The
entire observed range over six validations was 2.8 points -- every reading was
inside the noise band, while training reward rose 25%.

**Do not conclude "no improvement" from a flat greedy curve here.** Use
`_scratch/sampled_val_eval.py` (avg@k / pass@k / all@k at k=8) on the retained
checkpoints. Its verifier is a byte-faithful copy of Gym's `math_with_judge`
(same `math_metric` extraction configs, same `\boxed{}` gold wrapping, same
delimiter stripping) and was cross-checked at **100% agreement on 128 recorded
rollouts**, so its rewards mean the same thing as training rewards.

Read avg@k against pass@k: rising avg@k with flat pass@k means the model is
consolidating problems it could already solve (5/8 -> 8/8), which greedy
validation cannot see. On deepscaleR that consolidation was 40% of the gain
(+0.077 all-8/8 vs +0.106 pass@8) versus 15% on Gym qamathcode
(+0.020 vs +0.113) -- the direct consequence of starting at 0.42 instead of 0.09.

When cross-checking rewards against a rollout dump, note that **`idx` is the row
position within the step** (0..127 for 16 prompts x 8 generations), NOT the
dataset row. Match on the decoded prompt text instead.

## 8. Gotchas

- **`max_new_tokens` is inert on the Gym path.** `run_async_nemo_gym_rollout`
  ignores it and delegates truncation to vLLM's `max_model_len`. The config says
  256; observed responses are ~736. The only real cap is the joint 2048.
- **`data.max_input_seq_length` is a no-op.** `nemo_gym_data_processor` never
  tokenizes the prompt — it emits a placeholder `message_log` with `length: 0`.
- **Validation is always a single batch.** `run_grpo_nemo_gym.py:186-187`
  overwrites `max_val_samples` and `val_batch_size` with `len(val_dataset)`, so
  the whole split is issued as one concurrent burst. This is why the in-training
  split is subsampled to 340 rather than the full 2,105 (~8 min/validation).
- **AIME at n=30 is noise-dominated.** One problem = 3.33 points. Read the
  90-problem AIME average, not individual years, and expect no readable AIME
  trend before ~step 200. deepscaleR at n=250 is the trustworthy signal.
- Gym `config_paths` must be **absolute** — relative paths resolve into Ray's
  working_dir copy where `.venv` is excluded and every server races to build one.
