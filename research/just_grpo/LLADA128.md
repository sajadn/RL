# LLaDA-8B-Instruct JustGRPO on GSM8K and MATH

The initial checkpoint is `GSAI-ML/LLaDA-8B-Instruct`, revision
`08b83a6feb34df1a6011b80c3c00c7563e963b07` (original LLaDA).

## Shared Megatron training and generation

The current recipes use `runtime=megatron_llada` and `backend=megatron`.
Sampling, exact log-probability replay, and training use the same live Megatron
workers and interpreter. Production uses 16 nodes with eight GPUs each,
DP=128 and TP=PP=CP=EP=1. Generation and training run synchronously.

All eight recipes use a 256-token response canvas. Every rollout allocates
the entire masked canvas and reveals one leftmost position per forward at
temperature 1. Attention remains bidirectional, including the masked future.
Replay uses the same conditioning and same-position logits. Generation fills
the entire canvas, including positions after EOS.

The preparation hook extracts the recorded assistant token span and removes
subsequent environment feedback from the context before replay and training.
This preserves the canvas boundary and prevents bidirectional feedback leakage.

Long runs use 64 prompts x 16 completions per update, 125 updates, full-parameter
BF16 training, AdamW LR 5e-6, and zero reference KL. Training microbatch size
and log-probability batch size are both 8 in every long and smoke recipe.
Checkpoints retain the best three validation scores plus the latest recovery point.
The auxiliary variant reuses `policy.diffusion_aux_loss`: advantage-positive
completions, independent Bernoulli masking with a per-completion probability
sampled uniformly from [0.05, 1], masked-token cross-entropy divided by that
probability, weight 0.1, and normalization by all included response tokens.
Prompts remain visible. Both objectives accumulate into one optimizer update.
Rows without auxiliary targets remain in the distributed schedule.

## Eight experiment recipes

Under `configs/recipes/`, each dataset has a long run and a smoke run:

- `just_grpo-gsm8k-llada8b-canvas256-16n8g-megatron-inference-long.yaml`
- `just_grpo-gsm8k-llada8b-canvas256-2n8g-megatron-inference-smoke.yaml`
- `just_grpo-math-llada8b-canvas256-16n8g-megatron-inference-long.yaml`
- `just_grpo-math-llada8b-canvas256-2n8g-megatron-inference-smoke.yaml`

Each has a corresponding `-diffaux.yaml` variant, giving exactly eight YAMLs.
Shared defaults are inherited from these recipes, without separate YAML fragments.
The MATH recipes read `train.jsonl` and `test.jsonl` from `LLADA_MATH_DATA_DIR`.

Smoke runs use two nodes (DP16), 8 prompts x 16 completions, and a global batch
of 128, so each rank trains on a full microbatch of 8. GSM8K smoke runs take two
updates; MATH smoke runs take three. Each recipe has its own default log directory;
`LOG_DIR`, `CHECKPOINT_ROOT`, and `WANDB_RUN_NAME` can override the output locations.

To test a resume, reuse the same smoke recipe and checkpoint directory with a
higher `grpo.max_num_steps` and
`policy.megatron_cfg.scheduler.use_checkpoint_opt_param_scheduler=true`.
Keep the same training topology when resuming optimizer checkpoints.

## Validation and offline evaluation

Both datasets evaluate the same validation questions with four decoders at
temperature 1: one confidence-ranked token per forward, and confidence thresholds
0.9, 0.8, and 0.7, all using 32-token blocks. Production GSM8K validation covers all
1319 test examples, including the final partial batch and batches smaller than the
number of workers.

Standalone evaluation uses an existing recipe with `run_just_grpo.py --eval-only`
and checkpoint/sampling overrides; there are no separate evaluation YAMLs.
Set `policy.pretrained_checkpoint` to the desired checkpoint's `policy/weights`
path with format `megatron_bridge`, and disable checkpoint saving. For confidence
threshold decoding, set `just_grpo.validation_sampling.selection_policy` to
`confidence_threshold`; the threshold is 0.9. Fixed two-token decoding uses
`low_confidence` with `just_grpo.validation_sampling.max_steps=16` (block size 32).
All recipes select their decoders through `just_grpo.validation_modes`, below.
For a single decoder, clear that mapping before overriding `validation_sampling`.

Decoding statistics include per-example forward counts and aggregate
tokens/forward. These count the complete canvas, including tokens after EOS.
Use matching canvas lengths, temperatures, and decoding settings for comparisons.

## Launchers and checks

Run through `run_just_grpo.py` or the supplied `tools/llada_distributed.sbatch`.
The launchers require the provisioned policy interpreter, container, checkpoint,
and separate output directories. They set `EXPANDABLE_SEGMENTS=false` and preload
the selected NCCL library. `tools/llada_production.sbatch` verifies smoke and resume
results before launching. `tools/verify_llada_smoke.py` checks checkpoint shards,
optimizer moments, finite metrics, and initial generation/replay agreement.

The bridge registers original LLaDA with Megatron's LLaDA provider and uses a
public layer-spec callable so checkpoint configurations can be deserialized
without disabling the private-target guard.

Tests cover canvas geometry, future-token leakage, sampled-token/replay parity,
positive-only auxiliary targets and gradients, short validation batches,
validation-mode restoration, decoding statistics, and checkpoint configuration
serialization. Bridge preflight tests require the provisioned runtime.

LLaDA uses `runtime=megatron_llada` with synchronous colocated generation.
Training and generation use the policy interpreter alone.

References: [LLaDA](https://arxiv.org/abs/2502.09992),
[JustGRPO](https://arxiv.org/abs/2601.15165).


## Accuracy versus NFE

All eight recipes inherit four ordered `just_grpo.validation_modes`:

- `fixed1_t1`: temperature 1, block 32, one confidence-ranked token per forward.
- `threshold09_t1`: temperature 1, block 32, confidence threshold 0.9.
- `threshold08_t1`: temperature 1, block 32, confidence threshold 0.8.
- `threshold07_t1`: temperature 1, block 32, confidence threshold 0.7.

Every validation point evaluates all modes on the same questions and logs
`validation/accuracy/<mode>`, `validation/mean_forward_passes/<mode>`, and
`validation/tokens_per_forward/<mode>`. The first mode supplies the primary
accuracy used for best-checkpoint ranking. MATH's former greedy baseline is now
temperature 1 so the four modes compare unmasking policies at the same temperature.

The logger emits `validation/accuracy_vs_nfe` after each completed comparison,
with one labeled point per decoder. W&B also receives the numeric table
`validation/accuracy_vs_nfe_data`; PNGs are saved as
`accuracy_vs_nfe_step<step>.png` in the run directory. The x-axis is mean model
forwards per validation response, including the full canvas after EOS, rather
than cumulative training compute. Fixed one-token decoding uses 256 forwards.

The first mode keeps the original sample/statistics filenames; subsequent modes
append their names. The validation dataloader state and training decoder are
restored even if a pass fails; incomplete comparisons do not emit a plot.
