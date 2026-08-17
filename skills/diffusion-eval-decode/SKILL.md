---
name: diffusion-eval-decode
description: Standard for how Nemotron diffusion (dLLM) generation must be decoded in ANY evaluation or validation - use confidence-threshold reveal, never leftmost. Use when writing or reviewing a config with selection_policy / vllm_val_dllm_overrides / diffusion_config, when adding an eval or validation harness for a diffusion checkpoint, when choosing generate_dllm remasking/threshold arguments, or when a TPF / tokens-per-forward / decode-efficiency number needs to be reported or explained.
---

# Diffusion evaluation decode standard

**Rule: every evaluation and validation decodes with confidence-threshold reveal at
`threshold = 0.9`. Never `leftmost`. Never the default fixed schedule.**

This applies to in-training validation, offline checkpoint evaluation, and any
ad-hoc harness. Rollouts are exempt — they follow whatever the algorithm
requires (`ar_mode` for hybrid, leftmost for BlockJustGRPO recompute parity).

## Why

Both rejected policies commit exactly **one token per forward pass**, which
makes the decode strictly more expensive than autoregressive for identical
output and pins TPF to 1.0 *by construction* — so no checkpoint can ever look
more efficient than any other, no matter what it learned.

`leftmost` is obvious: one token per step, left to right. The trap is that
`low_confidence` is **equally degenerate under the defaults**. It takes its
commit count from the even transfer schedule, not from confidence:

```python
# nemotron_dllm.py
num_masks       = mask_index.sum(dim=-1)
remaining_steps = (max_denoising_steps - cur_step).clamp(min=1)
k = -(-num_masks // remaining_steps)     # ceil div -- tokens committed this step
```

`max_denoising_steps` falls back to `canvas_length`, and `canvas_length` falls
back to `hf_config.canvas_length` → **32** (the Nemotron model config has no
such key). With steps == canvas, `k == 1` at every step.

So `low_confidence` changes *which* positions commit, not *how many*. Switching
`leftmost` → `low_confidence` fixes ordering and fixes nothing about
efficiency. Confidence-threshold reveal drops the schedule entirely and commits
however many positions clear the bar, making tokens-per-forward adaptive: a
model that grows more confident decodes faster, and that shows up in the number.

Measured on deepscaleR + AIME validation (340 samples, Nemotron-Labs-Diffusion-3B):
**45.06 s at one token/forward → 21.14 s at threshold 0.9, a 2.13x speedup with
accuracy unchanged.**

## vLLM (in-training validation)

```yaml
policy:
  generation:
    vllm_val_dllm_overrides:
      vllm_kwargs:
        hf_overrides:
          ar_mode: false            # explicit false; overrides are deep-merged
        diffusion_config:
          temperature: 0.0
          selection_policy: "confidence_threshold"
          confidence_threshold: 0.9
```

`confidence_threshold` was **added to this vLLM tree** (`version2/vllm`) for
this purpose — upstream/newer checkouts ship it, the pinned one did not. If
`Unknown diffusion selection_policy` is raised, the tree predates the addition.
Selection-logic tests: `_scratch/test_confidence_threshold.py`.

## Megatron-Bridge (offline checkpoint evaluation)

```
--threshold 0.9 --no-neg-entropy --remasking low_confidence
```

`--threshold` is what engages the confidence gate; `remasking` only orders the
candidates. **`--threshold` is mandatory** — with `threshold=None`,
`get_num_transfer_tokens` fixes the per-step count in advance and TPF collapses
to the constant `block_length / steps_per_block`, identical for every
checkpoint. Harness: `_scratch/tpf_eval_dllm.py`, which flags this degenerate
value explicitly.

## Reporting TPF

```
TPF = sum(committed tokens) / sum(nfe)      # dataset-wide, token-weighted
```

Not the mean of per-prompt ratios. `nfe` counts denoising forwards only; the
per-block causal KV-update forward is excluded. Always state the decode config
next to the number — a TPF without its threshold and step budget is unreadable.

## Gotchas

- **`low_confidence` reads backwards.** It is LLaDA remasking: re-mask the
  *least* confident, i.e. commit the *most* confident. It is not a "low
  confidence" setting, and it is the model's native default.
- **`block_size` does NOT feed `canvas_length`.** Training configs set
  `hf_config_overrides.block_size: 16`; the decode canvas is still 32 unless
  `canvas_length` is set explicitly. Pin both when claiming an efficiency number.
- **Verify it actually took effect.** Check the resolved config in the run log
  for `'selection_policy': 'confidence_threshold'` before trusting any decode
  number — validation wall-clock is the fastest sanity check, since a
  degenerate decode is roughly 2x slower.
- **Do not compare across policies.** A curve produced under leftmost is not
  comparable to one under threshold reveal. Runs predating this standard
  (qamathcode hybrid, ce0, pg0) used the degenerate schedule; note it when
  putting their numbers in the same table.
