# AR GRPO with multi-mode validation

`ar_grpo/policy.py` contains the causal Nemotron policy adapter shared by the
SingleController launcher (`run_ar_grpo.py`) and the legacy GRPO launcher
(`../just_grpo/run_ar_grpo.py`). Both use
`ar_grpo.policy.ARModeForMultiModeMegatronPolicy`.

The adapter selects causal attention for whole-batch training, SingleController
microbatch training, logprob calculation, and top-k scoring. It disables KV
caching and restores the previous attention settings after each call.

The legacy launcher uses the DeepScaleR/AIME2024 recipe under
`../just_grpo/configs/recipes/`. Its shared generation and multi-mode validation
remain in the `block-diffusion` package; the SingleController launcher keeps its
existing controller behavior.

For provisioned interpreters, include `research/ar_grpo`, `research/block_diffusion`,
and the repository root in `PYTHONPATH`.

## Positive-completion diffusion loss

`configs/positive_diffusion.yaml` selects
`ar_grpo.positive_diffusion_policy.PositiveDiffusionARPolicy`, with causal AR
rollouts and ordinary AR GRPO scoring. Run it through the legacy launcher:

```bash
uv run research/just_grpo/run_ar_grpo.py --config research/ar_grpo/configs/positive_diffusion.yaml
```

The variant adds `policy.diffusion_aux_loss.weight` (default 0.1) times a masked
cross-entropy objective on samples whose mean response advantage is > 0. Set
`positive_samples: reward` to select reward > 0 instead, provided the controller
supplies `rewards` in the policy batch. These are binary selection rules; the
auxiliary CE is not weighted by advantage magnitude.

For each sequence, draw a masking probability uniformly between 0.05 and 1.0,
then mask eligible assistant tokens independently. Prompts, tool outputs,
nonpositive completions, and padding are excluded. Multiple assistant spans are
supported. The full original sequence forms both noisy and clean halves, using
the existing asymmetric block attention with a sequence-relative block grid.
Only the noisy canvas is extended to a block boundary. The clean context and AR
loss metadata keep their original batch width; NeMo-RL still applies ordinary
model alignment padding. The two halves may have different widths.
Clean tokens from the target block and future blocks are hidden from its noisy
queries. Targets are scored at the same position, at temperature 1.

The objective is `AR_GRPO + weight * sum_positive,sum_masked(-logp / p_mask) / N_AR`,
where `N_AR` is the valid response-token count for the AR batch, including
nonpositive responses. Both objectives use one asymmetric forward/backward,
followed by one normalization, reduction and optimizer step. An empty positive
set adds zero auxiliary gradients; weight 0 uses ordinary AR training. Mask draws are shared across
TP ranks through a CPU generator seeded by the configured seed plus DP rank.

Input preparation and the combined loss processor are registered through the
worker constructor. A per-sample training flag lets NeMo-RL validate and pad the
original AR batch before input preparation builds the wider noisy canvas.
Ordinary AR scoring batches pass through input preparation unchanged, and batches without diffusion metadata use the upstream AR loss
processor. The attention and temperature context remains active through backward.

The worker also supports SingleController `begin_train_step` / `train_microbatch`
/ `finish_train_step`. Select the same worker FQN and copy the `diffusion_aux_loss`
block into an existing SingleController policy configuration. The DFW launcher
includes both research packages. The ordinary `train` call currently accepts one
complete global batch; streaming uses the split API. Megatron PP=CP=1, unpacked
fixed-size batches, token-level GRPO loss, and ordinary vocabulary logits are
required. Draft/MTP heads and router replay are unsupported and rejected.

Metrics: `diffusion_aux_loss` reports the weighted contribution to total loss;
`diffusion_aux_masked_tokens` and `diffusion_aux_positive_samples` report counts.
The doubled auxiliary input increases memory and computation; choose context and
microbatch size accordingly.

GPU validation: smoke job 19760523 completed two optimizer updates with nonzero
auxiliary loss and finite gradients. The four-node DeepScaleR job 19760966
completed 80 updates before timing out during validation, with step 70 saved.
