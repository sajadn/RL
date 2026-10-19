# ForkGRPO

[Fork-dLLM / ForkGRPO](https://arxiv.org/html/2609.39859v1#S3) keeps
confidence-threshold parallel decoding and switches to the leftmost masked
position when no sampled token exceeds the threshold. Confidence uses the
untempered sampled-token probability; stored policy logprobs use the sampling
temperature. The sampler records an explicit `fallback_mask` and reveal steps.

ForkGRPO replays every fallback decision in its recorded masked context, scoring
only fallback tokens through the first stop token. Parallel tokens and retained
terminal-block tokens remain available as context. The shared block attention
allows the same recorded step in different response blocks to be scored together.
ForkGRPO directly uses `TraceGRPOSchedule` from `trace_grpo/algorithm.py`.
Fork's data preparation supplies all fallback levels, the fallback-only loss mask,
and per-response weights. There is no additional random level sampling or
entropy-percentile selection in ForkGRPO.

## Empty responses and normalization

Each response with fallback tokens has total loss weight one (`1 / fallback_count`
per token). Responses without fallback tokens contribute zero policy loss and
remain in the reward group. The policy loss is averaged over nonempty responses;
this explicitly extends the paper, whose `1 / |F|` formula does not define the
empty case. A globally empty batch still runs aligned zero-loss views but skips
optimizer and scheduler updates, including Adam momentum and weight decay.

## Run

From the repository root in a provisioned Megatron environment:

```bash
uv run --all-packages --extra mcore python research/fork_grpo/run_fork_grpo.py \
  policy.model_name=/path/to/Nemotron-Labs-Diffusion-3B \
  logger.log_dir=/path/to/fresh/results
```

The single recipe trains Nemotron-3B on DeepScaleR and validates on AIME2024,
using four nodes, 4K context, and 100 optimizer steps. It is an implementation baseline, not a reproduction
of the paper's LLaDA LoRA results. The recipe uses two training nodes and two
asynchronous vLLM generation nodes, with training microbatch 2 and logprob batch 4.
Set `NRL_VLLM_PY_EXECUTABLE` to the reference fork interpreter with explicit
`fallback_mask` output support. Megatron generation is also supported with
`runtime: megatron` and `selection_policy: fork`. Training uses the shared GRPO controller and
Megatron diffusion worker. The recipe disables the reference KL penalty.
`max_steps >= block_size` prevents forced parallel final-step commits.

For provisioned interpreters, add the repository, `research/block_diffusion`, and
`research/fork_grpo` and `research/trace_grpo` to `PYTHONPATH`.

## Tests

```bash
uv run --package fork-grpo --group test python -m pytest \
  -c research/fork_grpo/pyproject.toml research/fork_grpo/tests/unit \
  --confcutdir=research
```

A two-GPU, three-update execution smoke is available at
`tests/functional/deepscaler.sh`, with `MODEL` and a fresh `OUTPUT_DIR` supplied by the
caller. GPU execution is required to validate real Megatron training; CPU tests
cover sampler ordering, temperature semantics, replay logprob parity, terminal
context, metadata transport, empty/mixed rows, sharding, and recipe validation.
