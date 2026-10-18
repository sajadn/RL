# Long-context causal AR training

Use `examples/configs/ar_grpo_superv3_exp544_step100_helpful_128k_ready_first_tp2_cp8_300steps.yaml`
for 300 optimizer updates. The two-update smoke configuration is its parent
`ar_grpo_superv3_exp544_step100_helpful_128k_ready_first_tp2_cp8.yaml`.

Both use 131072 total context tokens, 110592 thinking tokens and 114688 maximum
output tokens per call. Training uses TP=2, CP=8, PP=1, DP=4 across eight
8-GPU nodes. Generation uses sixteen 8-GPU nodes with TP=1 and at most two
concurrent sequences per GPU. Each optimizer update consumes 16 prompt groups
with 16 responses each (256 responses), streamed in chunks of eight groups.
The sampler is `ready_first` with `max_staleness_versions: 1`.

The shorter 300-update variants keep the same topology, dataset and batch size:

- `ar_grpo_superv3_exp544_step100_helpful_64k_ready_first_tp2_cp8_300steps.yaml`:
  65536 context, 57344 generation and 55296 thinking tokens.
- `ar_grpo_superv3_exp544_step100_helpful_32kgen_28kthink_ready_first_tp2_cp8_300steps.yaml`:
  65536 context, 32768 generation and 28672 thinking tokens.

Sequence packing and activation checkpointing are enabled. CP zigzag sharding
and TP sequence parallelism require `policy.make_sequence_length_divisible_by`
to be a multiple of `2 * CP * TP`; this configuration uses 32. Gym HTTP ports
5000-5799 and Python tool port 5900 avoid the Ray, generation and sandbox ranges.
GPU monitoring and separate Gym service logs are enabled.

The Bridge submodule pins the causal CP port at
`bd6b1507bbee5ceea9ac3e5154a425d882b89784`, preserving native RoPE. Its Megatron-Core
revision is `1e7598cbfae888cdd3d741a351aae588d56f66c0`.

Submit through `research/ar_grpo/submit_dfw.sh` with `CONFIG` set to the desired
YAML, `NODES=24`, a unique `RUN_NAME`, and an eligible `ACCOUNT` selected from
current fairshare. The 300-update YAML uses a 3h45m checkpoint deadline for
`TIME=04:00:00` allocations. It inherits checkpoint saving after each update.
The launcher uses the provisioned DFW policy, vLLM, Gym and sandbox runtimes;
its deployment paths must exist on the submitting cluster.

The shared AR entrypoint is `research/ar_grpo/run_ar_grpo.py`, moved from
`research/just_grpo`. It selects SingleController for `grpo.async_grpo: null`
and preserves the legacy controller and validation route for other configs.

The driver checks its RL/Bridge imports, CUDA compatibility and packing
parameters before starting the SingleController. Both source YAMLs with
`defaults` and resolved YAML snapshots are accepted. The helpful chat template
is stored in this checkout and referenced through `REPO_DIR`.

Validation: job 20200968 completed both smoke-test updates and saved checkpoint
step 2 in 57m53s with exit code 0. The 300-update variants pass schema
and inheritance checks. This smoke result establishes two successful updates,
not completion of the 300-update target.
