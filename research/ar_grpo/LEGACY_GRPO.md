# AR GRPO with multi-mode validation

`ar_grpo/ar_policy_worker.py` contains the causal Nemotron policy adapter used
by `../just_grpo/run_ar_grpo.py` and its DeepScaleR/AIME2024 recipe. The adapter
selects causal attention for training and logprob calculation. Shared generation
and multi-mode validation remain in the `block-diffusion` package.

For provisioned interpreters, include `research/ar_grpo`, `research/block_diffusion`,
and the repository root in `PYTHONPATH`. The launcher and recipe remain under
`research/just_grpo`, matching their existing paths.

The existing `ar_grpo/policy.py` and `run_ar_grpo.py` are the separate
SingleController implementation. Their contents are unchanged by this move.
