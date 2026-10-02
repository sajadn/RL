# AR GRPO with multi-mode validation

`ar_grpo/policy.py` contains the causal Nemotron policy adapter shared by the
SingleController launcher (`run_ar_grpo.py`) and the legacy GRPO launcher
(`../just_grpo/run_ar_grpo.py`). Both use
`ar_grpo.policy.NemotronDiffusionMegatronPolicyWorker`.

The adapter selects causal attention for whole-batch training, SingleController
microbatch training, logprob calculation, and top-k scoring. It disables KV
caching and restores the previous attention settings after each call.

The legacy launcher uses the DeepScaleR/AIME2024 recipe under
`../just_grpo/configs/recipes/`. Its shared generation and multi-mode validation
remain in the `block-diffusion` package; the SingleController launcher keeps its
existing controller behavior.

For provisioned interpreters, include `research/ar_grpo`, `research/block_diffusion`,
and the repository root in `PYTHONPATH`.
