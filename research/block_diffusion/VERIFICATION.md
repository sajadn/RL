# Block diffusion extraction verification

2026-10-02, commit `5c8085c2b`: extracted common runtime code without changing algorithm behavior.
The causal AR adapter was subsequently moved to `research/ar_grpo/ar_grpo/ar_policy_worker.py`.

- 287 CPU tests passed across block diffusion, JustGRPO, and TraceGRPO.
  The Megatron adapter test module was skipped because the CPU interpreter has
  no `megatron.bridge` installation.
- All 18 algorithm recipes resolve identically before/after extraction, allowing
  only relocated worker class paths and removal of Trace's empty `just_grpo` key.
- Trace imports and all six Trace recipes load with JustGRPO imports blocked.
- Ruff lint/format checks and shell syntax checks pass.
- The block-diffusion source distribution and wheel build successfully; the
  wheel includes the shared runtime and Sudoku prompt data.
- Workspace lock metadata matches the three local project definitions. External
  dependency records remain unchanged. Native `uv lock` validation could not run:
  the login interpreter is 3.13.13, the repository requires 3.13.14, and the
  installed uv has no download for that version. The local workspace metadata
  was updated directly and checked against the project TOML definitions.

This extraction has not been submitted as a new GPU smoke job. Existing jobs
use their previous source trees or the separate SingleController AR entrypoint.
