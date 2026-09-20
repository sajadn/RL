# Understanding GDPO PR #3550

PR: https://github.com/NVIDIA-NeMo/RL/pull/3550

Reviewed revision: `70ec1fa225f15d15c26a874445a5c6a3c0fb375e`.

## Changes in `nemo_rl/algorithms/loss/loss_functions.py`

The changes are twofold:

1. **Enable position-aligned log probabilities** with `position_aligned_logprobs`. Autoregressive models score token `i+1` at position `i`, so the existing loss drops the first column of masks, advantages, and stored log probabilities. Masked diffusion models such as LLaDA score token `i` at position `i`. Setting this flag to `True` keeps every position aligned without dropping a column. Keeping this capability in core is acceptable.

2. **Support cases where generation log probabilities are unavailable** with `generation_logprobs_available`. GDPO explicitly passes `False` from its entrypoint. Its denoising sampler returns zero-filled placeholders rather than real generation log probabilities. The flag skips generation-versus-training discrepancy metrics and the generation-based entropy approximation, and sets sampler importance-sampling correction weights to `1`. The skipped metrics are reported as `0`; these zeros mean unavailable, not measured zero.

## What `curr_logprobs` and `prev_logprobs` mean

GDPO's intended scoring flow computes both scores separately from generation:

- `prev_logprobs`: ELBO contributions computed by scoring the completed response before optimization; held fixed during updates on that rollout batch.
- `curr_logprobs`: ELBO contributions recomputed for the same response during optimization, using the current policy with gradients enabled.
- `generation_logprobs`: zero-filled placeholders because this sampler does not supply the corresponding generation probabilities.

The PPO ratio remains active even when `generation_logprobs_available=False`. With GDPO's sequence-level configuration, it is:

```text
ratio = exp(mean_over_response_positions(curr_ELBO_contributions - prev_ELBO_contributions))
```

This uses the length-normalized ELBO difference. Disabling the sampler correction does not disable the current-versus-previous policy ratio.

## Correctness issues in the GDPO worker integration

These findings apply to the reviewed revision above and come from source inspection, not an executed training run. Neither fix has been implemented here.

### Previous-policy scoring bypasses the ELBO estimator

The core worker's `get_logprobs()` still executes its forward inline instead of calling the newly added `_logprobs_for_microbatch()` hook. Consequently, the override in `DTensorGDPOPolicyWorker` that invokes `_gdpo_elbo_logprobs()` is never used by that path.

The actual input to this faulty previous-score path is the clean prompt plus generated response. It is not fully masked; no diffusion corruption is applied at all. With position-aligned scoring, each answer token is visible in the model input while its own probability is scored.

The intended ELBO path instead masks subsets of response positions at different masking rates, scores the original tokens at those masked positions, and accumulates weighted contributions. It does not require all positions to be masked in every forward.

After resolving the constructor failure below, leaving this hook disconnected would compare current ELBO contributions against previous clean-input token scores. The ratio would mix a change in scoring procedure with the policy update, rather than comparing the same estimator under current and previous weights. This is a correctness bug, not just an inefficient implementation.

Fix: replace the inline scoring block inside `get_logprobs()` with a call to `self._logprobs_for_microbatch(...)`, allowing the GDPO override to run. See the [base microbatch loop](https://github.com/kashif/RL/blob/70ec1fa225f15d15c26a874445a5c6a3c0fb375e/nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py#L654) and [GDPO override](https://github.com/kashif/RL/blob/70ec1fa225f15d15c26a874445a5c6a3c0fb375e/research/gdpo/gdpo/worker.py#L173).

Before trusting training results, verify through the actual worker scoring paths that unchanged weights, identical sampled masks, and matching scoring conditions produce matching current/previous ELBO contributions and a policy ratio near `1`. Control stochastic behavior such as dropout so the comparison isolates scoring correctness.

### Training scorer passes unsupported constructor arguments

`_gdpo_train_elbo_scorer()` constructs `LogprobsPostProcessor` with `device_mesh`, `cp_mesh`, `tp_mesh`, and `cp_size`. The constructor at this revision accepts only `cfg`, `enable_seq_packing`, `sampling_params`, and `shift_targets`. The extra arguments therefore cause an unexpected-keyword-argument `TypeError` when constructing the training scorer.

Fix: remove the four unsupported arguments and use the actual constructor contract. See the [GDPO constructor call](https://github.com/kashif/RL/blob/70ec1fa225f15d15c26a874445a5c6a3c0fb375e/research/gdpo/gdpo/worker.py#L152) and [constructor definition](https://github.com/kashif/RL/blob/70ec1fa225f15d15c26a874445a5c6a3c0fb375e/nemo_rl/models/automodel/train.py#L718).

This issue causes an immediate failure. Fixing it alone would still leave the previous-policy scoring mismatch above.

## Changes in `nemo_rl/algorithms/loss/utils.py`

`prepare_loss_input()` gains `precomputed_logprobs=False` because GDPO aggregates scores across masked samples before passing them to this function. Each masked forward has already undergone `log_softmax` and target-token gathering. The estimator then combines those log probabilities using its masking indicators and weights into per-position ELBO contributions.

Consequently, the input is already a `[batch, sequence]` score tensor, rather than `[batch, sequence, vocabulary]` logits. With `precomputed_logprobs=True`, the function casts the scores to float32 and bypasses logits-to-log-probabilities conversion and next-token truncation. Passing accumulated scores through an argument named `logits`, then using a flag to bypass its usual processing, makes the interface less clear.

An alternative is to follow the [diffusion integration design document](https://docs.google.com/document/d/1HHzRpLfsRalQkqLGFe0vljZjRO7dwKaKWH43jZPxCu4/edit?tab=t.0) and introduce a `DenoisingSchedule` that supplies each masked sample separately. The schedule would define masked inputs, clean targets, scoring masks, and estimator weights; the shared diffusion worker would run each masked forward, convert its logits into position-aligned token log probabilities, and accumulate the weighted scores.

```text
DenoisingSchedule → individual masked forwards → aligned log-softmax and gather
                  → weighted score accumulation → policy loss
```

The scores must still be combined before computing the policy ratio and clipping; averaging a separate loss for each mask would change the objective. The current implementation also runs individual masked forwards—the proposed change is to organize that work through the shared schedule and worker interface.

The design document does not specify the training integration with `prepare_loss_input()`. If the worker passes accumulated scores directly to the loss, `precomputed_logprobs` can be avoided, while preserving the necessary masking, packing, and distributed normalization. Introducing the schedule alone does not eliminate the need to handle precomputed scores if they still pass through `prepare_loss_input()`.

For the initial implementation, prefer the simple execution approach below: retain a routine like `gdpo_forward_backward` and, with the existing loss interfaces, `precomputed_logprobs=True`. Passing scores directly to the loss is an optional later interface cleanup, not a prerequisite for introducing the schedule.

## Execution approaches for `train_gdpo.py`

`gdpo_forward_backward()` runs the worker's microbatch scoring, loss calculation, and backward pass. Its execution pattern can be shared by diffusion methods that combine several masked forwards into one score estimate; GDPO-specific sampling and weighting remain in the schedule.

**Decision: start with the simpler approach.** Retain the combined computation graph and apply the policy loss once per microbatch. Defer the more complicated two-pass approach until activation memory motivates it. This decision is documented; no training code has been changed.

### Simple approach: accumulate scores, then backpropagate once

1. The schedule supplies each masked input, clean targets, scoring mask, and estimator weight.
2. Run each masked forward with gradients enabled, convert logits into aligned token scores, and accumulate the weighted contributions while retaining their autograd graphs.
3. Compute the policy ratio and clipped loss from the combined current scores and fixed previous scores.
4. Backpropagate that loss once, through all masked forwards.

```text
schedule -> masked forwards -> weighted score accumulation with graphs
         -> one policy loss -> backward
```

With the PR's existing interfaces, this still needs a routine like `gdpo_forward_backward` and `precomputed_logprobs=True` to pass accumulated scores through `LossPostProcessor` and `prepare_loss_input`. The schedule organizes the masked evaluations; it does not eliminate these execution and interface responsibilities. The tradeoff is retaining activation graphs for multiple masked forwards until backward.

### More complicated approach: two-pass gradient accumulation

Compute the same objective with less retained activation memory by recomputing the masked forwards. For per-position score tensors, let `s_j(theta)` be one masked forward's scores and `a_j` include its fixed estimator coefficient and scoring mask:

```text
S = sum_j(a_j * s_j(theta))
L = policy_loss(S, previous_S, advantages)
g = dL/dS

gradient_theta(L) = sum_j gradient_theta(sum_positions(g * a_j * s_j(theta)))
```

The policy loss includes the response-length normalization, ratio, clipping, masks, and loss normalization. Treat `g` as fixed during recomputation.

1. First pass: compute and combine the current scores without retaining model graphs. Treat the combined tensor as a differentiable leaf, evaluate the final policy loss, and obtain `g`.
2. Second pass: recompute each masked forward with gradients enabled and backpropagate the scalar `sum(g * a_j * s_j)`, accumulating parameter gradients. Each forward's graph can then be released.
3. Apply the optimizer update only after all contributions have accumulated, respecting any outer microbatch accumulation.

Both passes must use the same model weights, masks, and scoring conditions; replay stochastic model behavior if present. Distributed synchronization and loss scaling must preserve the original gradient. This trades additional forward computation and implementation complexity for lower activation memory.

Neither approach applies a separate GRPO loss to each masked sample. These samples are evaluations of the same response, and exponentiation and clipping are nonlinear: averaging per-mask policy losses would change GDPO's objective. Ordinary gradient accumulation across independent response microbatches remains compatible with either approach.

## Changes in `nemo_rl/algorithms/grpo.py`

The PR introduces three main changes:

1. **Custom generation setup:** `setup()` accepts a `generation_factory` that constructs a generation adapter from the policy config and initialized policy. This path requires colocated inference and reserves one worker group, allowing GDPO generation to run on the policy workers without core importing GDPO.
2. **Unavailable generation log probabilities:** `setup()` forwards `generation_logprobs_available` to `ClippedPGLossFn`. Both synchronous and asynchronous training loops skip sequence-level generation-versus-training error metrics and their associated sample masking when generation log probabilities are unavailable.
3. **Repeated training on a rollout batch:** `grpo.num_updates_per_rollout` defaults to `1` and controls repeated calls to `policy.train()` on the same batch, with previous scores and advantages held fixed. The PR also adjusts Megatron's scheduler step count and adds per-iteration loss printing. Subsequent reporting uses the final iteration's `train_results`.

### Preferred research integration

Introduce a dedicated algorithm module such as `research/gdpo/gdpo/algorithms/gdpo.py` to own GDPO training orchestration. It should set up denoising generation, coordinate previous and current ELBO scoring, and omit metrics and sampling corrections that require unavailable generation log probabilities.

Reuse or extract shared setup and training utilities rather than copying all of `grpo.py`. A generic generation factory can remain in core if needed for that reuse, while GDPO-specific orchestration stays in research. Reusing the shared policy loss also requires separating its generation-dependent metrics and corrections; moving orchestration alone does not remove those dependencies.

### Separate PR for `num_updates_per_rollout`

Move `num_updates_per_rollout` into a separate PR, including the config field, changes to both training loops, scheduler adjustment, and relevant validation. Repeated training on rollout batches is a general GRPO feature and should be reviewed independently of the GDPO research integration.

These are agreed refactor directions recorded for review; no implementation changes have been made here.

## Changes in `nemo_rl/distributed/model_utils.py`

The PR adds `shift_targets=True` to `dtensor_from_parallel_logits_to_logprobs`, `from_parallel_logits_to_logprobs`, and the forwarding helper `get_logprobs_from_vocab_parallel_logits`. These cover different distributed paths; they do not apply three consecutive shifts.

With `shift_targets=True`, position `i` scores target token `i+1`, and the final score is dropped, producing `[batch, sequence - 1]`. With `False`, position `i` scores target token `i`, and all sequence positions are retained. The existing vocabulary normalization and target-score gathering remain in place. This general alignment support should stay in core.

### One alignment setting across all affected paths

Use one authoritative alignment flag, propagated through every affected path, rather than requiring users to configure opposite flags such as `shift_targets=False` and `position_aligned_logprobs=True` independently.

This setting must govern all operations whose behavior depends on next-token versus position alignment, including:

- Target shifting when gathering token scores from vocabulary logits.
- Score truncation and expected output sequence length.
- Slicing masks, advantages, and stored previous, reference, or generation scores to match the computed scores.
- Any corresponding alignment assumptions in loss preparation, metrics, filtering, sequence packing, and distributed scoring paths that are used or enabled.

The scope is not limited to the two locations discussed so far. Audit the relevant paths for target rolls, first/last-column slicing, and other next-token assumptions, and derive their behavior from the same setting. Each operation should align its tensors once; propagating one setting must not introduce repeated shifting. Model-specific layout or padding operations should remain distinct from token alignment.

This is the preferred refactor direction, not an implemented change or a claim that every path has already been audited. Whether scores are precomputed remains a separate property from their alignment.

## Changes in `nemo_rl/models/automodel/data.py`

The PR adds `target_ids: Optional[torch.Tensor] = None` to `ProcessedInputs`. GDPO feeds masked `input_ids` to the model but needs to score the original clean tokens. The new field carries those clean targets through the existing `LogprobsPostProcessor`, which uses `target_ids` when supplied and otherwise falls back to `input_ids`.

This is an interface choice, not a requirement of GDPO. Because the schedule or worker constructs the masked samples itself, it can retain the clean tokens before masking and pass them directly to the scoring helper:

```python
clean_ids = batch["input_ids"]
masked_ids = apply_mask(clean_ids)  # Construct separately; preserve clean_ids.
logits = model(masked_ids)
scores = aligned_log_softmax_and_gather(logits, clean_ids)
```

With that schedule/worker integration, adding `target_ids` to the core dataclass may be unnecessary. The scoring path still needs access to clean targets, with the appropriate packing or distributed layout, but they do not have to be carried in `ProcessedInputs`. Whether to retain the field depends on the chosen interface; keeping it in core is not an agreed requirement.

## Group generation code under `gdpo/generation/`

Move `research/gdpo/gdpo/denoise.py` into a generation package. It contains rollout generation utilities: canvas construction, token sampling, per-step unmasking budgets, block denoising, and output repacking. These belong with the generation adapter, separately from ELBO likelihood estimation.

Proposed layout:

```text
research/gdpo/gdpo/generation/
    __init__.py
    denoise.py       # Sampling loop and generation utilities
    automodel.py     # Adapter currently in generation.py
```

Move the existing `gdpo/generation.py` to `gdpo/generation/automodel.py` and update imports or package exports accordingly, avoiding a module/package name collision. This layout is a proposed refactor; the source files have not been moved.

### Keep `AutomodelGeneration` in research for this PR

Keep the generation adapter at the proposed `research/gdpo/gdpo/generation/automodel.py` location for this PR. The name `AutomodelGeneration` does not require a core location; placement should reflect its supported contract and maturity. Keeping that name is acceptable if its documentation clearly states that the implementation currently supports generation through GDPO's policy workers.

The adapter's dispatch and shared-weight mechanics are reusable, but those mechanics alone do not establish general Automodel generation support. Its current integration relies on GDPO's worker implementation and configuration validation, including restrictions on batching and colocation.

Consider promotion to core separately: define the shared worker generation interface, remove reliance on GDPO-specific validation, and validate lifecycle and parallelism behavior for the intended supported use cases. Model-specific sampling can remain in research. This keeps the GDPO PR focused without ruling out a general Automodel generation backend later.

## Move ELBO estimation into `gdpo/algorithms/`

Move `research/gdpo/gdpo/elbo.py` to `research/gdpo/gdpo/algorithms/elbo.py`. It implements GDPO's likelihood estimator and its variance-reduction strategy: deterministic quadrature for the masking-time integral, Monte Carlo mask sampling, reproducible seeds shared between current and previous policy scoring, and weighted accumulation into per-position ELBO contributions. These are algorithm responsibilities, separate from rollout generation.

Under the proposed `DenoisingSchedule` design, the mask sampling, quadrature points, and estimator weights would define GDPO's schedule. GDPO-specific weighting rules should remain in `algorithms/`, while generic model forwards, aligned token scoring, and accumulation mechanics can be handled by the shared diffusion worker. Update imports and corresponding test locations when implementing the move.

This is a proposed refactor; the source file has not been moved.

## Refactor direction

Keep the position-alignment support in core. Hopefully, we can remove `generation_logprobs_available` and its associated handling from core and move that responsibility into `research/gdpo`. This is a desired refactor direction, not an implemented change; the research implementation would still need to avoid using placeholder generation probabilities while preserving the current-versus-previous policy loss.

## Implementation and reproduction — September 20, 2026

The sections above record our review of the original PR revision. The refactor is implemented separately on DFW at:

```text
/lustre/fsw/portfolios/coreai/users/snorouzi/worktrees/gdpo-research-3550
branch: refactor/gdpo-research
```

The layout follows [the Flow-GRPO research refactor](https://github.com/triple-mu/RL/pull/1), reviewed at `543a5d9cd6d9f8bb5d0f385339fa6963a6985e79`.

Implemented changes:

- `run_gdpo.py` is the entrypoint. `gdpo/algorithms/gdpo.py` owns GDPO setup and selects its loss and rollout diagnostics while reusing the shared GRPO loop.
- `gdpo/algorithms/loss/gdpo.py` reuses the clipped policy objective and omits generation-dependent corrections and metrics. `generation_logprobs_available` has been removed from core; unavailable metrics are omitted rather than reported as measured zeros.
- Generation is grouped under `gdpo/generation/`; SDMC estimation lives in `gdpo/algorithms/elbo.py`. The Automodel generation adapter remains in research.
- `DenoisingSchedule` supplies masked inputs, clean targets, scoring masks, and weights. Shared research execution accumulates scores with all forward graphs retained, then runs one policy loss/backward per microbatch. The initial implementation keeps `precomputed_logprobs`; two-pass recomputation remains deferred.
- Clean targets remain in the loss-side microbatch, separate from the prepared masked model inputs. Core `ProcessedInputs.target_ids` and forward-signature inspection have been removed. The research LLaDA adapter omits model-facing `position_ids` without changing the clean microbatch.
- `loss_fn.position_aligned_logprobs` is the only user alignment setting. GDPO requires it to be true and derives unshifted scoring from that contract; the inverse `policy.masked_diffusion.shift_targets` setting is rejected. Dense, vocabulary-parallel, context-parallel, loss-preparation, filtering, and rollout-diagnostic helpers propagate alignment. GDPO still rejects context parallelism, dynamic batching, and sequence packing.
- The actual `get_logprobs()` loop calls the worker hook, so previous scores use masked ELBO estimation. The training scorer uses the real post-processor constructor contract.
- Two additional missing API connections were fixed: the training loss call now supplies `cp_sharder=None` for GDPO's supported CP=1 path, and the context-parallel scoring helper now accepts the `shift_targets` argument already passed by the PR's Automodel post-processor.
- The reproduction exposed a generation-length bug: each prompt received the full configured generation canvas even when that exceeded its remaining sequence budget. The refactor masks and attends only `min(max_new_tokens, max_total_sequence_length - prompt_length)` generation slots per row, and ignores stop tokens in padding when unpacking. Canvas width and forward count stay consistent across FSDP ranks.
- Research dependencies and actor environment registration follow the project-local pattern. A locked installation dry run includes Automodel; no dependency versions were changed by the refactor.

The generic `num_updates_per_rollout` changes are extracted to `refactor/grpo-num-iterations`, commit `a8c43970543c231d5dfb785cee5286d8f8310a7b`, based on upstream `612d5274059c821dc09c2c90767b546b6f2d50b5`. This includes the ordinary, asynchronous, and data-plane GRPO loops, scheduler scaling, config defaults, and the original repetition test. The GDPO recipe retains its 12 updates per rollout batch; the generic feature can be reviewed independently.

Validation before the September 21 rebase: 262 tests passed across the validation runs: 253 research/core loss tests and 3 GRPO integration tests in the CUDA container, 3 CPU distributed target-alignment tests, and 3 additional generation-budget regressions. After the budget fix, all 49 denoiser/worker tests passed on CPU, covering mixed prompt lengths, zero remaining budget, stop tokens outside the budget, and padding that shares the mask token ID. The worker regression exercises the actual previous-score path, scorer construction, loss post-processing, and backward: unchanged weights and deterministic scoring produce matching ELBO scores and a ratio of 1. These checks used an existing Python 3.13.13 test interpreter plus isolated test dependencies; they are not an end-to-end run in the recipe's new Python 3.13.14 environment. Project type checking reports zero errors, and lint checks pass.

### Original recipe reproduction

The original PR checkout is preserved separately at:

```text
/lustre/fsw/portfolios/coreai/users/snorouzi/runs/gdpo_pr3550_70ec1fa2/repo
```

Job `19023464` executed the unchanged `gdpo-llada-8b-instruct-1n8g-fsdp2tp1.yaml` recipe under `coreai_dlalgo_genai`: one node, eight H100 GPUs, and a three-hour allocation. The recipe uses LLaDA-8B-Instruct with LoRA, OpenMathInstruct-2's `train_1M` split with a 5% validation split, 200 rollout steps, and 12 policy updates per rollout. Model and dataset shards are cached. The Python 3.13.14/CUDA 13.2 environment is isolated from existing training environments; the retry installs it on node-local storage with a dedicated source-build cache.

Result: job `19023464` failed after 15 minutes 58 seconds during initial validation at step 0. Dependencies built successfully, the dataset loaded, and all eight workers loaded LLaDA and generated responses. The rollout rejected a returned sequence that exceeded the 512-token total limit:

```text
nemo_rl/experience/rollouts.py:1109
AssertionError: tokens_left_for_obs=-131 should not be negative.
This should not happen if the inference engine respects the max sequence length.
```

The sampler always allocated `max_new_tokens=256`, without accounting for prompt length. This failure occurred before a completed validation metric or training update. We have no recorded positive reward or accuracy-improvement result from this reproduction. The generation-budget fix above is implemented and regression-tested in the refactor; the refactor has not yet had an end-to-end training run.

Run: [W&B reproduction](https://wandb.ai/nvidia/nemo-rl/runs/zxbvjnky). Full log on DFW:

```text
/lustre/fsw/portfolios/coreai/users/snorouzi/runs/gdpo_pr3550_70ec1fa2/recipe-19023464.log
```

Setup qualification: earlier attempts failed before model execution. Job `19019941` had a launch-script typo; job `19020070` reused a stale CMake/Ninja path from an auxiliary temporary environment; jobs `19023256` and `19023349` lacked cached Git dependencies in the newly isolated cache. These were our setup failures, not GDPO results. The final attempt used clean source-build checkouts and the exact pinned cached artifacts, after an actual offline base installation and an offline Automodel dependency dry run passed. The original PR source remains unmodified.

### September 21 rebase onto upstream main

Rebased the two-commit stack onto `612d5274059c821dc09c2c90767b546b6f2d50b5`. Preserved upstream full-vocabulary distillation support and its tests. The new public GRPO wrapper now forwards the research rollout-metrics callback to the training implementation while retaining environment cleanup.

Validation: 241 CPU tests passed, covering GDPO research behavior, full-vocabulary distillation, target alignment, and config compatibility. Three mocked GRPO wrapper checks passed when invoked directly; pytest collection of those checks was blocked by the module-level Ray environment fixture timing out on the login node. Lint, formatting, and syntax checks passed. Upstream locked dependency packages are unchanged. These checks used the existing test environment, not a new installation or a full training run with the updated upstream dependencies.
