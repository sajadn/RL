# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Correctness filtering, replay states, and privileged future conditioning."""

from typing import Any, Literal, Self

import torch
from pydantic import Field, model_validator

from block_diffusion.block_layout import BlockDiffusionLayout
from block_diffusion.config import BlockDiffusionConfig, DiffusionExperimentConfig
from block_diffusion.denoising_schedule import DenoisingSchedule
from block_diffusion.replay import recorded_reveal_states
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class DOPSDScheduleConfig(BlockDiffusionConfig, extra="forbid"):
    seed_base: int


class DOPSDConfig(DiffusionExperimentConfig):
    schedule: DOPSDScheduleConfig
    teacher_retain_ratio: float = Field(ge=0, lt=1, allow_inf_nan=False)
    reward_threshold: float = Field(allow_inf_nan=False)
    correct_only: bool
    selection_source: Literal["teacher", "student"]
    pointwise_clip: float | None = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_replay(self) -> Self:
        if self.runtime not in ("megatron", "reference_vllm"):
            raise ValueError(
                "d-OPSD requires recorded Megatron or reference vLLM rollouts"
            )
        if self.sampling.temperature <= 0:
            raise ValueError("Training temperature must be positive")
        for sampling in (self.sampling, self.validation_sampling):
            if sampling.block_size != self.schedule.block_size:
                raise ValueError("Replay and generation block sizes must match")
            if not sampling.returns_reveal_steps or not sampling.emit_full_blocks:
                raise ValueError(
                    "d-OPSD requires reveal steps and full terminal blocks"
                )
            if sampling.returns_entropy:
                raise ValueError("d-OPSD does not consume rollout entropies")
            if self.runtime == "megatron" and sampling.selection_policy not in (
                "leftmost",
                "confidence_threshold",
            ):
                raise ValueError("Megatron supports leftmost or confidence_threshold")
            if (
                sampling.selection_policy == "leftmost"
                and sampling.block_size % sampling.max_steps
            ):
                raise ValueError("Leftmost sampling needs a divisible reveal schedule")
        return self


def select_correct_responses(
    rewards: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    group_size: int,
    threshold: float,
    correct_only: bool,
) -> torch.Tensor:
    """Keep the first accepted response in each contiguous prompt group.

    Generation uses a batched pass@k, rather than stopping the engine early.
    With correct_only=False, keep every valid response for the ablation.
    """
    if rewards.shape != sample_mask.shape or rewards.numel() % group_size:
        raise ValueError("Rewards and masks must contain complete prompt groups")
    if not torch.isfinite(rewards).all():
        raise ValueError("d-OPSD requires finite verification rewards")
    if not correct_only:
        return sample_mask.clone()
    accepted = ((rewards >= threshold) & (sample_mask > 0)).reshape(-1, group_size)
    first = accepted & (accepted.long().cumsum(1) == 1)
    return sample_mask * first.reshape_as(sample_mask)


class DOPSD:
    """Prepare on-policy replay and reward filtering before worker sharding."""

    def __init__(
        self, config: DOPSDConfig, *, stop_token_ids: list[int], group_size: int
    ):
        self.config = config
        self.stop_token_ids = stop_token_ids
        self.group_size = group_size

    def prepare_training_data(
        self,
        data: BatchedDataDict[Any],
        message_logs: list[list[dict]],
        step: int,
    ) -> None:
        # Validate recorded states before dropping failed responses.
        levels, loss_mask = recorded_reveal_states(
            data, message_logs, stop_token_ids=self.stop_token_ids
        )
        if (levels >= self.config.sampling.max_steps).any():
            raise ValueError("Recorded trajectory exceeds max_steps")
        data["sample_mask"] = select_correct_responses(
            data["total_rewards"],
            data["sample_mask"],
            group_size=self.group_size,
            threshold=self.config.reward_threshold,
            correct_only=self.config.correct_only,
        )
        data["dopsd_reveal_levels"] = levels
        data["dopsd_loss_mask"] = loss_mask
        data["dopsd_seeds"] = (
            self.config.schedule.seed_base
            + step * 1_000_003
            + torch.arange(data.size, device=levels.device)
        )


class DOPSDSchedule(DenoisingSchedule):
    """Replay all levels; each block's actual transitions receive equal weight.

    Independent response blocks can be evaluated in one doubled-attention view.
    The loss still averages over individual block transitions, then responses.
    """

    def __init__(
        self,
        data: BatchedDataDict[Any],
        config: DOPSDConfig,
        *,
        pad_token_id: int,
        padded_width: int | None = None,
    ):
        self.config = config
        self.layout = BlockDiffusionLayout(
            data,
            block_size=config.schedule.block_size,
            pad_token_id=pad_token_id,
            padded_width=padded_width,
        )
        positions = self.layout.base["original_positions"]
        self.levels = data["dopsd_reveal_levels"].gather(1, positions)
        self.levels = self.levels.masked_fill(~self.layout.response_mask, -1)
        self.loss_mask = (
            data["dopsd_loss_mask"].gather(1, positions) & self.layout.response_mask
        )
        self.seeds = data["dopsd_seeds"]
        blocks = self.levels.reshape(data.size, -1, config.schedule.block_size)
        active = self.loss_mask.reshape_as(blocks)
        self.transition_counts = (
            torch.stack(
                [
                    ((blocks == level) & active).any(-1)
                    for level in range(config.sampling.max_steps)
                ]
            )
            .sum((0, 2))
            .clamp_min(1)
        )
        super().__init__(num_steps=config.sampling.max_steps, num_samples=data.size)

    def generate_single_trajectory(self, time_step: int) -> BatchedDataDict[Any]:
        if not 0 <= time_step < self.num_steps:
            raise IndexError(time_step)
        result = BatchedDataDict(self.layout.base)
        result["sample_mask"] = self.layout.response_mask.any(1).float()
        result["input_lengths"] = torch.full_like(
            self.seeds, result["input_ids"].shape[1]
        )
        student_mask = self.layout.canvas_mask & (
            (self.levels < 0) | (self.levels >= time_step)
        )
        committed = self.loss_mask & (self.levels == time_step)
        teacher_mask = student_mask.clone()
        block_size = self.config.schedule.block_size
        for row in range(self.num_samples):
            generator = torch.Generator().manual_seed(
                (int(self.seeds[row]) + time_step * 10_007) & 0x7FFFFFFFFFFFFFFF
            )
            for start in range(0, committed.shape[1], block_size):
                end = start + block_size
                candidates = (
                    student_mask[row, start:end]
                    & self.layout.response_mask[row, start:end]
                    & (self.levels[row, start:end] >= 0)
                ).nonzero().flatten() + start
                count = int(committed[row, start:end].sum())
                # Teacher selection needs k still-masked positions, including the final step.
                available = int(
                    (
                        student_mask[row, start:end] & self.loss_mask[row, start:end]
                    ).sum()
                )
                reveal_count = (
                    min(
                        int(candidates.numel() * self.config.teacher_retain_ratio),
                        max(available - count, 0),
                    )
                    if count
                    else 0
                )
                if reveal_count:
                    # Only reveal scored tokens so the k-mask guarantee also holds around EOS.
                    candidates = candidates[self.loss_mask[row, candidates]]
                    indices = torch.randperm(candidates.numel(), generator=generator)[
                        :reveal_count
                    ]
                    teacher_mask[row, candidates[indices.to(candidates.device)]] = False
        result["masked_indices"] = student_mask
        result["dopsd_teacher_mask"] = teacher_mask
        result["dopsd_committed"] = committed
        result["dopsd_eligible"] = student_mask & teacher_mask & self.loss_mask
        result["dopsd_transition_counts"] = self.transition_counts
        result["token_mask"] = committed.float()
        return result


def prepare_teacher_targets(
    level: BatchedDataDict[Any],
    teacher_logits: torch.Tensor,
    *,
    block_size: int,
    selection_source: Literal["teacher", "student"],
) -> BatchedDataDict[Any]:
    """Select k positions per block and retain detached full-vocabulary targets."""
    selected = level["dopsd_committed"].clone()
    scores_fp32 = teacher_logits.float()
    confidence = scores_fp32.max(-1).values - scores_fp32.logsumexp(-1)
    weights = torch.zeros_like(level["token_mask"], dtype=torch.float32)
    for row in range(level.size):
        for start in range(0, selected.shape[1], block_size):
            end = start + block_size
            count = int(level["dopsd_committed"][row, start:end].sum())
            if not count:
                continue
            if selection_source == "teacher":
                eligible = level["dopsd_eligible"][row, start:end]
                if int(eligible.sum()) < count:
                    raise ValueError("Teacher must retain at least k masked positions")
                scores = confidence[row, start:end].masked_fill(~eligible, -torch.inf)
                chosen = scores.topk(count).indices + start
                selected[row, start:end] = False
                selected[row, chosen] = True
            weights[row, start:end] = selected[row, start:end].float() / (
                level["dopsd_transition_counts"][row] * count
            )
    result = BatchedDataDict(level)
    result["token_mask"] = weights
    # Megatron's microbatch iterator requires every tensor's dim 1 to be the
    # sequence dimension. Retain only one microbatch/level, never a full trace.
    result["dopsd_target_weights"] = weights
    result["dopsd_teacher_logits"] = teacher_logits.detach().clone()
    return result
