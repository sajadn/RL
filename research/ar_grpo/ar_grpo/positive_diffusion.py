# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partially masked denoising loss on positive AR GRPO completions."""

from typing import Any, Literal, Self

import torch
from pydantic import BaseModel, Field, model_validator

from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType, MetricNormalizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class PositiveDiffusionConfig(BaseModel, extra="forbid"):
    """Settings for the optional positive-completion auxiliary objective."""

    weight: float = Field(default=0.1, ge=0, allow_inf_nan=False)
    positive_samples: Literal["advantage", "reward"] = "advantage"
    mask_token_id: int = Field(ge=0)
    block_size: int = Field(gt=0)
    mask_probability_min: float = Field(default=0.05, gt=0, le=1)
    mask_probability_max: float = Field(default=1.0, gt=0, le=1)
    seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def validate_probabilities(self) -> Self:
        if self.mask_probability_min > self.mask_probability_max:
            raise ValueError(
                "mask_probability_min must not exceed mask_probability_max"
            )
        return self


def build_positive_diffusion_batch(
    data: BatchedDataDict[Any],
    config: PositiveDiffusionConfig,
    *,
    pad_token_id: int,
    pad_multiple: int,
    generator: torch.Generator,
) -> BatchedDataDict[Any]:
    """Mask positive responses in a full-sequence [noisy | clean] layout.

    Keeping the original sequence grid also supports multiple assistant spans.
    Prompts and environment feedback remain visible, but are never targets.
    Draws on CPU make selection identical across tensor-parallel ranks.
    Final-block padding is MASK context, but is excluded from the loss mask.
    """
    ids = data["input_ids"]
    lengths = data["input_lengths"].long()
    valid = data["token_mask"].bool() & (
        torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None]
    )
    valid = valid.clone()
    # Match the AR target denominator, which excludes column zero.
    valid[:, 0] = False
    key = "advantages" if config.positive_samples == "advantage" else "rewards"
    if key not in data:
        raise ValueError(f"Positive diffusion selection requires {key} in the batch")
    scores = data[key]
    if scores.ndim == 2:
        scores = (scores * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    if scores.shape != (ids.shape[0],):
        raise ValueError(f"Expected per-sample or per-token {key}")
    positive = (scores > 0) & (data["sample_mask"] > 0)
    width = (ids.shape[1] + pad_multiple - 1) // pad_multiple * pad_multiple
    noisy_targets = (
        torch.nn.functional.pad(ids, (0, width - ids.shape[1]), value=pad_token_id)
        if width > ids.shape[1]
        else ids
    )
    eligible = torch.nn.functional.pad(
        valid & positive[:, None], (0, width - ids.shape[1])
    )
    probabilities = config.mask_probability_min + (
        config.mask_probability_max - config.mask_probability_min
    ) * torch.rand(ids.shape[0], generator=generator)
    masked = (
        torch.rand(ids.shape[0], width, generator=generator) < probabilities[:, None]
    ).to(ids.device) & eligible
    positions = torch.arange(width, device=ids.device)[None].expand_as(noisy_targets)
    noisy_valid_lengths = (
        (lengths + config.block_size - 1) // config.block_size * config.block_size
    )
    # Match JustGRPO/Trace: tail slots in the final block remain MASK context.
    # Keep the random target selection separate so padding never contributes CE.
    tail = (
        (positions >= lengths[:, None])
        & (positions < noisy_valid_lengths[:, None])
        & data["sample_mask"].bool()[:, None]
    )
    # Keep the original AR IDs, masks, advantages, and rollout/reference logprobs.
    # Only the noisy canvas needs block alignment; NeMo-RL owns ordinary padding.
    combined = BatchedDataDict(data)
    combined.update(
        target_ids=noisy_targets,
        clean_input_ids=ids,
        prompt_lengths=torch.zeros_like(lengths),
        response_lengths=lengths,
        noisy_valid_lengths=noisy_valid_lengths,
        clean_lengths=lengths,
        position_ids=positions,
        masked_indices=masked | tail,
        diffusion_loss_mask=masked,
        sample_mask=data["sample_mask"],
        diffusion_mask_probability=probabilities.to(ids.device),
        diffusion_positive_samples=positive,
    )
    return combined


class PositiveDiffusionLoss:
    """Inverse-probability weighted masked CE, normalized with the AR loss.

    L_aux = weight * sum_positive sum_masked -log p(x_i | noisy) / p_mask / N_AR.
    N_AR includes all valid AR response tokens, including negative completions.
    This keeps the coefficient independent of masking probability and streaming
    chunk boundaries; fewer positive completions reduce the auxiliary strength.
    """

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB
    metric_normalizations = {
        "diffusion_aux_loss": MetricNormalizer.TOKENS,
        "diffusion_aux_masked_tokens": MetricNormalizer.NONE,
        "diffusion_aux_positive_samples": MetricNormalizer.NONE,
    }

    def __init__(self, config: PositiveDiffusionConfig) -> None:
        self.weight = config.weight

    def __call__(
        self,
        data: BatchedDataDict[Any],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        *,
        next_token_logprobs: torch.Tensor,
        shift_labels: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if shift_labels:
            raise ValueError("Diffusion CE requires same-position targets")
        weights = data["token_mask"] * data["sample_mask"][:, None]
        weighted_ce = (
            -next_token_logprobs * weights / data["diffusion_mask_probability"][:, None]
        )
        loss = self.weight * weighted_ce.sum() / global_valid_toks.clamp_min(1)
        return loss, {
            "loss": loss.item(),
            "diffusion_aux_loss": loss.item(),
            "diffusion_aux_masked_tokens": weights.sum().item(),
            "diffusion_aux_positive_samples": data["diffusion_positive_samples"]
            .sum()
            .item(),
        }
