# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partially masked denoising loss on positive AR GRPO completions."""

from typing import Any

import torch

from block_diffusion.positive_diffusion import (
    PositiveDiffusionConfig,
    positive_samples,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


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
    positive = positive_samples(data, config, valid)
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
