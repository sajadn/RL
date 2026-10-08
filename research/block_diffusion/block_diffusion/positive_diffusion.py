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
"""Partially masked denoising loss on positive GRPO completions."""

from typing import Any, Literal, Self

import torch
from pydantic import BaseModel, Field, model_validator

from block_diffusion.block_layout import BlockDiffusionLayout
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


def positive_samples(
    data: BatchedDataDict[Any], config: PositiveDiffusionConfig, valid: torch.Tensor
) -> torch.Tensor:
    """Select included rows whose advantage or reward is positive."""
    key = "advantages" if config.positive_samples == "advantage" else "rewards"
    if key not in data:
        raise ValueError(f"Positive diffusion selection requires {key} in the batch")
    scores = data[key]
    if scores.ndim == 2:
        scores = (scores * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    if scores.shape != (data["input_ids"].shape[0],):
        raise ValueError(f"Expected per-sample or per-token {key}")
    return (scores > 0) & (data["sample_mask"] > 0)


def build_positive_diffusion_level(
    data: BatchedDataDict[Any],
    config: PositiveDiffusionConfig,
    *,
    pad_token_id: int,
    padded_width: int,
    stop_token_ids: list[int],
    generator: torch.Generator,
) -> tuple[BatchedDataDict[Any], torch.Tensor]:
    """Render one uniformly masked level on a diffusion algorithm's canvas.

    The level reuses the response-relative layout of JustGRPO/Trace levels, so
    it has the same fixed shape. Responses are scored through the first stop
    token; later tokens and final-block padding remain MASK context. Every row
    is kept so all ranks execute the same forward/backward schedule.
    Returns the level and the local count of scorable response tokens over all
    included completions, the denominator shared with the AR auxiliary loss.
    """
    layout = BlockDiffusionLayout(
        data,
        block_size=config.block_size,
        pad_token_id=pad_token_id,
        padded_width=padded_width,
    )
    level = BatchedDataDict(layout.base)
    targets = level["target_ids"]
    response = layout.response_mask
    stops = torch.isin(
        targets,
        torch.tensor(stop_token_ids, dtype=targets.dtype, device=targets.device),
    )
    stops &= response
    after_stop = (stops.long().cumsum(1) - stops.long()) > 0
    scorable = response & ~after_stop
    ids = data["input_ids"]
    valid = data["token_mask"].bool() & (
        torch.arange(ids.shape[1], device=ids.device)[None]
        < data["input_lengths"][:, None]
    )
    positive = positive_samples(data, config, valid)
    probabilities = config.mask_probability_min + (
        config.mask_probability_max - config.mask_probability_min
    ) * torch.rand(ids.shape[0], generator=generator)
    masked = (
        torch.rand(targets.shape, generator=generator) < probabilities[:, None]
    ).to(targets.device) & scorable & positive[:, None]
    level["masked_indices"] = layout.canvas_mask & (~scorable | masked)
    level["token_mask"] = masked
    level["sample_mask"] = data["sample_mask"]
    level["input_lengths"] = torch.full_like(
        level["sample_indices"], targets.shape[1]
    )
    level["diffusion_mask_probability"] = probabilities.to(targets.device)
    level["diffusion_positive_samples"] = positive
    return level, scorable.sum()


class PositiveDiffusionLoss:
    """Inverse-probability weighted masked CE, normalized with the policy loss.

    L_aux = weight * sum_positive sum_masked -log p(x_i | noisy) / p_mask / N.
    N counts all valid response tokens, including negative completions.
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


def resolve_positive_diffusion_config(
    settings: dict[str, Any], *, mask_token_id: int, block_size: int
) -> PositiveDiffusionConfig:
    """Fill the diffusion algorithm's MASK token and block size into the config."""
    config = PositiveDiffusionConfig.model_validate(
        {"mask_token_id": mask_token_id, "block_size": block_size, **settings}
    )
    if (config.mask_token_id, config.block_size) != (mask_token_id, block_size):
        raise ValueError(
            "diffusion_aux_loss must use the algorithm's MASK token and block size"
        )
    return config
