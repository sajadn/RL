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
"""Configuration types and validation; experiment defaults live in YAML."""

from typing import Self

from omegaconf import DictConfig, OmegaConf
from pydantic import Field, model_validator

from block_diffusion.config import (
    BlockDiffusionConfig,
    DiffusionExperimentConfig,
    DiffusionSamplingParams as DiffusionSamplingParams,
)
from block_diffusion.validation import validate_experiment


class ScheduleConfig(BlockDiffusionConfig, extra="allow"):
    reveal_tokens_per_step: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_width(self) -> Self:
        if self.reveal_tokens_per_step > self.block_size:
            raise ValueError("reveal_tokens_per_step cannot exceed block_size")
        return self


class JustGRPOConfig(DiffusionExperimentConfig):
    """Diffusion-specific extension to the standard NeMo-RL configuration."""

    training_token_fraction: float = Field(gt=0, le=1)
    schedule: ScheduleConfig

    @model_validator(mode="after")
    def validate_training_distribution(self) -> Self:
        s, p = self.schedule, self.sampling
        if p.temperature <= 0:
            raise ValueError("Training temperature must be positive")
        if p.block_size != s.block_size or p.selection_policy != "leftmost":
            raise ValueError("Training requires matching blocks and leftmost sampling")
        if s.block_size % s.reveal_tokens_per_step:
            raise ValueError(
                "Reference vLLM needs block_size divisible by reveal width"
            )
        if p.max_steps != s.block_size // s.reveal_tokens_per_step:
            raise ValueError("Generation steps must match the training reveal schedule")
        if self.training_token_fraction < 1 and s.reveal_tokens_per_step != 1:
            raise ValueError("Fast partial views require one token per reveal")
        for params in (p, self.validation_sampling):
            if (
                params.returns_entropy
                or params.returns_reveal_steps
                or params.emit_full_blocks
            ):
                raise ValueError(
                    "Generation does not yet transport entropy/reveal channels"
                )
        if self.runtime not in ("reference_vllm", "megatron"):
            raise ValueError(
                "Upstream generation cannot yet replay leftmost reveals; use megatron or reference_vllm"
            )
        return self


def validate_config(config: DictConfig) -> JustGRPOConfig:
    """Parse JustGRPO settings and validate the shared experiment configuration."""
    diffusion = JustGRPOConfig.model_validate(
        OmegaConf.to_container(config.just_grpo, resolve=True)
    )
    validate_experiment(config, diffusion)
    return diffusion
