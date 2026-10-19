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
"""Replay every Fork-dLLM fallback decision using its recorded masked state."""

from typing import Any, Self

import torch
from block_diffusion.config import DiffusionExperimentConfig
from block_diffusion.replay import recorded_reveal_states
from pydantic import model_validator

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class ForkGRPOConfig(DiffusionExperimentConfig):
    @model_validator(mode="after")
    def validate_fork(self) -> Self:
        if self.runtime not in {"megatron", "reference_vllm"}:
            raise ValueError("ForkGRPO requires Megatron or reference vLLM generation")
        selection_policy = (
            "fork" if self.runtime == "megatron" else "confidence_threshold_leftmost"
        )
        if self.sampling.temperature <= 0:
            raise ValueError("Training temperature must be positive")
        for sampling in (self.sampling, self.validation_sampling):
            if sampling.selection_policy != selection_policy:
                raise ValueError(f"ForkGRPO requires {selection_policy} sampling")
            if sampling.block_size != self.schedule.block_size:
                raise ValueError("Fork sampling and training block sizes must match")
            if sampling.max_steps < sampling.block_size:
                raise ValueError("Fork decoding requires max_steps >= block_size")
            if not sampling.returns_reveal_steps or not sampling.emit_full_blocks:
                raise ValueError(
                    "ForkGRPO requires reveal steps and full terminal blocks"
                )
            if sampling.returns_entropy:
                raise ValueError("ForkGRPO uses fallback flags, not token entropy")
        return self


class ForkGRPO:
    """Retain rewards for every rollout and select only recorded fallback tokens."""

    def __init__(self, config: ForkGRPOConfig, *, stop_token_ids: list[int]) -> None:
        self.config = config
        self.stop_token_ids = stop_token_ids

    def prepare_training_data(
        self, data: BatchedDataDict[Any], message_logs: list[list[dict]], step: int
    ) -> None:
        levels, loss_mask = recorded_reveal_states(
            data, message_logs, stop_token_ids=self.stop_token_ids
        )
        fallback = torch.zeros_like(loss_mask)
        for row, messages in enumerate(message_logs):
            offset = 0
            for message in messages:
                length = len(message["token_ids"])
                if "reveal_steps" in message:
                    if "fallback_mask" not in message:
                        raise ValueError("ForkGRPO requires explicit fallback metadata")
                    flags = message["fallback_mask"]
                    if len(flags) != length:
                        raise ValueError(
                            "Fallback mask must align with generated tokens"
                        )
                    fallback[row, offset : offset + length] = flags.to(fallback.device)
                offset += length
        if (fallback & (levels < 0)).any():
            raise ValueError("Fallback tokens must have recorded reveal steps")
        fallback &= loss_mask
        # Levels restart per block. Replay matching levels across blocks together;
        # the shared asymmetric attention supplies clean earlier blocks.
        selected = [
            torch.unique(levels[row, fallback[row]], sorted=True)
            for row in range(data.size)
        ]
        views = max(1, max(len(values) for values in selected))
        sampled = torch.full(
            (data.size, views),
            self.config.sampling.max_steps,
            device=levels.device,
            dtype=torch.long,
        )
        for row, values in enumerate(selected):
            sampled[row, : len(values)] = values
        data["trace_reveal_levels"] = levels
        data["trace_loss_mask"] = fallback
        data["trace_sampled_levels"] = sampled
        data["trace_loss_weights"] = fallback.sum(1).clamp_min(1).float().reciprocal()
