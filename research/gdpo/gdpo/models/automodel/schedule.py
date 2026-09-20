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
"""Masked forward inputs and generic differentiable score accumulation."""

from dataclasses import dataclass
from typing import Callable, Iterator, Protocol

import torch


@dataclass
class DenoisingStep:
    """A masked forward, its clean scoring targets, and estimator weights."""

    input_ids: torch.Tensor
    target_ids: torch.Tensor
    score_mask: torch.Tensor
    weight: float | torch.Tensor


class DenoisingSchedule(Protocol):
    """Supply individual masked evaluations before loss preparation."""

    def return_schedule(
        self,
        input_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        *,
        seed: int | torch.Tensor | None,
    ) -> Iterator[DenoisingStep]: ...


def accumulate_schedule_logprobs(
    schedule: DenoisingSchedule,
    *,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    seed: int | torch.Tensor | None,
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Retain every forward's graph and combine scores before the policy loss."""
    total: torch.Tensor | None = None
    for step in schedule.return_schedule(input_ids, completion_mask, seed=seed):
        logprobs = score_fn(step.input_ids, step.target_ids)
        # Exclude unscored positions before multiplying (including filtered -inf).
        contribution = step.weight * logprobs.masked_fill(~step.score_mask, 0.0)
        total = contribution if total is None else total + contribution
    if total is None:
        raise ValueError("A denoising schedule must contain at least one forward")
    return total
