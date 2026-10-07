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
"""Recorded reveal-state extraction shared by trajectory algorithms."""

from typing import Any

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def recorded_reveal_states(
    data: BatchedDataDict[Any],
    message_logs: list[list[dict]],
    *,
    stop_token_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dense block-relative reveal levels and the through-stop loss mask."""
    ids = data["input_ids"]
    steps = torch.full_like(ids, -1)
    for row, messages in enumerate(message_logs):
        offset = 0
        for message in messages:
            length = len(message["token_ids"])
            if "reveal_steps" in message:
                recorded = message["reveal_steps"]
                if len(recorded) != length:
                    raise ValueError("Reveal steps must align with generated tokens")
                steps[row, offset : offset + length] = recorded.to(steps.device)
            offset += length
    response = data["token_mask"].bool()
    response &= (
        torch.arange(ids.shape[1], device=ids.device)[None]
        < data["input_lengths"][:, None]
    )
    stops = torch.isin(ids, torch.tensor(stop_token_ids, device=ids.device)) & response
    after_stop = (stops.long().cumsum(1) - stops.long()) > 0
    loss_mask = response & ~after_stop
    included = data["sample_mask"].bool()[:, None]
    if ((steps < 0) & loss_mask & included).any():
        raise ValueError(
            "Replay requires a recorded reveal step for every scored response token"
        )
    levels = torch.full_like(steps, -1)
    for row in range(ids.shape[0]):
        committed = response[row] & (steps[row] >= 0)
        # Reveal steps restart in each block. A sampled Trace level
        # must score that step across all response blocks together.
        distinct = torch.unique(steps[row, committed], sorted=True)
        levels[row, committed] = torch.searchsorted(distinct, steps[row, committed])
    return levels, loss_mask
