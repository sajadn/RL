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
"""Full-canvas left-to-right reveal schedule for original LLaDA."""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from block_diffusion.denoising_schedule import DenoisingSchedule
from block_diffusion.full_canvas import FullCanvasLayout

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class LLaDAJustGRPOSchedule(DenoisingSchedule):
    """One scored position per row; every future canvas position stays masked."""

    def __init__(self, data: BatchedDataDict[Any], *, canvas_length: int) -> None:
        self.layout = FullCanvasLayout(data, canvas_length=canvas_length)
        super().__init__(
            num_steps=canvas_length, num_samples=data["input_ids"].shape[0]
        )

    def generate_single_trajectory(self, time_step: int) -> BatchedDataDict[Any]:
        if not 0 <= time_step < self.num_steps:
            raise ValueError("Reveal step outside the full canvas")
        level = BatchedDataDict(self.layout.base)
        positions = level["position_ids"]
        current = level["prompt_lengths"][:, None] + time_step
        level["masked_indices"] = self.layout.canvas_mask & (positions >= current)
        level["token_mask"] = self.layout.response_mask & (positions == current)
        return level


def prepare_llada_training_data(
    data: BatchedDataDict[Any],
    message_logs: list[LLMMessageLogType],
    step: int,
    *,
    canvas_length: int,
) -> None:
    """Exclude appended environment feedback from bidirectional model inputs."""
    if len(message_logs) != data.size:
        raise ValueError("LLaDA message logs must align with training rows")
    ends = []
    for row, messages in enumerate(message_logs):
        offset = 0
        spans = []
        for message in messages:
            length = len(message["token_ids"])
            if message["role"] == "assistant":
                spans.append((offset, offset + length))
            offset += length
        if len(spans) != 1:
            raise ValueError("LLaDA expects exactly one generated assistant message")
        start, end = spans[0]
        if end - start != canvas_length or end > int(data["input_lengths"][row]):
            raise ValueError("LLaDA rollout must retain its complete generated canvas")
        mask = data["token_mask"][row]
        if mask[:start].any() or mask[end:].any():
            raise ValueError("LLaDA loss mask extends outside the generated message")
        ends.append(end)
    data["input_lengths"] = torch.tensor(
        ends, dtype=data["input_lengths"].dtype, device=data["input_lengths"].device
    )


def make_llada_training_data_preparer(
    stop_token_ids: list[int], *, canvas_length: int
) -> Callable[[BatchedDataDict[Any], list[LLMMessageLogType], int], None]:
    """Bind the canvas geometry to the native controller's preparation hook."""
    if stop_token_ids:
        raise ValueError("LLaDA full-canvas rollouts cannot stop early")
    return partial(prepare_llada_training_data, canvas_length=canvas_length)
