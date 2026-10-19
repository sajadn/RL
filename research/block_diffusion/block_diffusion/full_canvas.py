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
"""Absolute-position, bidirectional full-canvas diffusion inputs."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

if TYPE_CHECKING:
    from nemo_rl.models.megatron.data import ProcessedMicrobatch


class FullCanvasLayout:
    """Keep the prompt followed by the entire response canvas, without a clean half."""

    def __init__(self, data: BatchedDataDict[Any], *, canvas_length: int) -> None:
        ids, lengths = data["input_ids"], data["input_lengths"].long()
        positions = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
        starts = lengths - canvas_length
        if (starts < 0).any() or (lengths > ids.shape[1]).any():
            raise ValueError(
                f"Full-canvas replay requires prompt plus the complete response: "
                f"lengths={lengths.tolist()}, width={ids.shape[1]}, canvas={canvas_length}, "
                f"sample_mask={data['sample_mask'].tolist()}"
            )
        canvas = (positions >= starts[:, None]) & (positions < lengths[:, None])
        valid = data["token_mask"].bool()
        if (valid & ~canvas).any():
            raise ValueError(
                "Full-canvas token_mask must select only response positions"
            )
        included = data["sample_mask"].bool()[:, None]
        self.response_mask = canvas & valid & included
        self.canvas_mask = canvas
        self.base = BatchedDataDict(
            input_ids=ids,
            target_ids=ids,
            position_ids=positions,
            original_positions=positions,
            sample_indices=torch.arange(ids.shape[0], device=ids.device),
            prompt_lengths=starts,
            canvas_lengths=lengths,
            token_mask=self.response_mask,
        )


class FullCanvasMicrobatchProcessor:
    """Mask inputs in place and exclude padding from bidirectional attention."""

    def __init__(self, *, model: torch.nn.Module, mask_token_id: int) -> None:
        self.model, self.mask_token_id = model, mask_token_id

    def __call__(self, microbatch: ProcessedMicrobatch) -> ProcessedMicrobatch:
        if microbatch.packed_seq_params is not None:
            raise ValueError("LLaDA full-canvas training requires unpacked sequences")
        data = microbatch.data_dict
        width = int(data["canvas_lengths"].max())
        inputs = data["target_ids"][:, :width].masked_fill(
            data["masked_indices"][:, :width], self.mask_token_id
        )
        padding = (
            torch.arange(width, device=inputs.device)[None]
            >= data["canvas_lengths"][:, None]
        )
        modules = [m for m in self.model.modules() if hasattr(m, "set_padding_mask")]
        if not modules:
            raise RuntimeError(
                "LLaDA requires its bidirectional Bridge attention implementation"
            )
        for module in modules:
            module.set_padding_mask(padding if padding.any() else None)
        return replace(
            microbatch,
            input_ids=inputs,
            input_ids_cp_sharded=inputs,
            position_ids=data["position_ids"][:, :width],
            attention_mask=None,
            original_seq_length=width,
        )

    def clear_asymmetric_metadata(self) -> None:
        """Match the shared worker cleanup hook, including backward recomputation."""
        for module in self.model.modules():
            if hasattr(module, "reset_inference_state"):
                module.reset_inference_state()
