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
"""Full-vocabulary reverse KL with vocabulary-entry pointwise clipping."""

from typing import Any

import torch

from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class DOPSDLoss:
    """Use weighted tokens to implement mean-over-steps, then mean-over-responses."""

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGIT

    def __init__(self, *, pointwise_clip: float | None):
        self.pointwise_clip = pointwise_clip

    def __call__(
        self,
        data: BatchedDataDict[Any],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        *,
        logits: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        weights = data["dopsd_target_weights"] * data["sample_mask"][:, None]
        active = weights > 0
        student = logits[active].float().log_softmax(-1)
        teacher = data["dopsd_teacher_logits"][active].detach().float().log_softmax(-1)
        pointwise = student.exp() * (student - teacher)
        raw = pointwise.sum(-1)
        if self.pointwise_clip is not None:
            clipped = (pointwise > self.pointwise_clip).float().mean(-1)
            pointwise = pointwise.clamp(max=self.pointwise_clip)
        else:
            clipped = torch.zeros_like(raw)
        scale = weights[active] / global_valid_toks.clamp_min(1)
        loss = (pointwise.sum(-1) * scale).sum()
        # Empty selections preserve a differentiable zero for distributed backward.
        return loss, {
            "loss": loss.item(),
            "dopsd_reverse_kl": (raw * scale).sum().item(),
            "dopsd_clip_fraction": (clipped * scale).sum().item(),
        }
