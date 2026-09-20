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
"""GDPO uses the shared clipped objective with ELBO current/previous scores."""

from typing import Any

import torch

from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn


class GDPOLossFn(ClippedPGLossFn):
    """Omit metrics and corrections requiring denoising sampling probabilities."""

    def __init__(
        self, cfg: ClippedPGLossConfig, *, use_fused_linear_logprobs: bool = False
    ):
        if (
            cfg.use_importance_sampling_correction
            or cfg.reference_policy_kl_penalty
            or cfg.use_kl_in_reward
            or not cfg.position_aligned_logprobs
            or not cfg.sequence_level_importance_ratios
            or cfg.token_level_loss
        ):
            raise ValueError(
                "GDPO requires aligned sequence ratios, with sampler correction and reference KL disabled"
            )
        super().__init__(cfg, use_fused_linear_logprobs=use_fused_linear_logprobs)
        for name in (
            "token_mult_prob_error",
            "gen_kl_error",
            "policy_kl_error",
            "js_divergence_error",
            "sampling_importance_ratio",
            "approx_entropy",
        ):
            self.metric_normalizations.pop(name)

    def _generation_statistics(
        self, *, prev_logprobs: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return torch.ones_like(prev_logprobs), {}
