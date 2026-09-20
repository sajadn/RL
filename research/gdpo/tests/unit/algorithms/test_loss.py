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
"""Check the combined-score objective and exclusion of unavailable diagnostics."""

import math

import pytest
import torch
from gdpo.algorithms.elbo import SdmcElboEstimator
from gdpo.algorithms.loss.gdpo import GDPOLossFn
from gdpo.config import SdmcLikelihoodConfig
from gdpo.models.automodel.schedule import accumulate_schedule_logprobs

from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def make_loss():
    return GDPOLossFn(
        ClippedPGLossConfig(
            reference_policy_kl_penalty=0.0,
            sequence_level_importance_ratios=True,
            token_level_loss=False,
            position_aligned_logprobs=True,
        )
    )


@pytest.mark.parametrize("advantage", [-1.0, 1.0])
@pytest.mark.parametrize("ratio", [0.5, 1.05, 1.5])
def test_sequence_clipping_ignores_generation_placeholders(advantage, ratio):
    previous = torch.tensor([[0.0, -2.0, -3.0]])
    current = (previous + math.log(ratio)).requires_grad_()
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0]]),
            "sample_mask": torch.ones(1),
            "advantages": torch.full((1, 3), advantage),
            "prev_logprobs": previous,
            "generation_logprobs": torch.full((1, 3), float("nan")),
        }
    )
    loss, metrics = make_loss()(current, data, torch.tensor(1.0), torch.tensor(2.0))
    expected = max(-advantage * ratio, -advantage * min(max(ratio, 0.8), 1.2))
    assert loss.item() == pytest.approx(expected)
    assert metrics["probs_ratio"] == pytest.approx(ratio)
    assert (
        not {
            "token_mult_prob_error",
            "gen_kl_error",
            "policy_kl_error",
            "js_divergence_error",
            "sampling_importance_ratio",
            "approx_entropy",
        }
        & metrics.keys()
    )
    loss.backward()
    assert torch.isfinite(current.grad).all()
    assert current.grad[0, 0] == 0


def test_schedule_combines_before_nonlinear_loss_and_preserves_gradients():
    estimator = SdmcElboEstimator(
        SdmcLikelihoodConfig(quadrature="gauss-3", mc_samples=2), 9
    )
    clean = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[False, True, True, True]])
    weights = torch.randn(10, 10, dtype=torch.float64, requires_grad=True)

    def score(masked, targets):
        return (
            weights[masked]
            .log_softmax(-1)
            .gather(-1, targets.unsqueeze(-1))
            .squeeze(-1)
        )

    combined = accumulate_schedule_logprobs(
        estimator, input_ids=clean, completion_mask=mask, seed=42, score_fn=score
    )
    reference = sum(
        estimator.accumulate(point, score(point.input_ids, clean))
        for point in estimator.mask_points(clean, mask, seed=42)
    )
    torch.testing.assert_close(combined, reference)
    (actual_gradient,) = torch.autograd.grad(
        combined.mean().exp(), weights, retain_graph=True
    )
    (reference_gradient,) = torch.autograd.grad(reference.mean().exp(), weights)
    torch.testing.assert_close(actual_gradient, reference_gradient)
    assert actual_gradient.abs().sum() > 0
