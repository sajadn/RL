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

"""Unit tests for HybridARDiffusionLossFn.

The two terms are exercised in isolation (by zeroing the other's weight or
inputs) and then together, so a regression in one cannot hide behind the other.
"""

import math

import pytest
import torch

from nemo_rl.algorithms.loss.loss_functions import HybridARDiffusionLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# Layout: 4 positions -- 0,1 are the noisy half (CE), 2,3 the clean half (PG).
NOISY = [0, 1]
CLEAN = [2, 3]


def _cfg(**overrides):
    cfg = {
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.2,
        "ratio_clip_c": None,
        "token_level_loss": True,
        "ce_loss_weight": 1.0,
    }
    cfg.update(overrides)
    return cfg


def _data(
    advantages=(0.0, 0.0, 1.0, 1.0),
    prev=(0.0, 0.0, -1.0, -1.0),
    ce_mask=(1.0, 1.0, 0.0, 0.0),
    pg_mask=(0.0, 0.0, 1.0, 1.0),
    mask_ratio=0.5,
):
    return BatchedDataDict(
        {
            "advantages": torch.tensor([advantages], dtype=torch.float32),
            "prev_logprobs": torch.tensor([prev], dtype=torch.float32),
            "hybrid_ce_mask": torch.tensor([ce_mask], dtype=torch.float32),
            "hybrid_pg_mask": torch.tensor([pg_mask], dtype=torch.float32),
            "hybrid_mask_ratio": torch.tensor([mask_ratio], dtype=torch.float32),
            "sample_mask": torch.ones(1, dtype=torch.float32),
        }
    )


def _run(loss_fn, logprobs, data, global_valid_toks=2.0, global_valid_seqs=1.0):
    return loss_fn(
        torch.tensor([logprobs], dtype=torch.float32),
        data,
        torch.tensor(global_valid_seqs),
        torch.tensor(global_valid_toks),
    )


def test_both_terms_use_the_global_normalizer_not_local_mask_sums():
    # With global_valid_toks deliberately != the local mask sums, a term that
    # normalized by its own mask sum would give a different number. This is what
    # makes the sum-over-microbatches convention reconstruct the true mean.
    loss_fn = HybridARDiffusionLossFn(_cfg(ce_loss_weight=1.0))
    data = _data()
    _, m = _run(
        loss_fn, [-2.0, -2.0, -1.0, -1.0], data, global_valid_toks=8.0
    )
    # PG: 2 positions, A=1, ratio=1 -> sum(-1,-1) = -2, /8 = -0.25
    assert m["pg_loss"] == pytest.approx(-0.25)
    # CE: 2 masked positions at logprob -2 -> sum(2,2) = 4, /8 = 0.5
    assert m["ce_loss"] == pytest.approx(0.5)


def test_ce_and_pg_use_their_own_masks_not_each_others():
    # Unequal mask sizes: swapping the two masks would change both numbers.
    loss_fn = HybridARDiffusionLossFn(_cfg(ce_loss_weight=1.0))
    data = _data(
        advantages=(0.0, 0.0, 1.0, 1.0),
        ce_mask=(1.0, 0.0, 0.0, 0.0),  # 1 CE position
        pg_mask=(0.0, 0.0, 1.0, 1.0),  # 2 PG positions
    )
    _, m = _run(loss_fn, [-3.0, -7.0, -1.0, -1.0], data, global_valid_toks=4.0)
    assert m["num_ce_tokens"] == pytest.approx(1.0)
    assert m["num_pg_tokens"] == pytest.approx(2.0)
    # CE reads only position 0 (-3 -> 3.0), not position 1 (-7).
    assert m["ce_loss"] == pytest.approx(3.0 / 4.0)
    assert m["pg_loss"] == pytest.approx(-2.0 / 4.0)


def test_on_policy_ratio_is_one_and_pg_equals_neg_advantage():
    # curr == prev on the clean half -> ratio 1 -> actor_loss = -A.
    loss_fn = HybridARDiffusionLossFn(_cfg(ce_loss_weight=0.0))
    data = _data()
    loss, metrics = _run(loss_fn, [0.0, 0.0, -1.0, -1.0], data)
    # Two PG positions, A=1 each, normalized by global_valid_toks=2.
    assert metrics["pg_loss"] == pytest.approx(-1.0)
    assert loss.item() == pytest.approx(-1.0)
    assert metrics["ratio_clipped_fraction"] == pytest.approx(0.0)
    assert metrics["approx_kl"] == pytest.approx(0.0)


def test_ce_term_only_counts_masked_noisy_positions():
    # Zero advantages -> PG term vanishes; only CE remains.
    loss_fn = HybridARDiffusionLossFn(_cfg(ce_loss_weight=1.0))
    data = _data(advantages=(0.0, 0.0, 0.0, 0.0))
    # logprobs -2 on the two CE positions, garbage on the clean half.
    loss, metrics = _run(loss_fn, [-2.0, -2.0, -99.0, -99.0], data)
    # CE = sum(2.0, 2.0) / global_valid_toks(2) = 2.0
    assert metrics["ce_loss"] == pytest.approx(2.0)
    assert metrics["pg_loss"] == pytest.approx(0.0)
    assert loss.item() == pytest.approx(2.0)
    # The clean-half garbage must not leak into CE.
    assert metrics["num_ce_tokens"] == pytest.approx(2.0)


def test_ce_weight_scales_only_the_ce_term():
    data = _data(advantages=(0.0, 0.0, 0.0, 0.0))
    a, ma = _run(HybridARDiffusionLossFn(_cfg(ce_loss_weight=1.0)),
                 [-2.0, -2.0, 0.0, 0.0], data)
    b, mb = _run(HybridARDiffusionLossFn(_cfg(ce_loss_weight=0.25)),
                 [-2.0, -2.0, 0.0, 0.0], data)
    assert ma["ce_loss"] == pytest.approx(mb["ce_loss"])
    assert b.item() == pytest.approx(0.25 * a.item())


def test_elbo_weighting_scales_ce_by_inverse_t():
    logprobs = [-2.0, -2.0, 0.0, 0.0]
    data_plain = _data(advantages=(0.0,) * 4, mask_ratio=0.5)
    data_elbo = _data(advantages=(0.0,) * 4, mask_ratio=0.5)
    _, m_plain = _run(HybridARDiffusionLossFn(_cfg()), logprobs, data_plain)
    _, m_elbo = _run(
        HybridARDiffusionLossFn(_cfg(elbo_weight_ce=True)), logprobs, data_elbo
    )
    # t = 0.5 -> 1/t = 2x
    assert m_elbo["ce_loss"] == pytest.approx(2.0 * m_plain["ce_loss"])


def test_positive_advantage_ratio_is_clipped_above():
    # curr - prev = +1.0 -> ratio e ~= 2.718, clipped to 1.2 for A > 0.
    loss_fn = HybridARDiffusionLossFn(_cfg(ce_loss_weight=0.0))
    data = _data()
    _, metrics = _run(loss_fn, [0.0, 0.0, 0.0, 0.0], data)
    # max(-A*r, -A*r_clamped) with A=1 picks the *less negative* -> -1.2
    assert metrics["pg_loss"] == pytest.approx(-1.2)
    assert metrics["ratio_clipped_fraction"] == pytest.approx(1.0)


def test_negative_advantage_uses_unclipped_when_larger():
    loss_fn = HybridARDiffusionLossFn(_cfg(ce_loss_weight=0.0))
    data = _data(advantages=(0.0, 0.0, -1.0, -1.0))
    # ratio ~= 2.718; -A*r = +2.718 vs -A*r_clamped = +1.2 -> max picks 2.718
    _, metrics = _run(loss_fn, [0.0, 0.0, 0.0, 0.0], data)
    assert metrics["pg_loss"] == pytest.approx(math.e, rel=1e-4)


def test_dual_clip_bounds_negative_advantage_branch():
    loss_fn = HybridARDiffusionLossFn(_cfg(ce_loss_weight=0.0, ratio_clip_c=2.0))
    data = _data(advantages=(0.0, 0.0, -1.0, -1.0))
    # Unclipped branch would give e ~= 2.718; dual clip caps at -A*c = 2.0.
    _, metrics = _run(loss_fn, [0.0, 0.0, 0.0, 0.0], data)
    assert metrics["pg_loss"] == pytest.approx(2.0)


def test_dual_clip_leaves_positive_advantage_untouched():
    plain = HybridARDiffusionLossFn(_cfg(ce_loss_weight=0.0))
    dual = HybridARDiffusionLossFn(_cfg(ce_loss_weight=0.0, ratio_clip_c=2.0))
    data_a, data_b = _data(), _data()
    _, ma = _run(plain, [0.0, 0.0, 0.0, 0.0], data_a)
    _, mb = _run(dual, [0.0, 0.0, 0.0, 0.0], data_b)
    assert ma["pg_loss"] == pytest.approx(mb["pg_loss"])


def test_both_terms_sum():
    loss_fn = HybridARDiffusionLossFn(_cfg(ce_loss_weight=0.5))
    data = _data()
    loss, metrics = _run(loss_fn, [-2.0, -2.0, -1.0, -1.0], data)
    assert loss.item() == pytest.approx(
        metrics["pg_loss"] + 0.5 * metrics["ce_loss"]
    )
    assert metrics["weighted_ce_loss"] == pytest.approx(0.5 * metrics["ce_loss"])


def test_sample_mask_zeroes_a_sample():
    loss_fn = HybridARDiffusionLossFn(_cfg())
    data = _data()
    data["sample_mask"] = torch.zeros(1, dtype=torch.float32)
    loss, metrics = _run(loss_fn, [-2.0, -2.0, -1.0, -1.0], data)
    assert loss.item() == pytest.approx(0.0)
    assert metrics["num_pg_tokens"] == pytest.approx(0.0)
    assert metrics["num_ce_tokens"] == pytest.approx(0.0)


def test_sequence_level_loss_normalizes_per_sample():
    loss_fn = HybridARDiffusionLossFn(
        _cfg(ce_loss_weight=0.0, token_level_loss=False)
    )
    data = _data()
    _, metrics = _run(loss_fn, [0.0, 0.0, -1.0, -1.0], data)
    # Per-sample mean of actor_loss over its 2 PG tokens = -1.0, then averaged
    # over global_valid_seqs = 1.
    assert metrics["pg_loss"] == pytest.approx(-1.0)


def test_metrics_expose_both_terms_for_lambda_tuning():
    loss_fn = HybridARDiffusionLossFn(_cfg())
    _, metrics = _run(loss_fn, [-2.0, -2.0, -1.0, -1.0], _data())
    for key in (
        "pg_loss",
        "ce_loss",
        "weighted_ce_loss",
        "ratio_clipped_fraction",
        "approx_kl",
        "num_pg_tokens",
        "num_ce_tokens",
        "mean_mask_ratio",
    ):
        assert key in metrics
    assert metrics["mean_mask_ratio"] == pytest.approx(0.5)
