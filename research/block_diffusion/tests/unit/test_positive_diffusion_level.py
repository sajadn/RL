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
"""The positive diffusion level reuses the algorithm canvas and AR normalization."""

import pytest
import torch

from block_diffusion.positive_diffusion import (
    PositiveDiffusionLoss,
    build_positive_diffusion_level,
    resolve_positive_diffusion_config,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

STOP = 9
MASK = 31


def rollout_batch():
    # Prompts occupy 0..3. Row 0 stops at position 6; 7..8 follow the stop.
    ids = torch.tensor(
        [
            [1, 2, 3, 4, 11, 12, STOP, 13, 14, 0, 0, 0],
            [1, 2, 3, 4, 21, 22, 23, 24, 25, 26, STOP, 0],
            [1, 2, 3, 4, 31, 32, STOP, 0, 0, 0, 0, 0],
        ]
    )
    lengths = torch.tensor([9, 11, 7])
    token_mask = torch.zeros_like(ids, dtype=torch.float32)
    for row, length in enumerate(lengths.tolist()):
        token_mask[row, 4:length] = 1
    return BatchedDataDict(
        input_ids=ids,
        input_lengths=lengths,
        token_mask=token_mask,
        sample_mask=torch.tensor([1.0, 1.0, 0.0]),
        advantages=token_mask * torch.tensor([[1.0], [-1.0], [1.0]]),
    )


def config(**overrides):
    return resolve_positive_diffusion_config(
        {"seed": 0, **overrides}, mask_token_id=MASK, block_size=4
    )


def build(cfg, seed=0, data=None):
    return build_positive_diffusion_level(
        rollout_batch() if data is None else data,
        cfg,
        pad_token_id=0,
        padded_width=16,
        stop_token_ids=[STOP],
        generator=torch.Generator().manual_seed(seed),
    )


def test_full_masking_targets_positive_responses_through_the_stop_token():
    level, response_tokens = build(
        config(mask_probability_min=1.0, mask_probability_max=1.0)
    )
    assert level["target_ids"].shape == (3, 16)
    # Row 0: positive, scored through the stop; row 1 negative; row 2 excluded.
    assert level["token_mask"][0].nonzero().flatten().tolist() == [0, 1, 2]
    assert not level["token_mask"][1:].any()
    assert level["diffusion_positive_samples"].tolist() == [True, False, False]
    # Post-stop tokens and final-block padding stay MASK context.
    assert level["masked_indices"][0].nonzero().flatten().tolist() == list(range(8))
    # Negative rows keep every scorable token visible.
    assert level["masked_indices"][1].nonzero().flatten().tolist() == [7]
    assert not level["masked_indices"][2].any()
    # The denominator counts every included completion through its stop token.
    assert int(response_tokens) == 3 + 7


def test_partial_masks_are_reproducible_and_respect_sampled_probabilities():
    cfg = config(mask_probability_min=0.05, mask_probability_max=1.0)
    first, _ = build(cfg, seed=3)
    second, _ = build(cfg, seed=3)
    torch.testing.assert_close(first["token_mask"], second["token_mask"])
    probabilities = first["diffusion_mask_probability"]
    assert ((probabilities >= 0.05) & (probabilities <= 1.0)).all()
    assert first["input_lengths"].tolist() == [16, 16, 16]


def test_inverse_probability_weighted_ce():
    level, _ = build(config(mask_probability_min=0.5, mask_probability_max=0.5))
    level["token_mask"][0, :3] = True
    logprobs = -torch.ones(level["token_mask"].shape)
    loss = PositiveDiffusionLoss(config(weight=0.1))
    value, metrics = loss(
        level,
        torch.tensor(1.0),
        torch.tensor(1.0),
        next_token_logprobs=logprobs,
        shift_labels=False,
    )
    torch.testing.assert_close(value, torch.tensor(0.1 * 3 / 0.5))
    assert metrics["diffusion_aux_masked_tokens"] == 3


def test_reward_selection_requires_rewards():
    with pytest.raises(ValueError, match="requires rewards"):
        build(config(positive_samples="reward"))


def test_auxiliary_config_must_match_the_algorithm_canvas():
    assert config().mask_token_id == MASK
    with pytest.raises(ValueError, match="MASK token and block size"):
        resolve_positive_diffusion_config(
            {"block_size": 16}, mask_token_id=MASK, block_size=4
        )

