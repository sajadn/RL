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
"""Alignment must affect both score gathering and loss-input masking."""

import pytest
import torch

from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.utils import prepare_loss_input
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    get_cp_sharded_next_token_logprobs,
    get_next_token_logprobs_from_logits,
)


@pytest.mark.parametrize("aligned", [True, False])
def test_dense_logits_and_precomputed_scores_have_matching_alignment(aligned):
    ids = torch.tensor([[0, 1, 2]])
    logits = torch.tensor(
        [[[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]]], requires_grad=True
    )
    loss = ClippedPGLossFn(ClippedPGLossConfig(position_aligned_logprobs=aligned))
    data = BatchedDataDict({"input_ids": ids})
    prepared, _ = prepare_loss_input(logits, data, loss)
    targets = ids if aligned else ids[:, 1:]
    expected = (
        logits[:, : targets.shape[1]]
        .log_softmax(-1)
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
    )
    torch.testing.assert_close(prepared["next_token_logprobs"], expected)
    precomputed, _ = prepare_loss_input(expected, data, loss, precomputed_logprobs=True)
    torch.testing.assert_close(precomputed["next_token_logprobs"], expected)


class ReversedLayout:
    """A nontrivial canonical/local permutation, requiring no distributed runtime."""

    def shard_token_tensor(self, tensor, **kwargs):
        return tensor.flip(1)

    def gather_token_tensor(self, tensor, **kwargs):
        return tensor.flip(1)


@pytest.mark.parametrize("shift_targets", [True, False])
def test_cp_layout_uses_same_target_alignment_as_dense(shift_targets):
    ids = torch.tensor([[0, 1, 2]])
    logits = torch.randn(1, 3, 4, requires_grad=True)
    actual = get_cp_sharded_next_token_logprobs(
        logits.flip(1), ids, ReversedLayout(), shift_targets=shift_targets
    )
    expected = get_next_token_logprobs_from_logits(
        ids, logits, shift_targets=shift_targets
    )
    torch.testing.assert_close(actual, expected)
    (actual_grad,) = torch.autograd.grad(actual.sum(), logits, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected.sum(), logits)
    torch.testing.assert_close(actual_grad, expected_grad)
