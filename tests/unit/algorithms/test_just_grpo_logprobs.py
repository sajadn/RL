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

import torch

from nemo_rl.algorithms.just_grpo_logprobs import (
    build_leftmost_reveal_batch,
    build_leftmost_reveal_loss_batch,
    pad_reveal_batch_to_multiple,
    scatter_leftmost_reveal_logprobs,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_build_leftmost_reveal_batch_masks_selected_future_tokens():
    input_ids = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 25],
        ]
    )
    input_lengths = torch.tensor([5, 4])
    token_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 0],
            [0, 1, 1, 0, 1, 0],
        ]
    )

    reveal_batch = build_leftmost_reveal_batch(
        input_ids=input_ids,
        input_lengths=input_lengths,
        token_mask=token_mask,
        mask_token_id=99,
    )

    torch.testing.assert_close(
        reveal_batch["input_ids"],
        torch.tensor(
            [
                [10, 11, 99, 99, 99, 15],
                [10, 11, 12, 99, 99, 15],
                [10, 11, 12, 13, 99, 15],
                [20, 99, 99, 23, 24, 25],
                [20, 21, 99, 23, 24, 25],
            ]
        ),
    )
    torch.testing.assert_close(
        reveal_batch["input_lengths"], torch.tensor([5, 5, 5, 4, 4])
    )
    torch.testing.assert_close(
        reveal_batch["just_grpo_batch_indices"], torch.tensor([0, 0, 0, 1, 1])
    )
    torch.testing.assert_close(
        reveal_batch["just_grpo_target_positions"], torch.tensor([2, 3, 4, 1, 2])
    )
    torch.testing.assert_close(
        reveal_batch["just_grpo_target_tokens"], torch.tensor([12, 13, 14, 21, 22])
    )
    torch.testing.assert_close(
        reveal_batch["just_grpo_output_shape"], torch.tensor([[2, 6]]).repeat(5, 1)
    )


def test_build_leftmost_reveal_batch_respects_sample_mask():
    input_ids = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ]
    )
    input_lengths = torch.tensor([4, 4])
    token_mask = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ]
    )

    reveal_batch = build_leftmost_reveal_batch(
        input_ids=input_ids,
        input_lengths=input_lengths,
        token_mask=token_mask,
        mask_token_id=99,
        sample_mask=torch.tensor([1, 0]),
    )

    torch.testing.assert_close(
        reveal_batch["input_ids"],
        torch.tensor(
            [
                [10, 99, 99, 13],
                [10, 11, 99, 13],
            ]
        ),
    )
    torch.testing.assert_close(
        reveal_batch["just_grpo_batch_indices"], torch.tensor([0, 0])
    )
    torch.testing.assert_close(
        reveal_batch["just_grpo_target_positions"], torch.tensor([1, 2])
    )


def test_build_leftmost_reveal_batch_fixed_response_window_masks_invalid_rows():
    input_ids = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 25],
        ]
    )
    input_lengths = torch.tensor([6, 5])
    token_mask = torch.tensor(
        [
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    reveal_batch = build_leftmost_reveal_batch(
        input_ids=input_ids,
        input_lengths=input_lengths,
        token_mask=token_mask,
        mask_token_id=99,
        reveal_schedule="fixed_response_window",
        max_reveal_positions=4,
    )

    assert reveal_batch.size == 8
    torch.testing.assert_close(
        reveal_batch["just_grpo_batch_indices"],
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
    )
    torch.testing.assert_close(
        reveal_batch["just_grpo_target_positions"],
        torch.tensor([2, 3, 4, 5, 4, 5, 5, 5]),
    )
    torch.testing.assert_close(
        reveal_batch["just_grpo_row_mask"],
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        reveal_batch["input_ids"][:2],
        torch.tensor(
            [
                [10, 11, 99, 99, 14, 15],
                [10, 11, 12, 99, 14, 15],
            ]
        ),
    )


def test_build_leftmost_reveal_loss_batch_gathers_flat_training_tensors():
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
            "input_lengths": torch.tensor([4, 4]),
            "token_mask": torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]]),
            "sample_mask": torch.tensor([1, 0]),
            "advantages": torch.tensor([[0.0, 1.0, 2.0, 0.0], [0.0, 3.0, 4.0, 5.0]]),
            "prev_logprobs": torch.tensor([[0.0, 0.1, 0.2, 0.0], [0.0, 0.3, 0.4, 0.5]]),
            "generation_logprobs": torch.tensor(
                [[0.0, 1.1, 1.2, 0.0], [0.0, 1.3, 1.4, 1.5]]
            ),
            "reference_policy_logprobs": torch.tensor(
                [[0.0, 2.1, 2.2, 0.0], [0.0, 2.3, 2.4, 2.5]]
            ),
        }
    )

    loss_batch = build_leftmost_reveal_loss_batch(data, mask_token_id=99)

    torch.testing.assert_close(
        loss_batch["input_ids"],
        torch.tensor(
            [
                [10, 99, 99, 13],
                [10, 11, 99, 13],
            ]
        ),
    )
    assert "sample_mask" not in loss_batch
    assert "token_mask" not in loss_batch
    torch.testing.assert_close(loss_batch["just_grpo_loss_mask"], torch.tensor([1, 1]))
    torch.testing.assert_close(
        loss_batch["advantages"],
        torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(
        loss_batch["prev_logprobs"],
        torch.tensor([0.1, 0.2]),
    )
    torch.testing.assert_close(
        loss_batch["generation_logprobs"],
        torch.tensor([1.1, 1.2]),
    )
    torch.testing.assert_close(
        loss_batch["reference_policy_logprobs"],
        torch.tensor([2.1, 2.2]),
    )


def test_build_leftmost_reveal_loss_batch_fixed_response_window_zeroes_invalid_rows():
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
            "input_lengths": torch.tensor([4, 4]),
            "token_mask": torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0]]),
            "sample_mask": torch.tensor([1, 1]),
            "advantages": torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
            "prev_logprobs": torch.zeros(2, 4),
            "generation_logprobs": torch.zeros(2, 4),
            "reference_policy_logprobs": torch.zeros(2, 4),
        }
    )

    loss_batch = build_leftmost_reveal_loss_batch(
        data,
        mask_token_id=99,
        reveal_schedule="fixed_response_window",
        max_reveal_positions=3,
    )

    assert loss_batch.size == 6
    assert "sample_mask" not in loss_batch
    assert "token_mask" not in loss_batch
    torch.testing.assert_close(
        loss_batch["just_grpo_loss_mask"], torch.tensor([1, 0, 0, 0, 0, 0])
    )


def test_pad_reveal_batch_to_multiple_zeros_padded_masks():
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6]]),
            "input_lengths": torch.tensor([2, 2, 2]),
            "token_mask": torch.tensor([[1, 0], [0, 1], [1, 1]]),
            "sample_mask": torch.tensor([1, 1, 1]),
        }
    )

    original_size = pad_reveal_batch_to_multiple(data, multiple=2)

    assert original_size == 3
    torch.testing.assert_close(
        data["input_ids"], torch.tensor([[1, 2], [3, 4], [5, 6], [5, 6]])
    )
    torch.testing.assert_close(data["token_mask"][3], torch.tensor([0, 0]))
    torch.testing.assert_close(data["sample_mask"], torch.tensor([1, 1, 1, 0]))


def test_scatter_leftmost_reveal_logprobs_restores_original_shape():
    logprobs = scatter_leftmost_reveal_logprobs(
        flat_logprobs=torch.tensor([0.1, 0.2, 0.3]),
        batch_indices=torch.tensor([0, 0, 1]),
        target_positions=torch.tensor([2, 4, 1]),
        output_shape=torch.tensor([[2, 6], [2, 6], [2, 6]]),
    )

    torch.testing.assert_close(
        logprobs,
        torch.tensor(
            [
                [0.0, 0.0, 0.1, 0.0, 0.2, 0.0],
                [0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_scatter_leftmost_reveal_logprobs_ignores_invalid_rows():
    logprobs = scatter_leftmost_reveal_logprobs(
        flat_logprobs=torch.tensor([0.1, 0.2]),
        batch_indices=torch.tensor([0, 0]),
        target_positions=torch.tensor([2, 2]),
        output_shape=torch.tensor([[1, 4], [1, 4]]),
        row_mask=torch.tensor([1.0, 0.0]),
    )

    torch.testing.assert_close(logprobs, torch.tensor([[0.0, 0.0, 0.1, 0.0]]))
