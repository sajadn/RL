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
    build_leftmost_reveal_loss_batch,
)
from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _loss_fn() -> ClippedPGLossFn:
    return ClippedPGLossFn(
        {
            "reference_policy_kl_penalty": 0.05,
            "reference_policy_kl_type": "k1",
            "kl_input_clamp_value": None,
            "kl_output_clamp_value": None,
            "ratio_clip_min": 0.2,
            "ratio_clip_max": 0.2,
            "ratio_clip_c": None,
            "use_on_policy_kl_approximation": False,
            "use_importance_sampling_correction": False,
            "truncated_importance_sampling_ratio": None,
            "token_level_loss": True,
        }
    )


def test_aligned_clipped_pg_loss_matches_sequence_loss_for_reveal_tokens():
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
            "input_lengths": torch.tensor([4, 4]),
            "token_mask": torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]]),
            "sample_mask": torch.tensor([1, 1]),
            "advantages": torch.tensor([[0.0, 1.0, -0.5, 0.0], [0.0, 0.0, 0.7, 1.2]]),
            "prev_logprobs": torch.tensor(
                [[0.0, -0.2, -0.3, 0.0], [0.0, 0.0, -0.4, -0.5]]
            ),
            "generation_logprobs": torch.tensor(
                [[0.0, -0.21, -0.31, 0.0], [0.0, 0.0, -0.41, -0.51]]
            ),
            "reference_policy_logprobs": torch.tensor(
                [[0.0, -0.25, -0.35, 0.0], [0.0, 0.0, -0.45, -0.55]]
            ),
        }
    )
    sequence_curr_logprobs = torch.tensor(
        [
            [-0.18, -0.33, 0.0],
            [0.0, -0.39, -0.48],
        ]
    )
    global_valid_toks = torch.tensor(4.0)
    global_valid_seqs = torch.tensor(2.0)

    sequence_loss_fn = _loss_fn()
    sequence_loss, sequence_metrics = sequence_loss_fn(
        next_token_logprobs=sequence_curr_logprobs,
        data=data,
        global_valid_seqs=global_valid_seqs,
        global_valid_toks=global_valid_toks,
    )

    flat_data = build_leftmost_reveal_loss_batch(data, mask_token_id=99)
    flat_curr_logprobs = sequence_curr_logprobs[
        flat_data["just_grpo_batch_indices"],
        flat_data["just_grpo_target_positions"] - 1,
    ]
    flat_loss_fn = _loss_fn()
    flat_loss, flat_metrics = flat_loss_fn.compute_from_aligned_tensors(
        curr_logprobs=flat_curr_logprobs,
        token_mask=torch.ones_like(flat_data["just_grpo_loss_mask"]),
        sample_mask=flat_data["just_grpo_loss_mask"],
        advantages=flat_data["advantages"],
        prev_logprobs=flat_data["prev_logprobs"],
        generation_logprobs=flat_data["generation_logprobs"],
        reference_policy_logprobs=flat_data["reference_policy_logprobs"],
        global_valid_seqs=global_valid_seqs,
        global_valid_toks=global_valid_toks,
    )

    torch.testing.assert_close(flat_loss, sequence_loss)
    for metric_name in (
        "probs_ratio",
        "kl_penalty",
        "token_mult_prob_error",
        "gen_kl_error",
        "policy_kl_error",
        "js_divergence_error",
        "sampling_importance_ratio",
        "approx_entropy",
    ):
        assert abs(flat_metrics[metric_name] - sequence_metrics[metric_name]) < 1e-6
