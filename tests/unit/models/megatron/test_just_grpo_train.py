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

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams

pytestmark = pytest.mark.mcore


def _base_cfg(**overrides):
    cfg = {"sequence_packing": {"enabled": False}}
    cfg.update(overrides)
    return cfg


class _AlignedRecordingLoss:
    def __init__(self, loss, metrics=None, reference_policy_kl_penalty=0.0):
        self.loss = loss
        self.metrics = metrics or {}
        self.reference_policy_kl_penalty = reference_policy_kl_penalty
        self.calls = []

    def compute_from_aligned_tensors(self, **kwargs):
        self.calls.append(kwargs)
        return self.loss, self.metrics


class TestJustGRPOLogprobsPostProcessor:
    def test_logprobs_processor_uses_same_position_targets(self):
        from nemo_rl.models.megatron.just_grpo_train import (
            JustGRPOLogprobsPostProcessor,
        )

        sampling_params = TrainingSamplingParams(top_k=None, top_p=1.0)
        processor = JustGRPOLogprobsPostProcessor(
            cfg=_base_cfg(logprob_chunk_size=13),
            sampling_params=sampling_params,
        )

        data_dict = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "just_grpo_target_positions": torch.tensor([2, 0]),
            "just_grpo_target_tokens": torch.tensor([11, 12]),
        }
        output_tensor = torch.randn(2, 4, 7)
        expected_logprobs = torch.tensor([-0.25, -1.50])

        with (
            patch(
                "nemo_rl.models.megatron.just_grpo_train.get_tensor_model_parallel_group",
                return_value=MagicMock(),
            ) as mock_tp_group,
            patch(
                "nemo_rl.models.megatron.just_grpo_train.get_tensor_model_parallel_rank",
                return_value=3,
            ),
            patch(
                "nemo_rl.models.megatron.just_grpo_train.from_parallel_logits_to_same_position_logprobs",
                return_value=expected_logprobs,
            ) as mock_from_logits,
        ):
            wrapped_fn = processor(
                data_dict=data_dict,
                input_ids=data_dict["input_ids"],
                cu_seqlens_padded=torch.tensor([0, 4, 8]),
            )
            loss, result = wrapped_fn(output_tensor)

        assert loss.item() == 0.0
        assert loss.device == output_tensor.device
        assert result["logprobs"] is expected_logprobs

        mock_from_logits.assert_called_once()
        call = mock_from_logits.call_args
        assert call.args == (output_tensor,)
        assert torch.equal(
            call.kwargs["target_positions"], data_dict["just_grpo_target_positions"]
        )
        assert torch.equal(
            call.kwargs["target_tokens"], data_dict["just_grpo_target_tokens"]
        )
        assert call.kwargs["vocab_start_index"] == 3 * output_tensor.shape[-1]
        assert call.kwargs["vocab_end_index"] == 4 * output_tensor.shape[-1]
        assert call.kwargs["tp_group"] is mock_tp_group.return_value
        assert call.kwargs["inference_only"] is True
        assert call.kwargs["chunk_size"] == 13
        assert call.kwargs["sampling_params"] is sampling_params

    def test_logprobs_processor_masks_filtered_logprobs(self):
        from nemo_rl.models.megatron.just_grpo_train import (
            JustGRPOLogprobsPostProcessor,
        )

        processor = JustGRPOLogprobsPostProcessor(
            cfg=_base_cfg(),
            sampling_params=TrainingSamplingParams(top_p=0.8),
        )
        data_dict = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "just_grpo_target_positions": torch.tensor([1]),
            "just_grpo_target_tokens": torch.tensor([2]),
        }
        raw_logprobs = torch.tensor([-float("inf")])
        masked_logprobs = torch.tensor([0.0])

        with (
            patch(
                "nemo_rl.models.megatron.just_grpo_train.get_tensor_model_parallel_group",
                return_value=MagicMock(),
            ),
            patch(
                "nemo_rl.models.megatron.just_grpo_train.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "nemo_rl.models.megatron.just_grpo_train.from_parallel_logits_to_same_position_logprobs",
                return_value=raw_logprobs,
            ),
            patch(
                "nemo_rl.models.megatron.just_grpo_train.mask_out_neg_inf_logprobs",
                return_value=masked_logprobs,
            ) as mock_mask,
        ):
            _, result = processor(
                data_dict=data_dict,
                input_ids=data_dict["input_ids"],
                cu_seqlens_padded=torch.tensor([0, 3]),
            )(torch.randn(1, 3, 5))

        assert result["logprobs"] is masked_logprobs
        mock_mask.assert_called_once()
        mask_call = mock_mask.call_args
        assert mask_call.args[0] is raw_logprobs
        assert torch.equal(mask_call.args[1], torch.ones_like(raw_logprobs))
        assert mask_call.args[2] == "prev_logprobs"


class TestJustGRPOLossPostProcessor:
    def test_loss_processor_passes_aligned_tensors_to_loss(self):
        from nemo_rl.models.megatron.just_grpo_train import JustGRPOLossPostProcessor

        loss_fn = _AlignedRecordingLoss(torch.tensor(0.75), {"loss": 0.75})
        processor = JustGRPOLossPostProcessor(
            loss_fn=loss_fn,
            cfg=_base_cfg(logprob_chunk_size=9),
            num_microbatches=4,
        )

        data_dict = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            "just_grpo_target_positions": torch.tensor([1, 2, 0]),
            "just_grpo_target_tokens": torch.tensor([42, 43, 44]),
            "just_grpo_loss_mask": torch.tensor([1.0, 0.0, 1.0]),
            "advantages": torch.tensor([0.5, -0.25, 1.25]),
            "prev_logprobs": torch.tensor([-0.1, -0.2, -0.3]),
            "generation_logprobs": torch.tensor([-0.11, -0.22, -0.33]),
            "reference_policy_logprobs": torch.tensor([-0.15, -0.25, -0.35]),
        }
        output_tensor = torch.randn(3, 3, 8)
        curr_logprobs = torch.tensor([-0.2, -1.4, -9.0])
        global_valid_seqs = torch.tensor(2)
        global_valid_toks = torch.tensor(6)

        with (
            patch(
                "nemo_rl.models.megatron.just_grpo_train.get_context_parallel_world_size",
                return_value=1,
            ),
            patch(
                "nemo_rl.models.megatron.just_grpo_train.get_tensor_model_parallel_group",
                return_value=MagicMock(),
            ) as mock_tp_group,
            patch(
                "nemo_rl.models.megatron.just_grpo_train.get_tensor_model_parallel_rank",
                return_value=2,
            ),
            patch(
                "nemo_rl.models.megatron.just_grpo_train.from_parallel_logits_to_same_position_logprobs",
                return_value=curr_logprobs,
            ) as mock_from_logits,
        ):
            wrapped_fn = processor(
                data_dict=data_dict,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
            )
            loss, metrics = wrapped_fn(output_tensor)

        assert torch.isclose(loss, torch.tensor(3.0))
        assert metrics == {"loss": 0.75}
        assert len(loss_fn.calls) == 1

        call = loss_fn.calls[0]
        assert call["curr_logprobs"] is curr_logprobs
        assert torch.equal(call["token_mask"], torch.ones_like(curr_logprobs))
        assert call["sample_mask"] is data_dict["just_grpo_loss_mask"]
        assert call["advantages"] is data_dict["advantages"]
        assert call["prev_logprobs"] is data_dict["prev_logprobs"]
        assert call["generation_logprobs"] is data_dict["generation_logprobs"]
        assert call["reference_policy_logprobs"] is data_dict["reference_policy_logprobs"]
        assert call["curr_logprobs_unfiltered"] is None
        assert call["global_valid_seqs"] is global_valid_seqs
        assert call["global_valid_toks"] is global_valid_toks

        mock_from_logits.assert_called_once()
        from_logits_call = mock_from_logits.call_args
        assert from_logits_call.args == (output_tensor,)
        assert from_logits_call.kwargs["tp_group"] is mock_tp_group.return_value
        assert (
            from_logits_call.kwargs["vocab_start_index"]
            == 2 * output_tensor.shape[-1]
        )
        assert (
            from_logits_call.kwargs["vocab_end_index"] == 3 * output_tensor.shape[-1]
        )
        assert from_logits_call.kwargs["inference_only"] is False
        assert from_logits_call.kwargs["chunk_size"] == 9

    def test_loss_processor_masks_filtered_logprobs_and_keeps_unfiltered_for_kl(self):
        from nemo_rl.models.megatron.just_grpo_train import JustGRPOLossPostProcessor

        sampling_params = TrainingSamplingParams(top_k=2)
        loss_fn = _AlignedRecordingLoss(
            torch.tensor(1.0),
            reference_policy_kl_penalty=0.1,
        )
        processor = JustGRPOLossPostProcessor(
            loss_fn=loss_fn,
            cfg=_base_cfg(),
            sampling_params=sampling_params,
        )

        data_dict = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "just_grpo_target_positions": torch.tensor([1, 2]),
            "just_grpo_target_tokens": torch.tensor([20, 21]),
            "just_grpo_loss_mask": torch.tensor([1.0, 0.0]),
            "advantages": torch.tensor([0.5, -0.25]),
            "prev_logprobs": torch.tensor([-0.1, -0.2]),
            "generation_logprobs": torch.tensor([-0.11, -0.22]),
        }
        output_tensor = torch.randn(2, 3, 6)
        filtered_logprobs = torch.tensor([-float("inf"), -0.8])
        masked_filtered_logprobs = torch.tensor([0.0, -0.8])
        unfiltered_logprobs = torch.tensor([-0.3, -0.4])

        with (
            patch(
                "nemo_rl.models.megatron.just_grpo_train.get_context_parallel_world_size",
                return_value=1,
            ),
            patch(
                "nemo_rl.models.megatron.just_grpo_train.get_tensor_model_parallel_group",
                return_value=MagicMock(),
            ),
            patch(
                "nemo_rl.models.megatron.just_grpo_train.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "nemo_rl.models.megatron.just_grpo_train.from_parallel_logits_to_same_position_logprobs",
                side_effect=[filtered_logprobs, unfiltered_logprobs],
            ) as mock_from_logits,
            patch(
                "nemo_rl.models.megatron.just_grpo_train.mask_out_neg_inf_logprobs",
                return_value=masked_filtered_logprobs,
            ) as mock_mask,
        ):
            wrapped_fn = processor(data_dict=data_dict)
            wrapped_fn(output_tensor)

        assert mock_from_logits.call_count == 2
        assert (
            mock_from_logits.call_args_list[0].kwargs["sampling_params"]
            is sampling_params
        )
        assert mock_from_logits.call_args_list[1].kwargs["sampling_params"] is None
        mock_mask.assert_called_once()
        mask_call = mock_mask.call_args
        assert mask_call.args[0] is filtered_logprobs
        assert mask_call.args[1] is data_dict["just_grpo_loss_mask"]
        assert mask_call.args[2] == "curr_logprobs"

        call = loss_fn.calls[0]
        assert call["curr_logprobs"] is masked_filtered_logprobs
        assert call["curr_logprobs_unfiltered"] is unfiltered_logprobs
        assert data_dict["curr_logprobs_unfiltered"] is unfiltered_logprobs

    def test_loss_processor_requires_aligned_loss_function(self):
        from nemo_rl.models.megatron.just_grpo_train import JustGRPOLossPostProcessor

        class NonAlignedLoss:
            reference_policy_kl_penalty = 0.0

        processor = JustGRPOLossPostProcessor(
            loss_fn=NonAlignedLoss(),
            cfg=_base_cfg(),
        )

        data_dict = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "just_grpo_target_positions": torch.tensor([1]),
            "just_grpo_target_tokens": torch.tensor([2]),
            "just_grpo_loss_mask": torch.tensor([1.0]),
        }

        with (
            patch(
                "nemo_rl.models.megatron.just_grpo_train.get_context_parallel_world_size",
                return_value=1,
            ),
            patch(
                "nemo_rl.models.megatron.just_grpo_train.get_tensor_model_parallel_group",
                return_value=MagicMock(),
            ),
            patch(
                "nemo_rl.models.megatron.just_grpo_train.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "nemo_rl.models.megatron.just_grpo_train.from_parallel_logits_to_same_position_logprobs",
                return_value=torch.tensor([-0.1]),
            ),
        ):
            with pytest.raises(TypeError, match="requires ClippedPGLossFn"):
                processor(data_dict=data_dict)(torch.randn(1, 3, 5))

    @pytest.mark.parametrize(
        ("cfg", "packed_seq_params"),
        [
            ({"sequence_packing": {"enabled": True}}, None),
            (_base_cfg(), MagicMock()),
        ],
    )
    def test_loss_processor_rejects_sequence_packing(self, cfg, packed_seq_params):
        from nemo_rl.models.megatron.just_grpo_train import JustGRPOLossPostProcessor

        processor = JustGRPOLossPostProcessor(
            loss_fn=_AlignedRecordingLoss(torch.tensor(1.0)),
            cfg=cfg,
        )

        with pytest.raises(NotImplementedError, match="sequence_packing.enabled=false"):
            processor(
                data_dict={"input_ids": torch.ones(1, 3, dtype=torch.long)},
                packed_seq_params=packed_seq_params,
            )

    def test_loss_processor_rejects_context_parallelism(self):
        from nemo_rl.models.megatron.just_grpo_train import JustGRPOLossPostProcessor

        processor = JustGRPOLossPostProcessor(
            loss_fn=_AlignedRecordingLoss(torch.tensor(1.0)),
            cfg=_base_cfg(),
        )

        with patch(
            "nemo_rl.models.megatron.just_grpo_train.get_context_parallel_world_size",
            return_value=2,
        ):
            with pytest.raises(NotImplementedError, match="context_parallel_size=1"):
                processor(data_dict={"input_ids": torch.ones(1, 3, dtype=torch.long)})
