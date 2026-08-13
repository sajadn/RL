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

"""Megatron loss post-processor for hybrid AR + diffusion training."""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from megatron.core.packed_seq_params import PackedSeqParams

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.diffu_grpo_train import (
    _cp_sharded_same_position_logprobs,
)
from nemo_rl.models.megatron.train import LogprobsPostProcessor, LossPostProcessor
from nemo_rl.models.policy import PolicyConfig


class HybridARDiffusionLossPostProcessor(LossPostProcessor):
    """Score both halves of the ``[noisy | clean]`` layout with one gather.

    ``hybrid_target_ids`` already encodes each half's alignment -- the noisy half
    stores the token at its own position, the clean half stores the *next* clean
    token -- so a single same-position gather produces the ``[N, T]`` logprob
    vector both loss terms read. Unlike ``DiffuGRPOLossPostProcessor`` the result
    is *not* truncated to the noisy half: the clean half carries the policy
    gradient term.
    """

    def __init__(
        self,
        loss_fn: LossFunction,
        cfg: PolicyConfig,
        num_microbatches: int = 1,
        cp_normalize: bool = True,
        sampling_params: Optional[TrainingSamplingParams] = None,
    ):
        super().__init__(
            loss_fn=loss_fn,
            cfg=cfg,
            num_microbatches=num_microbatches,
            cp_normalize=cp_normalize,
            sampling_params=sampling_params,
            draft_model=None,
        )

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        packed_seq_params: Optional[PackedSeqParams] = None,
        global_valid_seqs: Optional[torch.Tensor] = None,
        global_valid_toks: Optional[torch.Tensor] = None,
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, Any]]]:
        """Build the per-microbatch loss closure.

        Args:
            data_dict: The hybrid training microbatch.
            packed_seq_params: Must be ``None`` -- sequence packing would break
                the fixed ``[noisy | clean]`` split the attention metadata and
                both loss masks depend on.
            global_valid_seqs: Global valid sequence count.
            global_valid_toks: Global valid (response) token count; normalizes
                both loss terms.

        Returns:
            A callable mapping the model output tensor to ``(loss, metrics)``.

        Raises:
            NotImplementedError: If sequence packing is enabled.
        """
        if self.cfg["sequence_packing"]["enabled"] or packed_seq_params is not None:
            raise NotImplementedError(
                "Hybrid AR+diffusion Megatron training requires "
                "sequence_packing.enabled=false"
            )

        def loss_fn_inner(
            output_tensor: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            logprob_estimation_cfg = self.cfg["logprob_estimation"]
            mask_token_id = logprob_estimation_cfg["mask_token_id"]
            exclude_mask = logprob_estimation_cfg.get(
                "exclude_mask_token_from_logits", None
            )
            # One gather serves both halves. The MASK id is excluded from the
            # scored logits on both -- the clean (AR) half can never legitimately
            # target MASK either, and applying it uniformly keeps the training
            # logprobs consistent with prev_logprobs, which take this same path.
            token_logprobs = _cp_sharded_same_position_logprobs(
                output_tensor,
                data_dict["hybrid_target_ids"],
                cfg=self.cfg,
                sampling_params=self.sampling_params,
                inference_only=False,
                exclude_token_id=mask_token_id if exclude_mask else None,
            )
            loss, metrics = self.loss_fn(
                token_logprobs,
                data_dict,
                global_valid_seqs,
                global_valid_toks,
            )
            # Megatron's forward-backward divides each microbatch loss by
            # num_microbatches. The loss is already globally normalized (both
            # terms divide by global_valid_toks), so that extra division has to
            # be cancelled or gradients come out num_microbatches times too
            # small. Every sibling post-processor does the same.
            return loss * self.num_microbatches, metrics

        return loss_fn_inner


class HybridARDiffusionLogprobsPostProcessor(LogprobsPostProcessor):
    """No-grad post-processor producing full-sequence hybrid logprobs.

    Differs from ``DiffuGRPOLogprobsPostProcessor`` in two ways: it gathers
    against ``hybrid_target_ids`` (which carries the clean half's next-token
    shift) and it does *not* truncate to the noisy half, because the caller wants
    the clean-half autoregressive logprobs.
    """

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        input_ids: torch.Tensor,
        cu_seqlens_padded: torch.Tensor,
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Build the per-microbatch logprob closure.

        Args:
            data_dict: The hybrid microbatch.
            input_ids: Unused; kept for interface compatibility.
            cu_seqlens_padded: Unused; sequence packing is not supported.

        Returns:
            A callable mapping the model output tensor to
            ``(zero_loss, {"logprobs": [N, T]})``.
        """

        def processor_fn_inner(output_tensor):
            logprob_estimation_cfg = self.cfg["logprob_estimation"]
            mask_token_id = logprob_estimation_cfg["mask_token_id"]
            exclude_mask = logprob_estimation_cfg.get(
                "exclude_mask_token_from_logits", None
            )
            token_logprobs = _cp_sharded_same_position_logprobs(
                output_tensor,
                data_dict["hybrid_target_ids"],
                cfg=self.cfg,
                sampling_params=self.sampling_params,
                inference_only=True,
                exclude_token_id=mask_token_id if exclude_mask else None,
            )
            return torch.tensor(0.0, device=token_logprobs.device), {
                "logprobs": token_logprobs
            }

        return processor_fn_inner
