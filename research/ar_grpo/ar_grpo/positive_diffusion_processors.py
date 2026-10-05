# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute AR GRPO and positive diffusion CE from one asymmetric forward."""

import math
from dataclasses import replace
from typing import Any, Callable

import torch
from block_diffusion.diffusion_processors import (
    DiffusionLossPostProcessor,
    DiffusionMicrobatchProcessor,
)

from ar_grpo.positive_diffusion import (
    PositiveDiffusionConfig,
    PositiveDiffusionLoss,
    build_positive_diffusion_batch,
)
from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.data import (
    ProcessedMicrobatch,
    _get_non_packed_sequence_pad_factor,
)
from nemo_rl.models.megatron.train import LossPostProcessor, apply_temperature_scaling
from nemo_rl.models.policy import PolicyConfig


class PositiveDiffusionMicrobatchProcessor(DiffusionMicrobatchProcessor):
    """Build positive-completion masks before shared diffusion preprocessing."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        config: PolicyConfig,
        pad_token_id: int,
        data_parallel_rank: int,
    ) -> None:
        self.config = PositiveDiffusionConfig.model_validate(
            config["diffusion_aux_loss"]
        )
        super().__init__(model=model, mask_token_id=self.config.mask_token_id)
        self.pad_token_id = pad_token_id
        self.pad_multiple = math.lcm(
            self.config.block_size, _get_non_packed_sequence_pad_factor(config)
        )
        self.generator = torch.Generator(device="cpu").manual_seed(
            self.config.seed + data_parallel_rank
        )

    def __call__(self, microbatch: ProcessedMicrobatch) -> ProcessedMicrobatch:
        # Scoring and weight-zero training keep the ordinary AR batch unchanged.
        if "diffusion_aux_enabled" not in microbatch.data_dict:
            return microbatch
        data = build_positive_diffusion_batch(
            microbatch.data_dict,
            self.config,
            pad_token_id=self.pad_token_id,
            pad_multiple=self.pad_multiple,
            generator=self.generator,
        )
        return super().__call__(replace(microbatch, data_dict=data))


class PositiveDiffusionLossPostProcessor(LossPostProcessor):
    """Reuse upstream loss processing independently for the clean and noisy halves."""

    def __init__(
        self,
        *,
        cfg: PolicyConfig,
        sampling_params: TrainingSamplingParams | None = None,
        **kwargs: Any,
    ) -> None:
        # Paired execution disables global logit scaling. Recover AR sampling
        # from the same generation config used by the native worker constructor.
        generation = cfg.get("generation")
        if sampling_params is None and generation is not None:
            sampling_params = TrainingSamplingParams(
                top_k=generation["top_k"],
                top_p=generation["top_p"],
                temperature=generation["temperature"],
            )
        super().__init__(cfg=cfg, sampling_params=sampling_params, **kwargs)
        diffusion_config = PositiveDiffusionConfig.model_validate(
            cfg["diffusion_aux_loss"]
        )
        self.diffusion_processor = DiffusionLossPostProcessor(
            loss_fn=PositiveDiffusionLoss(diffusion_config),
            cfg=self.cfg,
            num_microbatches=self.num_microbatches,
            cp_normalize=self.cp_normalize,
        )

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        packed_seq_params: Any = None,
        global_valid_seqs: torch.Tensor | None = None,
        global_valid_toks: torch.Tensor | None = None,
    ) -> Callable:
        if "masked_indices" not in data_dict:
            return super().__call__(
                data_dict=data_dict,
                packed_seq_params=packed_seq_params,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
            )
        if packed_seq_params is not None:
            raise ValueError(
                "Positive diffusion processors require unpacked microbatches"
            )
        process_ar = super().__call__(
            data_dict=data_dict,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
        )
        auxiliary = BatchedDataDict(data_dict)
        auxiliary["token_mask"] = data_dict["masked_indices"]
        process_diffusion = self.diffusion_processor(
            data_dict=auxiliary,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
        )
        noisy_width = data_dict["target_ids"].shape[1]
        clean_width = data_dict["clean_input_ids"].shape[1]

        def process(logits: torch.Tensor) -> tuple[torch.Tensor, dict]:
            clean_logits = logits[:, noisy_width : noisy_width + clean_width]
            if (
                self.sampling_params is not None
                and self.sampling_params.temperature != 1
            ):
                # Scaling is in place: isolate this view from the noisy CE branch.
                clean_logits = clean_logits.clone()
            clean_logits = apply_temperature_scaling(clean_logits, self.sampling_params)
            ar_loss, ar_metrics = process_ar(clean_logits)
            diffusion_loss, diffusion_metrics = process_diffusion(logits)
            return ar_loss + diffusion_loss, {
                **ar_metrics,
                **diffusion_metrics,
                "loss": ar_metrics["loss"] + diffusion_metrics["loss"],
            }

        return process
