# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Causal AR GRPO plus positive-completion diffusion cross-entropy."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import ray
import torch
from megatron.core import parallel_state

from ar_grpo.policy import ARModeForMultiModeMegatronPolicyImpl
from ar_grpo.positive_diffusion import (
    PositiveDiffusionConfig,
    PositiveDiffusionLoss,
)
from ar_grpo.positive_diffusion_processors import (
    PositiveDiffusionLossPostProcessor,
    PositiveDiffusionMicrobatchProcessor,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.draft_config import coerce_draft_config
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)


class PositiveDiffusionARPolicyImpl(ARModeForMultiModeMegatronPolicyImpl):
    """Compute causal policy and denoising gradients in one forward/backward."""

    def __init__(self, config: PolicyConfig, *args: Any, **kwargs: Any) -> None:
        self.diffusion_config = PositiveDiffusionConfig.model_validate(
            config["diffusion_aux_loss"]
        )
        mg = config["megatron_cfg"]
        if (
            not mg["enabled"]
            or mg["pipeline_model_parallel_size"] != 1
            or mg["context_parallel_size"] != 1
        ):
            raise ValueError("Positive diffusion AR requires Megatron PP=CP=1")
        if (
            config["sequence_packing"]["enabled"]
            or config["dynamic_batching"]["enabled"]
        ):
            raise ValueError(
                "Positive diffusion AR requires unpacked, fixed-size microbatches"
            )
        replay = config.get("router_replay")
        if mg.get("use_fused_linear_logprobs") or (
            replay is not None and replay["enabled"]
        ):
            raise ValueError(
                "Positive diffusion AR does not support fused logprobs or router replay"
            )
        draft = coerce_draft_config(config.get("draft"))
        if (draft is not None and draft.enabled) or mg.get("mtp_num_layers"):
            raise ValueError(
                "Positive diffusion AR does not support draft or MTP heads"
            )
        mg.setdefault("model_overrides", {})["block_size"] = (
            self.diffusion_config.block_size
        )
        super().__init__(
            config,
            *args,
            loss_postprocessor_factory=PositiveDiffusionLossPostProcessor,
            **kwargs,
        )
        if self.mtp_enabled:
            raise ValueError("Positive diffusion AR does not support MTP heads")
        self.microbatch_processor = PositiveDiffusionMicrobatchProcessor(
            model=self.model,
            config=self.cfg,
            pad_token_id=self.tokenizer.pad_token_id,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
        )
        self.prepare_microbatch_fn = self.microbatch_processor

    def begin_train_step(
        self, loss_fn: LossFunction, gbs: int | None = None, mbs: int | None = None
    ) -> None:
        if loss_fn.loss_type != LossType.TOKEN_LEVEL:
            raise ValueError("Positive diffusion AR requires token_level_loss=true")
        super().begin_train_step(loss_fn, gbs=gbs, mbs=mbs)
        state = self._assert_step_open()
        state["metric_normalizations"] = {
            **state["metric_normalizations"],
            **PositiveDiffusionLoss.metric_normalizations,
        }

    @contextmanager
    def _use_positive_diffusion_forward(self) -> Iterator[None]:
        # Hold diffusion attention through backward, including checkpoint recomputation.
        saved_sampling = self.sampling_params
        modules = [
            module
            for module in self.model.modules()
            if hasattr(module, "set_inference_mode")
        ]
        saved_modes = [module._inference_mode for module in modules]
        # The combined processor applies AR temperature only to the clean logits.
        self.sampling_params = None
        for module in modules:
            module.set_inference_mode(False)
        try:
            yield
        finally:
            self.sampling_params = saved_sampling
            self.microbatch_processor.clear_asymmetric_metadata()
            for module, mode in zip(modules, saved_modes):
                module.set_inference_mode(mode)

    def train_microbatch(self, data: BatchedDataDict[Any]) -> None:
        if self.diffusion_config.weight == 0:
            return super().train_microbatch(data)
        # Keep every row, including empty selections, so all DP/TP ranks execute
        # the same number of forward/backward calls and vocabulary collectives.
        # A per-sample flag keeps the batch valid for NeMo-RL's sequence-width
        # checks. The registered preparation callback builds the noisy canvas.
        training_data = BatchedDataDict(data)
        training_data["diffusion_aux_enabled"] = torch.ones_like(
            data["input_lengths"], dtype=torch.bool
        )
        with self._use_positive_diffusion_forward():
            # Bypass the AR adapter's causal-only context for the paired layout.
            # Its clean half is causal; both losses share one backward and the
            # original AR token count used by finish_train_step.
            MegatronPolicyWorkerImpl.train_microbatch(self, training_data)

    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: int | None = None,
        mbs: int | None = None,
        check_dim_skip_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        if eval_mode or self.diffusion_config.weight == 0:
            return super().train(
                data,
                loss_fn,
                eval_mode=eval_mode,
                gbs=gbs,
                mbs=mbs,
                check_dim_skip_keys=check_dim_skip_keys,
            )
        if check_dim_skip_keys is not None:
            raise ValueError(
                "Positive diffusion AR does not support check_dim_skip_keys"
            )
        global_batch_size = self.cfg["train_global_batch_size"] if gbs is None else gbs
        if (
            global_batch_size % self.dp_size
            or data.size != global_batch_size // self.dp_size
        ):
            raise ValueError(
                "Positive diffusion train expects one complete global batch; use the split API for streaming"
            )
        data.to("cuda")
        self.begin_train_step(loss_fn, gbs=global_batch_size, mbs=mbs)
        try:
            self.train_microbatch(data)
            return self.finish_train_step()
        except Exception:
            self.abort_train_step()
            raise


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class PositiveDiffusionARPolicy(PositiveDiffusionARPolicyImpl):
    pass
