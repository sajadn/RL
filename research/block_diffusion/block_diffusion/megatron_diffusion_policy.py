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
"""Shared Megatron execution for caller-supplied diffusion schedules."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable

import torch
from megatron.bridge.utils.instantiate_utils import register_allowed_target_prefix
from megatron.core import parallel_state
from transformers import AutoConfig

from block_diffusion.config import DiffusionSamplingParams
from block_diffusion.denoising_schedule import (
    DenoisingSchedule,
    SchedulePurpose,
    aggregate_diffusion_logprobs,
)
from block_diffusion.diffusion_processors import (
    DiffusionLogprobsPostProcessor,
    DiffusionLossPostProcessor,
    DiffusionMicrobatchProcessor,
)
from block_diffusion.generation.megatron_generation import (
    generate_responses,
    pack_responses,
)
from block_diffusion.positive_diffusion import (
    PositiveDiffusionLoss,
    build_positive_diffusion_level,
    resolve_positive_diffusion_config,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.data.interfaces import TokenizerType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.train import model_forward
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)


class MegatronDiffusionPolicyWorkerImpl(MegatronPolicyWorkerImpl):
    """Execute diffusion schedules using the native worker lifecycle."""

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: TokenizerType,
        *,
        mask_token_id: int,
        block_size: int,
        sampling: DiffusionSamplingParams,
        generation_seed: int,
        max_generation_kl: float,
        loss_postprocessor_factory: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Register the trusted configuration class before Bridge reads the checkpoint.
        hf_config = AutoConfig.from_pretrained(
            config["model_name"], trust_remote_code=True
        )
        if hf_config.model_type != "nemotron_labs_diffusion":
            raise ValueError("Diffusion worker requires Nemotron Labs Diffusion")
        if (
            2 * config["max_total_sequence_length"]
            > hf_config.rope_parameters["original_max_position_embeddings"]
        ):
            raise ValueError(
                "Doubled diffusion context exceeds the unscaled position window"
            )
        register_allowed_target_prefix(type(hf_config).__module__ + ".")
        config["megatron_cfg"]["model_overrides"] = {
            "seq_length": config["max_total_sequence_length"],
            "block_size": block_size,
        }
        self.mask_token_id = mask_token_id
        super().__init__(
            config,
            tokenizer,
            loss_postprocessor_factory=(
                loss_postprocessor_factory or DiffusionLossPostProcessor
            ),
            logprobs_postprocessor_factory=DiffusionLogprobsPostProcessor,
            **kwargs,
        )
        # The native constructor creates the model before a processor can bind it.
        self.prepare_microbatch_fn: DiffusionMicrobatchProcessor = (
            DiffusionMicrobatchProcessor(model=self.model, mask_token_id=mask_token_id)
        )
        self.generation_vocab_size = hf_config.vocab_size
        self.mask_token_id = mask_token_id
        self.sampling = sampling
        self.generation_seed = generation_seed
        self.max_generation_kl = max_generation_kl
        self.generation_index = 0
        settings = self.cfg.get("diffusion_aux_loss")
        self.positive_diffusion = (
            None
            if settings is None
            else resolve_positive_diffusion_config(
                settings, mask_token_id=mask_token_id, block_size=block_size
            )
        )
        if self.positive_diffusion is not None and self.positive_diffusion.weight == 0:
            self.positive_diffusion = None
        if self.positive_diffusion is not None:
            # Identical draws on tensor-parallel peers; distinct across DP ranks.
            self.positive_diffusion_generator = torch.Generator(
                device="cpu"
            ).manual_seed(
                self.positive_diffusion.seed + parallel_state.get_data_parallel_rank()
            )

    def _streaming_loss_token_mask(self, data: BatchedDataDict[Any]) -> torch.Tensor:
        """Diffusion predicts every selected position, including column zero.

        The auxiliary level supplies its own denominator; its targets must not
        dilute the policy loss.
        """
        if "diffusion_mask_probability" in data:
            return torch.zeros_like(data["token_mask"])
        return data["token_mask"]

    @contextmanager
    def _use_positive_diffusion_loss(self) -> Iterator[None]:
        """Score the auxiliary level with untempered denoising CE."""
        saved = self.sampling_params, self.loss_postprocessor_factory
        loss_fn = PositiveDiffusionLoss(self.positive_diffusion)
        self.sampling_params = None
        self.loss_postprocessor_factory = lambda **kwargs: DiffusionLossPostProcessor(
            **{**kwargs, "loss_fn": loss_fn}
        )
        try:
            yield
        finally:
            self.sampling_params, self.loss_postprocessor_factory = saved

    def _train_positive_diffusion_level(self, data: BatchedDataDict[Any]) -> None:
        """Accumulate one uniformly masked level on positive completions.

        Runs after the schedule so the policy denominator is known. The
        auxiliary sum is rescaled from that denominator to the count of all
        valid response tokens, matching the AR GRPO auxiliary loss.
        """
        state = self._assert_step_open()
        state["metric_normalizations"] = {
            **state["metric_normalizations"],
            **PositiveDiffusionLoss.metric_normalizations,
        }
        level, response_tokens = build_positive_diffusion_level(
            data,
            self.positive_diffusion,
            pad_token_id=self.tokenizer.pad_token_id,
            padded_width=self.cfg["max_total_sequence_length"],
            stop_token_ids=self.cfg["generation"]["stop_token_ids"] or [],
            generator=self.positive_diffusion_generator,
        )
        counts = torch.stack(
            [state["local_valid_toks"], response_tokens.to(state["local_valid_toks"])]
        )
        torch.distributed.all_reduce(
            counts, group=parallel_state.get_data_parallel_group()
        )
        level["diffusion_aux_scale"] = torch.full_like(
            level["diffusion_mask_probability"],
            float(counts[0] / counts[1].clamp_min(1)),
        )
        with self._use_positive_diffusion_loss():
            self.train_microbatch(level)

    def _build_schedule(
        self, data: BatchedDataDict[Any], *, purpose: SchedulePurpose
    ) -> DenoisingSchedule:
        """Build an algorithm schedule for policy scoring, reference scoring, or training."""
        raise NotImplementedError(
            "Diffusion algorithms must provide a denoising schedule"
        )

    def get_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: int | None = None,
        require_router_replay: bool = True,
    ) -> BatchedDataDict[Any]:
        """Score the policy schedule; router replay does not choose coverage."""
        data.to("cuda")
        return self._get_schedule_logprobs(
            data=data,
            schedule=self._build_schedule(data, purpose="policy"),
            micro_batch_size=micro_batch_size,
            require_router_replay=require_router_replay,
        )

    def get_reference_policy_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: int | None = None,
    ) -> BatchedDataDict[Any]:
        """Score the reference schedule with reference weights across all levels."""
        data.to("cuda")
        with self.use_reference_model():
            scores = self._get_schedule_logprobs(
                data=data,
                schedule=self._build_schedule(data, purpose="reference"),
                micro_batch_size=micro_batch_size,
                require_router_replay=False,
            )
        return BatchedDataDict(
            reference_logprobs=scores["logprobs"],
            logprob_token_mask=scores["logprob_token_mask"],
        )

    def _get_schedule_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        schedule: DenoisingSchedule,
        micro_batch_size: int | None = None,
        require_router_replay: bool = True,
    ) -> BatchedDataDict[Any]:
        """Score exactly the supplied schedule, independent of router replay."""

        def score(level: BatchedDataDict[Any]) -> torch.Tensor:
            return super(MegatronDiffusionPolicyWorkerImpl, self).get_logprobs(
                data=level,
                micro_batch_size=micro_batch_size,
                require_router_replay=require_router_replay,
            )["logprobs"]

        try:
            scores = aggregate_diffusion_logprobs(
                schedule.iter_levels(data),
                score,
                original_shape=data["input_ids"].shape,
                device=data["input_ids"].device,
            )
        finally:
            self.prepare_microbatch_fn.clear_asymmetric_metadata()
        return scores.to("cpu")

    def train(
        self,
        *,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: int | None = None,
        mbs: int | None = None,
        check_dim_skip_keys: Any = None,
    ) -> dict[str, Any]:
        """Accumulate all schedule levels before one optimizer update."""
        if eval_mode:
            raise ValueError("Use get_logprobs for diffusion scoring")
        data.to("cuda")
        schedule = self._build_schedule(data, purpose="train")
        local_tokens = torch.zeros((), device=data["input_ids"].device)
        self.begin_train_step(loss_fn, gbs=gbs, mbs=mbs)
        try:
            for level in schedule.iter_levels(data):
                local_tokens += (
                    level["token_mask"] * level["sample_mask"][:, None]
                ).sum()
                self.train_microbatch(level)
            if self.positive_diffusion is not None:
                self._train_positive_diffusion_level(data)
            result = self.finish_train_step()
        except Exception:
            self.abort_train_step()
            raise
        finally:
            self.prepare_microbatch_fn.clear_asymmetric_metadata()
        # Per-view sample counts must not multiply the actual rollout count.
        result["all_mb_metrics"]["num_valid_samples"] = [
            float((data["sample_mask"] > 0).sum())
        ]
        views = schedule.num_steps + (self.positive_diffusion is not None)
        result["all_mb_metrics"]["global_valid_seqs"] = [
            count / views for count in result["all_mb_metrics"]["global_valid_seqs"]
        ]
        kl = (
            sum(result["all_mb_metrics"]["gen_kl_error"])
            * result["all_mb_metrics"]["global_valid_toks"][0]
            / local_tokens.clamp_min(1)
        )
        torch.distributed.all_reduce(
            kl,
            op=torch.distributed.ReduceOp.MAX,
            group=parallel_state.get_data_parallel_group(),
        )
        if not torch.isfinite(kl) or kl > self.max_generation_kl:
            raise RuntimeError(f"Generation/training KL exceeds limit: {float(kl)}")
        return result

    @torch.no_grad()
    def generate(
        self, *, data: BatchedDataDict[Any], greedy: bool = False
    ) -> BatchedDataDict[Any]:
        if (
            self.cfg["generation"]["backend"] != "megatron"
            or parallel_state.get_tensor_model_parallel_world_size() != 1
        ):
            raise ValueError(
                "In-process diffusion generation requires Megatron runtime with TP=1"
            )
        prompts = [
            row[: int(length)].tolist()
            for row, length in zip(data["input_ids"], data["input_lengths"])
        ]
        sampling = self.sampling.model_copy()
        if greedy:
            sampling.temperature = 0.0
        generation = self.cfg["generation"]
        stop = generation["stop_token_ids"]

        def forward(
            input_ids: torch.Tensor, position_ids: torch.Tensor
        ) -> torch.Tensor:
            logits = model_forward(
                self.model,
                BatchedDataDict(),
                input_ids,
                position_ids,
                attention_mask=None,
                defer_fp32_logits=True,
            )
            # MCore can pad the vocabulary for tensor parallelism. These
            # additional IDs are not part of the tokenizer's distribution.
            return logits[..., : self.generation_vocab_size]

        try:
            responses = generate_responses(
                forward,
                prompts,
                batch_size=self.cfg["logprob_batch_size"],
                device=torch.device("cuda", torch.cuda.current_device()),
                sampling=sampling,
                prepare_attention=self.prepare_microbatch_fn.set_asymmetric_metadata,
                max_new_tokens=generation["max_new_tokens"],
                max_sequence_length=self.cfg["max_total_sequence_length"],
                mask_token_id=self.mask_token_id,
                stop_token_ids=stop,
                seed=self.generation_seed
                + parallel_state.get_data_parallel_rank()
                + self.generation_index * parallel_state.get_data_parallel_world_size(),
            )
        finally:
            self.prepare_microbatch_fn.clear_asymmetric_metadata()
        self.generation_index += 1
        return pack_responses(
            prompts,
            responses,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=generation["max_new_tokens"],
            stop_token_ids=stop,
        )
