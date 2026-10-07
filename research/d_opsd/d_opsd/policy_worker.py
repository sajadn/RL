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
"""Frozen self-teacher and full-vocabulary loss on the shared diffusion worker."""

from typing import Any

import ray
import torch
from omegaconf import OmegaConf

from block_diffusion.megatron_diffusion_policy import MegatronDiffusionPolicyWorkerImpl
from d_opsd.algorithm import DOPSDConfig, DOPSDSchedule, prepare_teacher_targets
from d_opsd.loss import DOPSDLoss
from megatron.core import parallel_state
from nemo_rl.data.interfaces import TokenizerType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.data import ProcessedMicrobatch
from nemo_rl.models.megatron.train import (
    LossPostProcessor,
    apply_temperature_scaling,
    model_forward,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker


def prepare_distillation_logits(
    logits: torch.Tensor, data: BatchedDataDict[Any], **kwargs: Any
):
    """Gather same-position predictions; teacher targets are already detached."""
    vocab_size = data["dopsd_teacher_logits"].shape[-1]
    return {"logits": logits[:, : data["target_ids"].shape[1], :vocab_size]}, data


class DOPSDLossPostProcessor(LossPostProcessor):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, prepare_fn=prepare_distillation_logits, **kwargs)


class DOPSDPolicyWorkerImpl(MegatronDiffusionPolicyWorkerImpl):
    """Accumulate all replay transitions, then update only the student weights."""

    def __init__(self, config: PolicyConfig, tokenizer: TokenizerType, **kwargs: Any):
        research_config = OmegaConf.create(config.pop("diffusion_config"))
        self.dopsd = DOPSDConfig.model_validate(
            OmegaConf.to_container(research_config.d_opsd, resolve=True)
        )
        self.distillation_loss = DOPSDLoss(pointwise_clip=self.dopsd.pointwise_clip)
        # Teacher initialization is required even though GRPO's reference KL is disabled.
        kwargs["init_reference_model"] = True
        super().__init__(
            config,
            tokenizer,
            mask_token_id=self.dopsd.schedule.mask_token_id,
            block_size=self.dopsd.schedule.block_size,
            sampling=self.dopsd.sampling,
            generation_seed=research_config.grpo.seed,
            max_generation_kl=self.dopsd.max_generation_kl,
            loss_postprocessor_factory=DOPSDLossPostProcessor,
            **kwargs,
        )

    @torch.no_grad()
    def _teacher_targets(self, level: BatchedDataDict[Any]) -> BatchedDataDict[Any]:
        teacher = BatchedDataDict(level)
        teacher["masked_indices"] = level["dopsd_teacher_mask"]
        prepared = self.prepare_microbatch_fn(
            ProcessedMicrobatch(
                data_dict=teacher,
                input_ids=teacher["input_ids"],
                input_ids_cp_sharded=teacher["input_ids"],
                attention_mask=None,
                position_ids=teacher["position_ids"],
                packed_seq_params=None,
                cu_seqlens_padded=None,
            )
        )
        logits = model_forward(
            self.model,
            teacher,
            prepared.input_ids_cp_sharded,
            prepared.position_ids,
            attention_mask=None,
            defer_fp32_logits=True,
        )[:, : teacher["input_ids"].shape[1], : self.generation_vocab_size]
        # Direct teacher inference bypasses forward_with_post_processing_fn;
        # use its same helper exactly once to match the student's distribution.
        logits = apply_temperature_scaling(logits, self.sampling_params)
        return prepare_teacher_targets(
            level,
            logits,
            block_size=self.dopsd.schedule.block_size,
            selection_source=self.dopsd.selection_source,
        )

    def train(
        self,
        *,
        data: BatchedDataDict[Any],
        loss_fn: Any,
        eval_mode: bool = False,
        gbs: int | None = None,
        mbs: int | None = None,
        check_dim_skip_keys: Any = None,
    ) -> dict[str, Any]:
        if eval_mode:
            raise ValueError("d-OPSD train does not support eval_mode")
        data.to("cuda")
        local_valid = (data["sample_mask"] * data["dopsd_loss_mask"].any(1)).sum()
        global_valid = local_valid.clone()
        torch.distributed.all_reduce(
            global_valid, group=parallel_state.get_data_parallel_group()
        )
        if global_valid.item() == 0:
            # Adam momentum and weight decay must not update a fully rejected batch.
            return {
                "global_loss": torch.tensor(0.0),
                "grad_norm": torch.tensor([0.0]),
                "rank": torch.distributed.get_rank(),
                "gpu_name": torch.cuda.get_device_name(),
                "model_dtype": self.dtype,
                "all_mb_metrics": {
                    "loss": [0.0],
                    "dopsd_skipped_update": [1.0],
                    "num_valid_samples": [0.0],
                },
            }
        schedule = DOPSDSchedule(
            data,
            self.dopsd,
            pad_token_id=self.tokenizer.pad_token_id,
            padded_width=self.cfg["max_total_sequence_length"],
        )
        batch_size = mbs if mbs is not None else self.cfg["train_micro_batch_size"]
        self.begin_train_step(self.distillation_loss, gbs=gbs, mbs=batch_size)
        try:
            for level in schedule.iter_levels():
                for start in range(0, level.size, batch_size):
                    microbatch = BatchedDataDict(
                        {
                            key: value[start : start + batch_size]
                            for key, value in level.items()
                        }
                    )
                    # Reference weights come from the initial pretrained checkpoint,
                    # including when the student resumes an optimizer checkpoint.
                    with self.use_reference_model():
                        self.model.eval()
                        targets = self._teacher_targets(microbatch)
                    self.model.train()
                    self.train_microbatch(targets)
                    del targets
            result = self.finish_train_step()
        except Exception:
            self.abort_train_step()
            raise
        finally:
            self.prepare_microbatch_fn.clear_asymmetric_metadata()
        result["all_mb_metrics"]["num_valid_samples"] = [float(local_valid)]
        result["all_mb_metrics"]["dopsd_skipped_update"] = [0.0]
        return result


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class DOPSDPolicyWorker(DOPSDPolicyWorkerImpl):
    pass
