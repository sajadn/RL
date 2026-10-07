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
"""d-OPSD validation on top of the shared diffusion runtime constraints."""

from omegaconf import DictConfig, OmegaConf

from block_diffusion.validation import validate_experiment
from d_opsd.algorithm import DOPSDConfig


def validate_config(config: DictConfig) -> DOPSDConfig:
    diffusion = DOPSDConfig.model_validate(
        OmegaConf.to_container(config.d_opsd, resolve=True)
    )
    validate_experiment(config, diffusion, require_policy_scores=False)
    if (
        config.policy.worker_extension_cls_fqn
        != "d_opsd.policy_worker.DOPSDPolicyWorker"
    ):
        raise ValueError("d-OPSD requires DOPSDPolicyWorker")
    if config.grpo.async_grpo.enabled:
        raise ValueError(
            "d-OPSD requires synchronous, freshly sampled trajectories with one update"
        )
    if config.policy.megatron_cfg.tensor_model_parallel_size != 1:
        raise ValueError("d-OPSD full-vocabulary targets currently require TP=1")
    if config.grpo.seq_logprob_error_threshold is not None:
        raise ValueError("d-OPSD does not use whole-response logprob filtering")
    if (config.get("data_plane") or {}).get("enabled"):
        raise ValueError(
            "d-OPSD requires driver-side reward and reveal-state preparation"
        )
    if (
        not config.loss_fn.force_on_policy_ratio
        or config.loss_fn.use_importance_sampling_correction
        or config.loss_fn.truncated_importance_sampling_type is not None
    ):
        raise ValueError(
            "d-OPSD does not use policy ratios or importance sampling correction"
        )
    if (
        config.loss_fn.reference_policy_kl_penalty != 0
        or not config.grpo.skip_reference_policy_logprobs_calculation
    ):
        raise ValueError(
            "d-OPSD uses privileged teacher logits, not GRPO reference-policy KL"
        )
    if config.grpo.reward_scaling.enabled or config.grpo.normalize_rewards:
        raise ValueError(
            "d-OPSD correctness thresholds require unscaled verification rewards"
        )
    return diffusion
