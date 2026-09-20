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
"""GDPO orchestration using shared GRPO rollout, reward, and update utilities."""

from typing import Any

from transformers import PreTrainedTokenizerBase

from gdpo.actor_environments import POLICY_WORKER, register_actor_environments
from gdpo.algorithms.loss.gdpo import GDPOLossFn
from gdpo.generation.automodel import AutomodelGeneration
from nemo_rl.algorithms.grpo import MasterConfig, grpo_train
from nemo_rl.algorithms.grpo import setup as setup_grpo
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy


def _make_policy(**kwargs: Any) -> Policy:
    """Construct a policy using the GDPO worker extension."""
    return Policy(**kwargs, worker_extension_cls_fqn=POLICY_WORKER)


def _make_generation(
    policy_config: PolicyConfig, policy: ColocatablePolicyInterface
) -> AutomodelGeneration:
    """Wrap the live Automodel policy as a generation interface."""
    assert isinstance(policy, Policy), (
        "GDPO generation requires the paired Policy factory"
    )
    return AutomodelGeneration(policy_config, policy)


def setup(
    master_config: MasterConfig,
    tokenizer: PreTrainedTokenizerBase,
    dataset: AllTaskProcessedDataset | dict[str, AllTaskProcessedDataset],
    val_dataset: AllTaskProcessedDataset | None,
) -> tuple[Any, ...]:
    """Build colocated GDPO policy/generation and its ELBO-aware loss."""
    register_actor_environments()
    return setup_grpo(
        master_config,
        tokenizer,
        dataset,
        val_dataset,
        policy_factory=_make_policy,
        generation_factory=_make_generation,
        loss_factory=GDPOLossFn,
    )


def _rollout_metrics(**kwargs: Any) -> dict[str, float]:
    """Denoising provides no generation likelihood for discrepancy filtering."""
    return {}


def gdpo_train(*args: Any, **kwargs: Any) -> None:
    """Run synchronous GDPO, retaining shared reward/checkpoint/validation handling."""
    grpo_train(*args, **kwargs, rollout_metrics_fn=_rollout_metrics)
