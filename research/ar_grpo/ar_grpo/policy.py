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

"""Causal Nemotron policy for SingleController and legacy AR GRPO."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import ray

from nemo_rl.models.policy import PolicyConfig

from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)


class NemotronDiffusionMegatronPolicyWorkerImpl(MegatronPolicyWorkerImpl):
    def __init__(self, config: PolicyConfig, *args: Any, **kwargs: Any) -> None:
        # Bridge and checkpoint code are only needed inside the policy actor.
        from megatron.bridge.utils.instantiate_utils import (
            register_allowed_target_prefix,
        )
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(
            config["model_name"], trust_remote_code=True
        )
        register_allowed_target_prefix(type(hf_config).__module__ + ".")
        config["megatron_cfg"].setdefault("model_overrides", {})["seq_length"] = config[
            "max_total_sequence_length"
        ]
        super().__init__(config, *args, **kwargs)

    @contextmanager
    def use_nemotron_causal_attention_forward(self) -> Iterator[None]:
        """Run NemotronLabsDiffusionAttention through its causal inference path."""
        attention_modules = [
            module
            for module in self.model.modules()
            if hasattr(module, "set_inference_mode")
            and hasattr(module, "set_inference_params")
        ]
        if not attention_modules:
            raise RuntimeError("AR training requires Nemotron causal attention modules")

        saved_states = []
        for module in attention_modules:
            saved_states.append(
                (
                    module,
                    getattr(module, "_inference_mode", False),
                    getattr(module, "_inference_causal", True),
                    getattr(module, "_cache_enabled", False),
                )
            )
            if hasattr(module, "clear_kv_cache"):
                module.clear_kv_cache()
            module.set_inference_params(causal=True, cache_enabled=False)
            module.set_inference_mode(True)

        try:
            yield
        finally:
            for module, inference_mode, inference_causal, cache_enabled in saved_states:
                module.set_inference_params(
                    causal=inference_causal, cache_enabled=cache_enabled
                )
                module.set_inference_mode(inference_mode)
                if hasattr(module, "clear_kv_cache"):
                    module.clear_kv_cache()

    def train(self, *args: Any, **kwargs: Any) -> Any:
        with self.use_nemotron_causal_attention_forward():
            return super().train(*args, **kwargs)

    def train_microbatch(self, *args: Any, **kwargs: Any) -> Any:
        with self.use_nemotron_causal_attention_forward():
            return super().train_microbatch(*args, **kwargs)

    def get_logprobs(self, *args: Any, **kwargs: Any) -> Any:
        with self.use_nemotron_causal_attention_forward():
            return super().get_logprobs(*args, **kwargs)

    def get_topk_logits(self, *args: Any, **kwargs: Any) -> Any:
        with self.use_nemotron_causal_attention_forward():
            return super().get_topk_logits(*args, **kwargs)


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class NemotronDiffusionMegatronPolicyWorker(NemotronDiffusionMegatronPolicyWorkerImpl):
    pass
