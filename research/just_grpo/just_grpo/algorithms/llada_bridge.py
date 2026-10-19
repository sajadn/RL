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
"""Public layer-spec target for native LLaDA Bridge checkpoint configs."""

from megatron.bridge.diffusion.conversion.llada15.llada15_bridge import LLaDA15Bridge
from megatron.bridge.diffusion.models.llada15.llada15_provider import (
    LLaDA15ModelProvider,
    _llada15_layer_spec,
)
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.utils.instantiate_utils import register_allowed_target_prefix
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import ModuleSpec


def llada_layer_spec(config: LLaDA15ModelProvider) -> ModuleSpec:
    """Reuse Bridge's bidirectional attention through a public callable."""
    return _llada15_layer_spec(config)


# Register only this trusted adapter module for serialized checkpoint callables.
register_allowed_target_prefix(__name__)


@MegatronModelBridge.register_bridge(
    source="LLaDAModelLM",
    target=GPTModel,
    provider=LLaDA15ModelProvider,
    model_type="llada",
)
class LLaDAJustGRPOBridge(LLaDA15Bridge):
    """Reuse all LLaDA mappings and serialize a loadable layer-spec target."""

    def provider_bridge(
        self, hf_pretrained: PreTrainedCausalLM
    ) -> LLaDA15ModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        provider.transformer_layer_spec = llada_layer_spec
        return provider
