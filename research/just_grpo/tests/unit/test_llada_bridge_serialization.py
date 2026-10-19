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
"""Checkpoint loading checks run inside the provisioned Megatron container."""

import os
from pathlib import Path

import pytest
from just_grpo.algorithms.llada_bridge import llada_layer_spec
from megatron.bridge.training.model_load_save import load_model_config
from megatron.bridge.utils.instantiate_utils import instantiate


def test_public_layer_spec_round_trips_without_disabling_private_target_guard():
    restored = instantiate(
        {
            "_target_": "just_grpo.algorithms.llada_bridge.llada_layer_spec",
            "_call_": False,
        }
    )
    assert restored is llada_layer_spec


def test_converted_llada_checkpoint_model_config_is_loadable():
    root = Path(os.environ["NRL_MEGATRON_CHECKPOINT_DIR"])
    configs = list(root.rglob("run_config.yaml"))
    if not configs:
        pytest.skip("No converted checkpoint cache yet")
    for config in configs:
        provider, _ = load_model_config(str(config.parent))
        assert provider.transformer_layer_spec is llada_layer_spec
        spec = provider.transformer_layer_spec(provider)
        assert spec.submodules.self_attention.submodules.core_attention.__name__ == (
            "LLaDA15TEDotProductAttention"
        )
