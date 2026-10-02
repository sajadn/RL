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
"""Trace-specific experiment configuration and validation."""

from omegaconf import DictConfig, OmegaConf

from block_diffusion.validation import validate_experiment
from trace_grpo.algorithm import TraceGRPOConfig, validate_trace_experiment


def validate_config(config: DictConfig) -> TraceGRPOConfig:
    """Parse Trace settings and validate algorithm and shared runtime constraints."""
    diffusion = TraceGRPOConfig.model_validate(
        OmegaConf.to_container(config.trace_grpo, resolve=True)
    )
    validate_trace_experiment(config)
    validate_experiment(config, diffusion)
    return diffusion
