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
"""AR GRPO with native math datasets and shared AR/diffusion validation."""

import argparse
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from block_diffusion import training
from block_diffusion.generation.validation import validation_variants
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def run(config: DictConfig, *, generation_python: str) -> None:
    """Use the causal policy adapter and upstream GRPO controllers."""
    rollout = OmegaConf.to_container(config.policy.generation, resolve=True)
    if rollout["vllm_kwargs"]["diffusion_config"] is not None or rollout["vllm_kwargs"][
        "hf_overrides"
    ]["architectures"] != ["NemotronLabsDiffusionForCausalLM"]:
        raise ValueError("AR GRPO requires causal rollout engines")
    if config.policy.worker_extension_cls_fqn != (
        "ar_grpo.ar_policy_worker.NemotronDiffusionMegatronPolicyWorker"
    ):
        raise ValueError("AR GRPO requires the Nemotron causal policy worker")
    if config.grpo.async_grpo.enabled and rollout["colocated"]["enabled"]:
        raise ValueError("Async reference vLLM requires dedicated generation GPUs")
    if not validation_variants(rollout):
        raise ValueError("Configure named AR/diffusion validation variants")
    training.run(config, generation_python=generation_python)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent
        / "configs/recipes/ar_grpo-deepscaler-3b-4n8g-megatron-vllm-async-dualval-long.yaml",
    )
    parser.add_argument("--model")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--generation-python", default=os.environ.get("NRL_VLLM_PY_EXECUTABLE")
    )
    args, overrides = parser.parse_known_args()
    if not args.generation_python:
        parser.error(
            "Set NRL_VLLM_PY_EXECUTABLE or --generation-python to the diffusion vLLM runtime"
        )
    register_omegaconf_resolvers()
    config = parse_hydra_overrides(load_config(args.config), overrides)
    if args.model is not None:
        config.policy.model_name = args.model
    if args.output_dir is not None:
        config.logger.log_dir = args.output_dir
    OmegaConf.resolve(config)
    run(config, generation_python=args.generation_python)


if __name__ == "__main__":
    main()
