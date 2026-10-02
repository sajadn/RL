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
"""Run Block JustGRPO through the shared diffusion driver."""

import argparse
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from block_diffusion import training
from just_grpo.config import validate_config


from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def run(config: DictConfig) -> None:
    """Validate JustGRPO settings and start the shared training loop."""
    training.run(config, diffusion=validate_config(config))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent
        / "configs/recipes/just_grpo-sudoku6x6-4n8g-megatron-inference-long.yaml",
    )
    args, overrides = parser.parse_known_args()
    register_omegaconf_resolvers()
    config = parse_hydra_overrides(load_config(args.config), overrides)
    OmegaConf.resolve(config)
    run(config)


if __name__ == "__main__":
    main()
