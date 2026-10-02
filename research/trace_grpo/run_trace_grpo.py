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
"""Run TraceGRPO through the shared diffusion GRPO driver."""

from pathlib import Path

from omegaconf import DictConfig

from block_diffusion import training
from block_diffusion.cli import main as diffusion_main
from block_diffusion.training import PrepareTrainingData
from trace_grpo.algorithm import TraceGRPO
from trace_grpo.config import validate_config


def run(config: DictConfig) -> None:
    """Validate Trace settings and supply its rollout batch preparation callback."""
    diffusion = validate_config(config)

    def prepare_training_data_factory(stop_token_ids: list[int]) -> PrepareTrainingData:
        return TraceGRPO(diffusion, stop_token_ids=stop_token_ids).prepare_training_data

    training.run(
        config,
        diffusion=diffusion,
        prepare_training_data_factory=prepare_training_data_factory,
    )


def main() -> None:
    diffusion_main(
        Path(__file__).parent
        / "configs/recipes/trace_grpo-sudoku6x6-4n8g-megatron-inference-long.yaml",
        config_key="trace_grpo",
        run=run,
    )


if __name__ == "__main__":
    main()
