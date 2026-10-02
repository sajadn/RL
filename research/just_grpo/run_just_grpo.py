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

from pathlib import Path

from just_grpo import train

from block_diffusion.cli import main as diffusion_main


def main() -> None:
    diffusion_main(
        Path(__file__).parent
        / "configs/recipes/just_grpo-sudoku6x6-4n8g-megatron-inference-long.yaml",
        config_key="just_grpo",
        run=train.run,
    )


if __name__ == "__main__":
    main()
