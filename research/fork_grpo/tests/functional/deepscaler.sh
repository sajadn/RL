#!/usr/bin/env bash
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
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MODEL:?Set MODEL to the Nemotron Labs Diffusion checkpoint}"
: "${OUTPUT_DIR:?Set a fresh OUTPUT_DIR}"
export PYTHONPATH="${project_dir}/../..:${project_dir}:${project_dir}/../block_diffusion:${project_dir}/../trace_grpo${PYTHONPATH:+:${PYTHONPATH}}"
uv run --directory "${project_dir}/../.." --all-packages --extra mcore python "${project_dir}/run_fork_grpo.py" \
  --config "${CONFIG:-${project_dir}/configs/recipes/fork_grpo-deepscaler-3b-4n8g-megatron-inference-long.yaml}" \
  "policy.model_name=${MODEL}" "logger.log_dir=${OUTPUT_DIR}" \
  logger.tensorboard_enabled=true \
  cluster.num_nodes=1 cluster.gpus_per_node=2 \
  grpo.num_prompts_per_step=2 grpo.num_generations_per_prompt=4 \
  grpo.max_num_steps=3 grpo.max_val_samples=2 grpo.val_batch_size=2 \
  grpo.val_period=-1 grpo.val_at_start=true grpo.val_at_end=true \
  policy.train_micro_batch_size=1 policy.logprob_batch_size=1 \
  policy.megatron_cfg.scheduler.lr_warmup_iters=0
uv run --directory "${project_dir}/../.." --all-packages --extra mcore python \
  "${project_dir}/../just_grpo/tests/functional/check_metrics.py" "${OUTPUT_DIR}/tensorboard"
