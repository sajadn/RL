# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLaDA inference on the live Megatron training workers."""

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any

import ray
from block_diffusion.generation.megatron_generation import MegatronDiffusionGeneration

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class LLaDAMegatronGeneration(MegatronDiffusionGeneration):
    """Reuse the shared-worker lifecycle and switch decoding for validation."""

    def __init__(self, *, policy: Any, config: Any) -> None:
        super().__init__(policy=policy, config=config)
        self.forward_counts: list[int] = []

    def generate(
        self, data: BatchedDataDict[Any], greedy: bool = False
    ) -> BatchedDataDict[Any]:
        # The final validation batch can be smaller than DP. TP=PP=1 workers
        # can receive empty shards without invoking model collectives.
        shards = data.shard_by_batch_size(
            self.policy.data_parallel_size, allow_uneven_shards=True
        )
        group = self.policy.worker_group
        results = ray.get(
            group.run_all_workers_multiple_data(
                "generate", data=shards, common_kwargs={"greedy": greedy}
            )
        )
        for result in results:
            self.forward_counts.extend(result.pop("num_forward_passes").tolist())
        return BatchedDataDict.from_batches(
            results,
            pad_value_dict={
                "output_ids": self.policy.cfg["generation"]["_pad_token_id"]
            },
        )

    async def generate_async(
        self, data: BatchedDataDict[Any], greedy: bool = False
    ) -> AsyncIterator[tuple[int, BatchedDataDict[Any]]]:
        async for index, result in super().generate_async(data, greedy=greedy):
            self.forward_counts.extend(result.pop("num_forward_passes").tolist())
            yield index, result

    @contextmanager
    def validation_sampling(self, mode: str | None = None) -> Iterator[None]:
        self.forward_counts = []
        workers = self.policy.worker_group
        try:
            ray.get(
                workers.run_all_workers_single_data(
                    "set_validation_sampling", enabled=True, mode=mode
                )
            )
            yield
        finally:
            ray.get(
                workers.run_all_workers_single_data(
                    "set_validation_sampling", enabled=False
                )
            )
