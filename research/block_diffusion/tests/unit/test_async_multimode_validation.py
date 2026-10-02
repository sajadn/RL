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
"""Async validation must drain before sleeping and refit the intended group."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nemo_rl.algorithms.async_utils.trajectory_collector import AsyncTrajectoryCollector
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.weight_sync.collective_weight_synchronizer import (
    CollectiveWeightSynchronizer,
)
from block_diffusion.generation.validation import (
    MultiModeValidation,
    validation_variants,
)
from test_multimode_validation import variants


def collector():
    cls = AsyncTrajectoryCollector.__ray_metadata__.modified_class
    obj = object.__new__(cls)
    obj.running = True
    obj._threads_lock = threading.Lock()
    obj._outstanding_lock = threading.Lock()
    obj._inflight_threads = set()
    obj._live_threads = set()
    obj._outstanding_task_indices = set()
    obj._manual_pause_cleared = threading.Event()
    obj._manual_pause_cleared.set()
    obj._refit_pause_cleared = threading.Event()
    obj._refit_pause_cleared.set()
    return obj


@pytest.mark.parametrize("gate", ["_manual_pause_cleared", "_refit_pause_cleared"])
def test_paused_dispatch_stays_blocked_until_resume(gate):
    obj = collector()
    getattr(obj, gate).clear()
    started = threading.Event()
    worker = threading.Thread(target=started.set)
    dispatch = threading.Thread(
        target=obj._start_worker_when_unpaused, args=(worker, [5])
    )
    dispatch.start()
    try:
        obj.pause_and_drain()
        assert not started.is_set()
        assert not obj._inflight_threads
        assert not obj._outstanding_task_indices
    finally:
        obj._refit_pause_cleared.set()
        obj.resume()
        dispatch.join(timeout=2)
    assert not dispatch.is_alive()
    worker.join(timeout=2)
    assert started.is_set()
    assert obj._outstanding_task_indices == {5}


def test_pause_and_drain_waits_for_active_work():
    obj = collector()
    started, release, drained = threading.Event(), threading.Event(), threading.Event()

    def work():
        started.set()
        release.wait(timeout=3)

    worker = threading.Thread(target=work)
    assert obj._start_worker_when_unpaused(worker, [])
    assert started.wait(timeout=1)

    def drain():
        obj.pause_and_drain()
        drained.set()

    waiter = threading.Thread(target=drain)
    waiter.start()
    try:
        assert not drained.wait(timeout=0.05)
    finally:
        release.set()
        worker.join(timeout=2)
        waiter.join(timeout=2)
    assert drained.is_set()
    assert not obj._manual_pause_cleared.is_set()


def test_registration_and_thread_start_hold_pause_lock():
    obj = collector()
    worker = Mock()

    def start():
        assert obj._threads_lock.locked()
        assert worker in obj._inflight_threads
        assert obj._outstanding_task_indices == {9}

    worker.start.side_effect = start
    assert obj._start_worker_when_unpaused(worker, [9])


def test_failed_or_stopped_dispatch_leaves_no_pending_work():
    obj = collector()
    worker = Mock()
    worker.start.side_effect = RuntimeError("thread creation failed")
    with pytest.raises(RuntimeError, match="thread creation failed"):
        obj._start_worker_when_unpaused(worker, [1])
    assert not obj._inflight_threads
    assert not obj._live_threads
    assert not obj._outstanding_task_indices
    obj.running = False
    assert not obj._start_worker_when_unpaused(Mock(), [2])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_explicit_sleep_wake_releases_noncolocated_engine(monkeypatch, asynchronous):
    import nemo_rl.models.generation.vllm.vllm_generation as module

    group = object.__new__(VllmGeneration)
    group.cfg = {
        "colocated": {"enabled": False},
        "vllm_cfg": {"async_engine": asynchronous},
    }
    group.worker_group = Mock()
    monkeypatch.setattr(module.ray, "get", Mock(return_value=[True]))
    assert group.sleep()
    assert group.wake_up()
    methods = [
        call.args[0]
        for call in group.worker_group.run_all_workers_single_data.call_args_list
    ]
    assert methods == (
        ["sleep_async", "wake_up_async"] if asynchronous else ["sleep", "wake_up"]
    )


def test_named_collectives_preserve_rollout_and_each_other(monkeypatch):
    import nemo_rl.distributed.stateless_process_group as process_groups
    import nemo_rl.distributed.refit_watchdog as watchdog
    import nemo_rl.models.policy.workers.base_policy_worker as module

    created = []

    def create(**kwargs):
        group = Mock()
        created.append(group)
        return group

    monkeypatch.setattr(process_groups, "StatelessProcessGroup", create)
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(module.torch.cuda, "empty_cache", lambda: None)
    released = Mock()
    monkeypatch.setattr(watchdog, "release_within", released)
    worker = AbstractPolicyWorker()
    worker.rank = 0
    for port, name in enumerate([None, "vllm_val_ar", "vllm_val_diffusion"], 20000):
        worker.init_collective(
            "localhost", port, 2, train_world_size=1, generation_group=name
        )
    assert worker.get_model_update_group() is created[0]
    assert worker.get_model_update_group("vllm_val_ar") is created[1]
    assert worker.get_model_update_group("vllm_val_diffusion") is created[2]
    released.assert_not_called()
    worker.init_collective(
        "localhost", 20004, 2, train_world_size=1, generation_group="vllm_val_ar"
    )
    assert worker.get_model_update_group() is created[0]
    assert worker.get_model_update_group("vllm_val_diffusion") is created[2]
    assert released.call_args.args[0] == created[1].abort
    with pytest.raises(RuntimeError, match="not been initialized"):
        worker.get_model_update_group("missing")


@pytest.mark.parametrize("name", [None, "vllm_val_ar", "vllm_val_diffusion"])
def test_collective_sender_routes_init_and_weight_updates(monkeypatch, name):
    import nemo_rl.weight_sync.collective_weight_synchronizer as module

    policy = Mock()
    generation = Mock(cfg={"refit_namespace": name}, worker_group=None)
    generation.get_collective_sender_spec.return_value = SimpleNamespace(
        nccl_peer="nemo", buffer_size_bytes=1024, num_buffers=2
    )
    policy.init_collective.return_value = []
    generation.init_collective.return_value = []
    generation.get_inference_world_size.return_value = 1
    train = Mock()
    train.world_size.return_value = 1
    train.get_master_address_and_port.return_value = ("localhost", 20001)
    monkeypatch.setattr(module.ray, "get", Mock(return_value=[True]))
    sync = CollectiveWeightSynchronizer(policy, generation, train, Mock())
    sync.init_communicator()
    sync.sync_weights()
    for call in [
        policy.init_collective.call_args,
        policy.broadcast_weights_for_collective.call_args,
    ]:
        if name is None:
            assert "generation_group" not in call.kwargs
        else:
            assert call.kwargs["generation_group"] == name


def test_runner_uses_noncolocated_synchronizers(monkeypatch):
    import nemo_rl.weight_sync.factory as factory

    runner = MultiModeValidation(validation_variants(variants()), colocated=False)
    runner.generations = {name: Mock() for name in runner.variants}
    create = Mock()
    monkeypatch.setattr(factory, "create_weight_synchronizer", create)
    policy, train, inference = Mock(), Mock(), Mock()
    runner.initialize(policy, train_cluster=train, inference_cluster=inference)
    assert create.call_count == 2
    for call in create.call_args_list:
        assert call.kwargs["policy"] is policy
        assert call.kwargs["train_cluster"] is train
        assert call.kwargs["inference_cluster"] is inference
        assert call.kwargs["colocated"] is False
