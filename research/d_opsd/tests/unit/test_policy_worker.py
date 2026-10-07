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
"""Exercise the d-OPSD worker lifecycle with CPU execution adapters."""

import copy
import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from test_d_opsd import batch, config
from d_opsd.loss import DOPSDLoss
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.fixture
def worker_module(monkeypatch):
    # Match the research AR adapter tests: replace only GPU/backend boundaries.
    ray = ModuleType("ray")
    ray.remote = lambda **_: lambda cls: cls
    monkeypatch.setitem(sys.modules, "ray", ray)
    core = ModuleType("megatron.core")
    core.parallel_state = SimpleNamespace(get_data_parallel_group=lambda: None)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    data = ModuleType("nemo_rl.models.megatron.data")
    data.ProcessedMicrobatch = SimpleNamespace
    monkeypatch.setitem(sys.modules, data.__name__, data)
    train = ModuleType("nemo_rl.models.megatron.train")
    train.LossPostProcessor = object
    train.apply_temperature_scaling = lambda logits, params: logits / params.temperature
    train.model_forward = lambda model, data, ids, positions, **_: model(ids)
    monkeypatch.setitem(sys.modules, train.__name__, train)
    processors = ModuleType("block_diffusion.diffusion_processors")

    def prepare(mb, *, mask_token_id):
        ids = mb.data_dict["input_ids"].masked_fill(
            mb.data_dict["masked_indices"], mask_token_id
        )
        mb.input_ids_cp_sharded = torch.cat([ids, mb.data_dict["clean_input_ids"]], 1)
        return mb

    processors.prepare_diffusion_microbatch = prepare
    monkeypatch.setitem(sys.modules, processors.__name__, processors)
    shared = ModuleType("block_diffusion.megatron_diffusion_policy")
    shared.MegatronDiffusionPolicyWorkerImpl = object
    monkeypatch.setitem(sys.modules, shared.__name__, shared)
    utils = ModuleType("nemo_rl.models.policy.utils")
    utils.get_runtime_env_for_policy_worker = lambda _: {}
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_, **__: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU test")
    original_to = BatchedDataDict.to
    monkeypatch.setattr(
        BatchedDataDict, "to", lambda self, device, **kw: original_to(self, "cpu", **kw)
    )
    path = Path(__file__).resolve().parents[2] / "d_opsd/policy_worker.py"
    spec = importlib.util.spec_from_file_location("dopsd_worker_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32)

    def forward(self, ids):
        hidden = self.embedding(ids)
        return self.head(hidden + hidden.mean(1, keepdim=True))


def worker(module):
    torch.manual_seed(11)
    result = object.__new__(module.DOPSDPolicyWorkerImpl)
    result.dopsd = config()
    result.distillation_loss = DOPSDLoss(pointwise_clip=0.05)
    result.cfg = {"max_total_sequence_length": 12, "train_micro_batch_size": 1}
    result.tokenizer = SimpleNamespace(pad_token_id=0)
    result.mask_token_id = 31
    result.generation_vocab_size = 32
    result.dtype = torch.float32
    result.model = TinyModel()
    teacher = copy.deepcopy(result.model.state_dict())
    optimizer = torch.optim.AdamW(result.model.parameters(), lr=0.01)
    calls = []
    normalization = []

    @contextmanager
    def reference():
        student = copy.deepcopy(result.model.state_dict())
        result.model.load_state_dict(teacher)
        try:
            with torch.no_grad():
                yield
        finally:
            result.model.load_state_dict(student)

    def begin(loss_fn, **kwargs):
        calls.append("begin")
        optimizer.zero_grad()
        normalization.clear()

    def train(data):
        calls.append("backward")
        assert result.model.training
        assert not data["dopsd_teacher_logits"].requires_grad
        ids = data["input_ids"].masked_fill(data["masked_indices"], 31)
        raw = result.model(torch.cat([ids, data["clean_input_ids"]], 1))
        raw = raw / result.sampling_params.temperature
        prepared, _ = module.prepare_distillation_logits(raw, data)
        loss, _ = result.distillation_loss(
            data, torch.tensor(1.0), torch.tensor(1.0), **prepared
        )
        loss.backward()
        normalization.append(data["token_mask"].sum())

    def finish():
        calls.append("finish")
        for parameter in result.model.parameters():
            parameter.grad.div_(sum(normalization).clamp_min(1))
        optimizer.step()
        return {
            "global_loss": torch.tensor(0.0),
            "grad_norm": torch.tensor([0.0]),
            "all_mb_metrics": {},
        }

    result.use_reference_model = reference
    result.begin_train_step = begin
    result.train_microbatch = train
    result.finish_train_step = finish
    result.abort_train_step = lambda: calls.append("abort")

    class MicrobatchProcessor:
        def __call__(self, microbatch):
            calls.append("attention")
            return sys.modules[
                "block_diffusion.diffusion_processors"
            ].prepare_diffusion_microbatch(
                microbatch, mask_token_id=result.mask_token_id
            )

        def clear_asymmetric_metadata(self):
            calls.append("clear")

    result.prepare_microbatch_fn = MicrobatchProcessor()
    result.sampling_params = SimpleNamespace(temperature=0.7)
    return result, teacher, calls


def test_student_updates_once_and_frozen_teacher_survives_two_iterations(worker_module):
    result, teacher, calls = worker(worker_module)
    original = copy.deepcopy(result.model.state_dict())
    for _ in range(2):
        result.train(data=batch(), loss_fn=object())
    assert calls.count("begin") == calls.count("finish") == calls.count("clear") == 2
    assert "abort" not in calls
    assert any(
        not torch.equal(original[key], result.model.state_dict()[key])
        for key in original
    )
    for key in teacher:
        torch.testing.assert_close(teacher[key], original[key])


def test_all_rejected_batch_does_not_open_optimizer_step(worker_module):
    result, teacher, calls = worker(worker_module)
    metrics = result.train(data=batch(rewards=(0.0, 0.0)), loss_fn=object())
    assert not calls
    assert metrics["all_mb_metrics"]["dopsd_skipped_update"] == [1.0]
    for key in teacher:
        torch.testing.assert_close(result.model.state_dict()[key], teacher[key])


def test_teacher_failure_restores_student_and_aborts_training(worker_module):
    result, teacher, calls = worker(worker_module)
    with torch.no_grad():
        result.model.head.weight.add_(0.1)
    original = copy.deepcopy(result.model.state_dict())

    def fail(_):
        raise RuntimeError("teacher forward failed")

    result._teacher_targets = fail
    with pytest.raises(RuntimeError, match="teacher forward failed"):
        result.train(data=batch(), loss_fn=object())
    assert calls == ["begin", "abort", "clear"]
    for key in original:
        torch.testing.assert_close(result.model.state_dict()[key], original[key])


@pytest.mark.parametrize("microbatch_size", [2, 4])
def test_batched_updates_match_single_sample_gradients(worker_module, microbatch_size):
    baseline, _, baseline_calls = worker(worker_module)
    batched, _, batched_calls = worker(worker_module)
    baseline.train(data=batch(rewards=(1.0, 1.0)), loss_fn=object(), mbs=1)
    batched.train(
        data=batch(rewards=(1.0, 1.0)),
        loss_fn=object(),
        mbs=microbatch_size,
    )
    for expected, actual in zip(
        baseline.model.parameters(), batched.model.parameters()
    ):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert batched_calls.count("backward") == baseline_calls.count("backward") // 2


def test_rejected_rows_are_not_replayed(worker_module):
    result, _, calls = worker(worker_module)
    data = batch(rewards=(1.0, 1.0)).select_indices([0, 1] * 8)
    data["sample_mask"][2:] = 0
    data["input_ids"][2:] = 30
    sizes = []

    def capture_inputs(_, args):
        ids = args[0]
        sizes.append(ids.shape[0])
        assert not (ids == 30).any()

    handle = result.model.register_forward_pre_hook(capture_inputs)
    try:
        metrics = result.train(data=data, loss_fn=object(), mbs=2)
    finally:
        handle.remove()
    assert sizes == [2] * (2 * result.dopsd.sampling.max_steps)
    assert calls.count("backward") == result.dopsd.sampling.max_steps
    assert metrics["all_mb_metrics"]["num_valid_samples"] == [2.0]


def test_imbalanced_dp_ranks_execute_equal_batches(worker_module, monkeypatch):
    # Three ranks have 0, 1, and 3 accepted trajectories. All must execute the
    # same microbatch schedule; padding contributes no loss or normalization.
    def reduce_counts(value, *, group, op=None):
        value.fill_(3 if op == torch.distributed.ReduceOp.MAX else 4)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_counts)
    chunk_counts = []
    for accepted in (0, 1, 3):
        result, _, calls = worker(worker_module)
        data = batch(rewards=(1.0, 1.0)).select_indices([0, 1] * 4)
        data["sample_mask"][accepted:] = 0
        masks = []
        original_targets = result._teacher_targets

        def capture_targets(level):
            masks.append(level["sample_mask"].clone())
            return original_targets(level)

        result._teacher_targets = capture_targets
        metrics = result.train(data=data, loss_fn=object(), mbs=2)
        chunk_counts.append(calls.count("backward"))
        assert all(mask.shape == (2,) for mask in masks)
        assert torch.stack(masks).sum() == accepted * result.dopsd.sampling.max_steps
        assert metrics["all_mb_metrics"]["num_valid_samples"] == [float(accepted)]
        assert metrics["all_mb_metrics"]["dopsd_skipped_update"] == [0.0]
        assert all(torch.isfinite(p.grad).all() for p in result.model.parameters())
    assert chunk_counts == [2 * result.dopsd.sampling.max_steps] * 3


def test_invalid_microbatch_size_is_rejected(worker_module):
    result, _, calls = worker(worker_module)
    with pytest.raises(ValueError, match="microbatch size must be positive"):
        result.train(data=batch(), loss_fn=object(), mbs=0)
    assert not calls


@pytest.mark.parametrize("accepted_rows", [2, 6])
def test_teacher_weights_swap_once_per_level_and_cache_stays_on_cpu(
    worker_module: ModuleType, monkeypatch: pytest.MonkeyPatch, accepted_rows: int
) -> None:
    result, _, _ = worker(worker_module)
    data = batch(rewards=(1.0, 1.0)).select_indices([0, 1] * (accepted_rows // 2))
    events = []
    transfers = []
    cached_sources = []
    original_reference = result.use_reference_model
    original_targets = result._teacher_targets
    original_train = result.train_microbatch
    original_to = BatchedDataDict.to

    @contextmanager
    def reference() -> Iterator[None]:
        events.append("reference")
        with original_reference():
            yield
        events.append("restore")

    def targets(level: BatchedDataDict[Any]) -> BatchedDataDict[Any]:
        events.append("teacher")
        return original_targets(level)

    def train(data: BatchedDataDict[Any]) -> None:
        events.append("student")
        original_train(data)

    def move(
        data: BatchedDataDict[Any], device: str | torch.device
    ) -> BatchedDataDict[Any]:
        if "dopsd_teacher_logits" in data:
            transfers.append(str(device))
            if str(device) == "cpu":
                cached_sources.append(data)
            else:
                assert all(data is not cached for cached in cached_sources)
        return original_to(data, device)

    result.use_reference_model = reference
    result._teacher_targets = targets
    result.train_microbatch = train
    monkeypatch.setattr(BatchedDataDict, "to", move)
    result.train(data=data, loss_fn=object(), mbs=2)

    count = accepted_rows // 2
    assert (
        events
        == (["reference"] + ["teacher"] * count + ["restore"] + ["student"] * count)
        * result.dopsd.sampling.max_steps
    )
    if count == 1:
        assert "cpu" not in transfers
    else:
        assert transfers.count("cpu") == count * result.dopsd.sampling.max_steps
        assert transfers.count("cuda") == count * result.dopsd.sampling.max_steps


def test_grouped_teacher_phase_matches_interleaved_gradients(
    worker_module: ModuleType,
) -> None:
    baseline, _, _ = worker(worker_module)
    grouped, _, _ = worker(worker_module)
    data = batch(rewards=(1.0, 1.0)).select_indices([0, 1] * 3)
    schedule = worker_module.DOPSDSchedule(
        data,
        baseline.dopsd,
        pad_token_id=0,
        padded_width=baseline.cfg["max_total_sequence_length"],
    )
    baseline.begin_train_step(baseline.distillation_loss, mbs=2)
    # Oracle: retain the original teacher/student interleaving for each microbatch.
    for microbatch in schedule.return_schedule(2):
        with baseline.use_reference_model():
            baseline.model.eval()
            targets = baseline._teacher_targets(microbatch)
        baseline.model.train()
        baseline.train_microbatch(targets)
    baseline.finish_train_step()
    grouped.train(data=data, loss_fn=object(), mbs=2)
    for expected, actual in zip(
        baseline.model.parameters(), grouped.model.parameters()
    ):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_student_failure_occurs_after_teacher_weights_are_restored(
    worker_module: ModuleType,
) -> None:
    result, _, calls = worker(worker_module)
    with torch.no_grad():
        result.model.head.weight.add_(0.1)
    student = copy.deepcopy(result.model.state_dict())
    data = batch(rewards=(1.0, 1.0)).select_indices([0, 1] * 3)

    def fail(targets: BatchedDataDict[Any]) -> None:
        for key, expected in student.items():
            torch.testing.assert_close(result.model.state_dict()[key], expected)
        raise RuntimeError("student backward failed")

    result.train_microbatch = fail
    with pytest.raises(RuntimeError, match="student backward failed"):
        result.train(data=data, loss_fn=object(), mbs=2)
    assert calls.count("begin") == calls.count("abort") == calls.count("clear") == 1
    assert "finish" not in calls
