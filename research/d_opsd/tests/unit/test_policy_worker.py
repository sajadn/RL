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
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

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

    result.microbatch_processor = MicrobatchProcessor()
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
