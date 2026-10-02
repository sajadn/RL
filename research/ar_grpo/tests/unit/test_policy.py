# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise both causal policy entry points without initializing Megatron."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import ray
import torch


class Attention(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 1)
        self._inference_mode = False
        self._inference_causal = False
        self._cache_enabled = True
        self.cache_clears = 0

    def set_inference_params(self, *, causal: bool, cache_enabled: bool) -> None:
        self._inference_causal = causal
        self._cache_enabled = cache_enabled

    def set_inference_mode(self, enabled: bool) -> None:
        self._inference_mode = enabled

    def clear_kv_cache(self) -> None:
        self.cache_clears += 1


class NativeWorker:
    def __init__(self, config: Any) -> None:
        self.cfg = config

    def train(self, data: torch.Tensor, *, fail: bool = False) -> torch.Tensor:
        # Exercise restoration when a batch call also invokes a microbatch call.
        return self.train_microbatch(data, fail=fail)

    def train_microbatch(
        self, data: torch.Tensor, *, fail: bool = False
    ) -> torch.Tensor:
        return self.get_logprobs(data, fail=fail)

    def get_logprobs(self, data: torch.Tensor, *, fail: bool = False) -> torch.Tensor:
        layer = self.model[0]
        assert layer._inference_mode
        assert layer._inference_causal
        assert not layer._cache_enabled
        assert torch.is_grad_enabled()
        if fail:
            raise RuntimeError("forward failed")
        return self.model(data)

    def get_topk_logits(
        self, data: torch.Tensor, *, fail: bool = False
    ) -> torch.Tensor:
        return self.get_logprobs(data, fail=fail)


@pytest.fixture
def policy(monkeypatch: pytest.MonkeyPatch) -> Any:
    native = ModuleType("nemo_rl.models.policy.workers.megatron_policy_worker")
    native.MegatronPolicyWorkerImpl = NativeWorker
    monkeypatch.setitem(sys.modules, native.__name__, native)
    runtime = ModuleType("nemo_rl.models.policy.utils")
    runtime.get_runtime_env_for_policy_worker = lambda _: {}
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setattr(ray, "remote", lambda **_: lambda cls: cls)
    path = Path(__file__).resolve().parents[2] / "ar_grpo/policy.py"
    spec = importlib.util.spec_from_file_location("causal_policy_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "method", ["train", "train_microbatch", "get_logprobs", "get_topk_logits"]
)
@pytest.mark.parametrize("fail", [False, True])
def test_causal_calls_preserve_gradients_and_restore_attention(
    policy: Any, method: str, fail: bool
) -> None:
    worker = object.__new__(policy.NemotronDiffusionMegatronPolicyWorkerImpl)
    layer = Attention()
    worker.model = torch.nn.Sequential(layer)
    data = torch.ones(1, 2)
    if fail:
        with pytest.raises(RuntimeError, match="forward failed"):
            getattr(worker, method)(data, fail=True)
    else:
        output = getattr(worker, method)(data)
        output.sum().backward()
        torch.testing.assert_close(layer.weight.grad, data)
    assert not layer._inference_mode
    assert not layer._inference_causal
    assert layer._cache_enabled
    assert layer.cache_clears >= 2


@pytest.mark.parametrize("with_overrides", [False, True])
def test_constructor_preserves_model_overrides(
    policy: Any, monkeypatch: pytest.MonkeyPatch, with_overrides: bool
) -> None:
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    registered = []
    bridge = ModuleType("megatron.bridge.utils.instantiate_utils")
    bridge.register_allowed_target_prefix = registered.append
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    config = {
        "model_name": "test-model",
        "max_total_sequence_length": 4096,
        "megatron_cfg": {"model_overrides": {"hidden_size": 2}}
        if with_overrides
        else {},
    }
    worker = policy.NemotronDiffusionMegatronPolicyWorkerImpl(config)
    assert worker.cfg["megatron_cfg"]["model_overrides"]["seq_length"] == 4096
    if with_overrides:
        assert worker.cfg["megatron_cfg"]["model_overrides"]["hidden_size"] == 2
    assert registered == ["types."]


def test_missing_causal_attention_fails_explicitly(policy: Any) -> None:
    worker = object.__new__(policy.NemotronDiffusionMegatronPolicyWorkerImpl)
    worker.model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    with pytest.raises(RuntimeError, match="causal attention modules"):
        worker.train_microbatch(torch.ones(1, 2))
