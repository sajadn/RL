# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise single-pass training and attention restoration without Megatron GPUs."""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from test_policy import Attention
from test_policy import policy as policy
from test_positive_diffusion import config, trajectories

from nemo_rl.algorithms.loss.interfaces import LossType


@dataclass
class ProcessedMicrobatch:
    data_dict: object


@pytest.fixture
def auxiliary_policy(policy, monkeypatch):  # noqa: F811 - imported pytest fixture
    monkeypatch.setitem(sys.modules, "ar_grpo.policy", policy)
    core = ModuleType("megatron.core")
    core.parallel_state = SimpleNamespace(get_data_parallel_rank=lambda: 0)
    monkeypatch.setitem(sys.modules, core.__name__, core)
    data = ModuleType("nemo_rl.models.megatron.data")
    data.ProcessedMicrobatch = ProcessedMicrobatch
    data._get_non_packed_sequence_pad_factor = lambda cfg: cfg[
        "make_sequence_length_divisible_by"
    ]
    monkeypatch.setitem(sys.modules, data.__name__, data)
    train = ModuleType("nemo_rl.models.megatron.train")

    class NativeLossProcessor:
        def __init__(
            self,
            *,
            cfg,
            sampling_params=None,
            num_microbatches=1,
            cp_normalize=True,
            **kwargs,
        ):
            self.cfg = cfg
            self.sampling_params = sampling_params
            self.num_microbatches = num_microbatches
            self.cp_normalize = cp_normalize
            self.loss_fn = kwargs.get("loss_fn")

    train.LossPostProcessor = NativeLossProcessor
    train.LogprobsPostProcessor = object
    train.apply_temperature_scaling = lambda logits, _: logits
    monkeypatch.setitem(sys.modules, train.__name__, train)
    root = Path(__file__).resolve().parents[4]
    for name, path in (
        (
            "block_diffusion.diffusion_processors",
            root / "research/block_diffusion/block_diffusion/diffusion_processors.py",
        ),
        (
            "ar_grpo.positive_diffusion_processors",
            root / "research/ar_grpo/ar_grpo/positive_diffusion_processors.py",
        ),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        processor_module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, processor_module)
        spec.loader.exec_module(processor_module)
    # These worker tests inspect batch construction and metadata; the numerical
    # processor tests below exercise real input transformation and losses.
    monkeypatch.setattr(
        sys.modules["block_diffusion.diffusion_processors"],
        "prepare_diffusion_microbatch",
        lambda mb, **_: mb,
    )
    path = Path(__file__).resolve().parents[2] / "ar_grpo/positive_diffusion_policy.py"
    spec = importlib.util.spec_from_file_location("positive_policy_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(module, *, weight=0.1):
    result = object.__new__(module.PositiveDiffusionARPolicyImpl)
    attention = Attention()
    attention.metadata = None
    attention.clear_asymmetric_ar_metadata = lambda: setattr(
        attention, "metadata", None
    )
    attention.build_asymmetric_ar_metadata = lambda **kwargs: kwargs
    attention.set_asymmetric_ar_metadata = lambda metadata: setattr(
        attention, "metadata", metadata
    )
    result.model = torch.nn.Sequential(attention)
    result.diffusion_config = config(weight=weight)
    result.cfg = {
        "make_sequence_length_divisible_by": 1,
        "diffusion_aux_loss": result.diffusion_config.model_dump(),
        "generation": {"temperature": 0.6, "top_k": None, "top_p": 1.0},
    }
    result.tokenizer = SimpleNamespace(pad_token_id=0)
    result.sampling_params = SimpleNamespace(temperature=0.6)
    result.microbatch_processor = module.PositiveDiffusionMicrobatchProcessor(
        model=result.model, config=result.cfg, pad_token_id=0, data_parallel_rank=0
    )
    result.prepare_microbatch_fn = result.microbatch_processor
    result.loss_postprocessor_factory = module.PositiveDiffusionLossPostProcessor
    state = {
        "loss_fn": SimpleNamespace(loss_type=LossType.TOKEN_LEVEL),
        "local_valid_seqs": torch.tensor(0.0),
        "local_valid_toks": torch.tensor(0.0),
        "num_chunks": 0,
    }
    result._assert_step_open = lambda: state
    return result, state


@pytest.mark.parametrize("fail_backward", [False, True])
def test_one_pass_restores_attention_and_counts_ar_targets(
    auxiliary_policy, monkeypatch, fail_backward
):
    result, state = worker(auxiliary_policy)
    original_loss = state["loss_fn"]
    original_prepare = result.prepare_microbatch_fn
    original_factory = result.loss_postprocessor_factory
    original_sampling = result.sampling_params
    passes = []

    def native_microbatch(self, data):
        auxiliary = "diffusion_aux_enabled" in data
        # The native iterator requires all sequence fields to share the AR width.
        assert all(
            value.shape[1] == data["input_ids"].shape[1]
            for value in data.values()
            if torch.is_tensor(value) and value.ndim > 1
        )
        if auxiliary:
            data = self.prepare_microbatch_fn(ProcessedMicrobatch(data)).data_dict
            assert data["target_ids"].shape[1] == 8
            assert data["clean_input_ids"].shape[1] == 7
        passes.append(auxiliary)
        assert self.model[0]._inference_mode is (not auxiliary)
        if auxiliary:
            assert self.sampling_params is None
            assert state["loss_fn"] is original_loss
            assert self.prepare_microbatch_fn is original_prepare
            assert self.loss_postprocessor_factory is original_factory
        state["local_valid_seqs"] = (
            state["local_valid_seqs"] + data["sample_mask"].sum()
        )
        state["local_valid_toks"] = (
            state["local_valid_toks"]
            + (data["token_mask"][:, 1:] * data["sample_mask"][:, None]).sum()
        )
        state["num_chunks"] += 1
        if auxiliary and fail_backward:
            raise RuntimeError("auxiliary backward failed")
        self.model[0].weight.sum().backward()

    monkeypatch.setattr(
        auxiliary_policy.MegatronPolicyWorkerImpl, "train_microbatch", native_microbatch
    )
    if fail_backward:
        with pytest.raises(RuntimeError, match="auxiliary backward failed"):
            result.train_microbatch(trajectories())
    else:
        result.train_microbatch(trajectories())
        torch.testing.assert_close(
            result.model[0].weight.grad, torch.ones_like(result.model[0].weight)
        )
    assert passes == [True]
    assert state["loss_fn"] is original_loss
    assert state["local_valid_seqs"] == 3
    assert state["local_valid_toks"] == 10
    assert state["num_chunks"] == 1
    assert result.prepare_microbatch_fn is original_prepare
    assert result.loss_postprocessor_factory is original_factory
    assert result.sampling_params is original_sampling
    assert not result.model[0]._inference_mode
    assert result.model[0].metadata is None


def test_zero_weight_skips_auxiliary_forward(auxiliary_policy, monkeypatch):
    result, _ = worker(auxiliary_policy, weight=0.0)
    calls = []
    monkeypatch.setattr(
        auxiliary_policy.MegatronPolicyWorkerImpl,
        "train_microbatch",
        lambda self, data: calls.append(data),
    )
    result.train_microbatch(trajectories())
    assert len(calls) == 1


def test_sequence_normalization_is_rejected_before_opening_a_step(auxiliary_policy):
    result, _ = worker(auxiliary_policy)
    with pytest.raises(ValueError, match="token_level_loss"):
        result.begin_train_step(SimpleNamespace(loss_type=LossType.SEQUENCE_LEVEL))


@pytest.mark.parametrize("draft_enabled", [False, True])
def test_constructor_accepts_disabled_draft_and_rejects_enabled_draft(
    auxiliary_policy, monkeypatch, draft_enabled
):
    def native_init(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self.mtp_enabled = False
        self.sampling_params = SimpleNamespace(temperature=0.6)
        self.prepare_microbatch_fn = kwargs.get("prepare_microbatch_fn")
        self.loss_postprocessor_factory = kwargs["loss_postprocessor_factory"]
        self.model = torch.nn.Sequential(Attention())
        self.tokenizer = SimpleNamespace(pad_token_id=0)

    monkeypatch.setattr(
        auxiliary_policy.ARModeForMultiModeMegatronPolicyImpl, "__init__", native_init
    )
    cfg = {
        "diffusion_aux_loss": {"mask_token_id": 100, "block_size": 16},
        "megatron_cfg": {
            "enabled": True,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
        },
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
        "router_replay": None,
        "draft": {"enabled": draft_enabled},
        "make_sequence_length_divisible_by": 1,
    }
    if draft_enabled:
        with pytest.raises(ValueError, match="draft or MTP"):
            auxiliary_policy.PositiveDiffusionARPolicyImpl(cfg)
    else:
        result = auxiliary_policy.PositiveDiffusionARPolicyImpl(cfg)
        assert result.diffusion_config.weight == 0.1
        assert isinstance(
            result.prepare_microbatch_fn,
            auxiliary_policy.PositiveDiffusionMicrobatchProcessor,
        )
        assert (
            result.loss_postprocessor_factory
            is auxiliary_policy.PositiveDiffusionLossPostProcessor
        )
        assert cfg["megatron_cfg"]["model_overrides"]["block_size"] == 16


def test_ar_scoring_preparation_leaves_inputs_and_attention_unchanged(auxiliary_policy):
    result, _ = worker(auxiliary_policy)
    microbatch = SimpleNamespace(data_dict=trajectories())
    assert result.prepare_microbatch_fn(microbatch) is microbatch
    assert not result.model[0]._inference_mode


def test_registered_class_recovers_ar_sampling_during_paired_forward(auxiliary_policy):
    result, _ = worker(auxiliary_policy)
    with result._use_positive_diffusion_forward():
        assert result.sampling_params is None
        processor = result.loss_postprocessor_factory(
            cfg=result.cfg, sampling_params=result.sampling_params
        )
        assert processor.sampling_params.temperature == 0.6
        assert processor.sampling_params.top_k is None
        assert processor.sampling_params.top_p == 1.0
        assert (
            processor.diffusion_processor.loss_fn.weight
            == result.diffusion_config.weight
        )


def test_asymmetric_metadata_uses_separate_noisy_and_clean_widths(auxiliary_policy):
    result, _ = worker(auxiliary_policy)
    batch = trajectories()
    batch["diffusion_aux_enabled"] = torch.ones(batch.size, dtype=torch.bool)
    seen = []
    result.model[0].set_asymmetric_ar_metadata = seen.append
    microbatch = ProcessedMicrobatch(data_dict=batch)
    prepared = result.prepare_microbatch_fn(microbatch)
    assert prepared is not microbatch
    assert prepared.data_dict["input_ids"] is batch["input_ids"]
    assert seen[0]["noisy_length"] == 8
    assert seen[0]["clean_length"] == 7
    torch.testing.assert_close(seen[0]["clean_lengths"], batch["input_lengths"])
