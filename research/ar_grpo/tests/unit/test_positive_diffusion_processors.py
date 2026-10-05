# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare one asymmetric forward/backward with the former two-pass objective."""

import copy
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch
from ar_grpo.positive_diffusion import PositiveDiffusionLoss
from test_positive_diffusion import build, config, trajectories
from torch.nn import functional as F

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

pytestmark = pytest.mark.mcore


@pytest.fixture
def processors(monkeypatch):
    # Only model construction requires Bridge/Transformer Engine. Keep the real
    # upstream loss processors, input preparation, and vocabulary logprob code.
    for name, symbol in (
        ("nemo_rl.models.megatron.data", "ProcessedMicrobatch"),
        ("nemo_rl.models.megatron.config", "MegatronModule"),
    ):
        module = ModuleType(name)
        setattr(module, symbol, object)
        if name == "nemo_rl.models.megatron.data":
            module._get_non_packed_sequence_pad_factor = lambda cfg: 1
        monkeypatch.setitem(sys.modules, name, module)
    train = importlib.import_module("nemo_rl.models.megatron.train")
    for name, value in (
        ("get_tensor_model_parallel_rank", 0),
        ("get_tensor_model_parallel_group", None),
        ("get_context_parallel_group", None),
        ("get_context_parallel_world_size", 1),
    ):
        monkeypatch.setattr(train, name, lambda value=value: value)
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: None)

    # Load fresh modules so the policy entry-point tests' stubs cannot leak here.
    root = Path(__file__).resolve().parents[4]
    modules = []
    for name, path in (
        (
            "block_diffusion.diffusion_processors",
            root / "research/block_diffusion/block_diffusion/diffusion_processors.py",
        ),
        (
            "ar_grpo.positive_diffusion_processors",
            root / "research/ar_grpo/ar_grpo/positive_diffusion_processors.py",
        ),
        (
            "asymmetric_mask_under_test",
            root
            / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/diffusion/common/dllm.py",
        ),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules.append(module)
    return train, *modules


@dataclass
class TinyProcessedMicrobatch:
    data_dict: BatchedDataDict
    packed_seq_params: object = None
    input_ids: torch.Tensor | None = None
    input_ids_cp_sharded: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    original_seq_length: int | None = None


class TinyAttentionPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = torch.nn.Embedding(128, 12)
        self.positions = torch.nn.Embedding(8, 12)
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(12, 36, bias=False) for _ in range(2)
        )
        self.output = torch.nn.Linear(12, 128, bias=False)
        self.forward_calls = 0
        self.metadata = None

    def build_asymmetric_ar_metadata(self, **kwargs):
        return kwargs

    def set_asymmetric_ar_metadata(self, metadata):
        self.metadata = metadata

    def clear_asymmetric_ar_metadata(self):
        self.metadata = None

    def forward(self, ids, positions, allowed):
        self.forward_calls += 1
        hidden = self.tokens(ids) + self.positions(positions)
        for layer in self.layers:
            q, k, v = layer(hidden).chunk(3, dim=-1)
            hidden = hidden + F.scaled_dot_product_attention(
                q[:, None], k[:, None], v[:, None], attn_mask=allowed[:, None]
            ).squeeze(1)
        return self.output(hidden)


@pytest.mark.parametrize(
    "temperature,logit_dtype",
    [(1.0, torch.float32), (0.6, torch.float32), (0.6, torch.bfloat16)],
)
@pytest.mark.parametrize("num_microbatches", [1, 4])
@pytest.mark.parametrize("empty_positive", [False, True])
def test_one_forward_matches_two_pass_losses_metrics_and_parameter_gradients(
    processors, temperature, logit_dtype, num_microbatches, empty_positive
):
    train, diffusion, combined, masks = processors
    torch.manual_seed(123)
    model = TinyAttentionPolicy()
    reference = copy.deepcopy(model)
    data = trajectories()
    for key, offset in (
        ("prev_logprobs", 0.0),
        ("generation_logprobs", 0.2),
        ("reference_policy_logprobs", -0.1),
    ):
        data[key] = torch.full_like(data["token_mask"], -5.0 + offset)
    if empty_positive:
        # Retain nonzero negative AR gradients while eliminating diffusion targets.
        data["advantages"].fill_(-1)
    cfg = config(mask_probability_min=0.5, mask_probability_max=0.5)
    batch = build(data, cfg)
    width = batch["target_ids"].shape[1]
    clean_width = batch["clean_input_ids"].shape[1]
    assert clean_width == data["input_ids"].shape[1] == 7
    assert width == 8
    predicate = masks.asymmetric_semi_ar_mask_mod(
        block_size=cfg.block_size,
        noisy_length=width,
        noisy_response_offset=0,
        prompt_lengths=batch["prompt_lengths"],
        noisy_valid_lengths=batch["noisy_valid_lengths"],
        clean_lengths=batch["clean_lengths"],
    )
    indices = torch.arange(width + clean_width)
    allowed = predicate(
        torch.arange(data.size)[:, None, None],
        torch.tensor(0),
        indices[None, :, None],
        indices[None, None, :],
    )
    assert not allowed[:, width:, :width].any()
    # A noisy query cannot see its clean target or the current clean block.
    for row, position in batch["masked_indices"].nonzero():
        block_start = int(position) // cfg.block_size * cfg.block_size
        assert not allowed[row, position, width + block_start :].any()

    input_processor = combined.PositiveDiffusionMicrobatchProcessor(
        model=model,
        config={"diffusion_aux_loss": cfg.model_dump()},
        pad_token_id=0,
        data_parallel_rank=0,
    )
    training_data = BatchedDataDict(data)
    training_data["diffusion_aux_enabled"] = torch.ones(data.size, dtype=torch.bool)
    prepared = input_processor(TinyProcessedMicrobatch(data_dict=training_data))
    for key in ("target_ids", "masked_indices", "diffusion_mask_probability"):
        torch.testing.assert_close(prepared.data_dict[key], batch[key])
    assert model.metadata["noisy_length"] == width
    assert model.metadata["clean_length"] == clean_width
    inputs, paired_positions = prepared.input_ids, prepared.position_ids
    assert inputs.shape == (data.size, width + clean_width)
    torch.testing.assert_close(inputs[:, width:], data["input_ids"])
    assert prepared.data_dict["input_ids"] is data["input_ids"]
    positions = torch.arange(clean_width).expand(data.size, -1)
    logits = model(inputs, paired_positions, allowed)
    causal_logits = reference(
        batch["clean_input_ids"], positions, allowed[:, width:, width:]
    )
    torch.testing.assert_close(logits[:, width:], causal_logits, rtol=1e-5, atol=1e-6)
    noisy_logits = reference(inputs, paired_positions, allowed).to(logit_dtype)
    causal_logits = causal_logits.to(logit_dtype)
    logits = logits.to(logit_dtype)
    sampling = TrainingSamplingParams(temperature=temperature)
    ar_loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            token_level_loss=True,
            reference_policy_kl_penalty=0.01,
            use_importance_sampling_correction=True,
        )
    )
    policy_cfg = {
        "sequence_packing": {"enabled": False},
        "diffusion_aux_loss": cfg.model_dump(),
        "generation": {"temperature": temperature, "top_k": None, "top_p": 1.0},
    }
    seqs = batch["sample_mask"].sum()
    toks = (batch["token_mask"][:, 1:] * batch["sample_mask"][:, None]).sum()
    kwargs = dict(
        cfg=policy_cfg, num_microbatches=num_microbatches, sampling_params=sampling
    )
    one_pass = combined.PositiveDiffusionLossPostProcessor(
        loss_fn=ar_loss_fn,
        **{**kwargs, "sampling_params": None},
    )
    raw_before = logits.detach().clone()
    actual, actual_metrics = one_pass(
        batch, global_valid_seqs=seqs, global_valid_toks=toks
    )(logits)
    torch.testing.assert_close(logits.detach(), raw_before)
    auxiliary = BatchedDataDict(batch)
    auxiliary["token_mask"] = batch["masked_indices"]
    expected_ar, ar_metrics = train.LossPostProcessor(loss_fn=ar_loss_fn, **kwargs)(
        data, global_valid_seqs=seqs, global_valid_toks=toks
    )(train.apply_temperature_scaling(causal_logits.clone(), sampling))
    expected_aux, aux_metrics = diffusion.DiffusionLossPostProcessor(
        loss_fn=PositiveDiffusionLoss(cfg), **kwargs
    )(auxiliary, global_valid_seqs=seqs, global_valid_toks=toks)(noisy_logits)
    expected = expected_ar + expected_aux
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected_metrics = {
        **ar_metrics,
        **aux_metrics,
        "loss": ar_metrics["loss"] + aux_metrics["loss"],
    }
    assert actual_metrics == pytest.approx(expected_metrics, rel=1e-5, abs=1e-6)
    actual.backward()
    expected.backward()
    for parameter, expected_parameter in zip(
        model.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            parameter.grad, expected_parameter.grad, rtol=1e-4, atol=1e-6
        )
        assert torch.isfinite(parameter.grad).all()
    assert model.forward_calls == 1
    assert reference.forward_calls == 2
    if empty_positive:
        assert actual_metrics["diffusion_aux_loss"] == 0
    else:
        assert actual_metrics["diffusion_aux_loss"] > 0


@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_ordinary_ar_batch_matches_upstream_loss_and_gradients(processors, temperature):
    train, _, combined, _ = processors
    data = trajectories()
    for key in ("prev_logprobs", "generation_logprobs", "reference_policy_logprobs"):
        data[key] = torch.full_like(data["token_mask"], -5.0)
    sampling = TrainingSamplingParams(temperature=temperature)
    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(token_level_loss=True))
    kwargs = dict(
        loss_fn=loss_fn,
        cfg={
            "sequence_packing": {"enabled": False},
            "diffusion_aux_loss": config().model_dump(),
            "generation": {"temperature": temperature, "top_k": None, "top_p": 1.0},
        },
        sampling_params=sampling,
        cp_normalize=False,
    )
    processor = combined.PositiveDiffusionLossPostProcessor(**kwargs)
    logits = torch.randn(*data["input_ids"].shape, 128, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    seqs = data["sample_mask"].sum()
    toks = (data["token_mask"][:, 1:] * data["sample_mask"][:, None]).sum()
    counts = dict(global_valid_seqs=seqs, global_valid_toks=toks)
    actual, metrics = processor(data, **counts)(
        train.apply_temperature_scaling(logits.clone(), sampling)
    )
    expected, expected_metrics = train.LossPostProcessor(**kwargs)(data, **counts)(
        train.apply_temperature_scaling(reference_logits.clone(), sampling)
    )
    torch.testing.assert_close(actual, expected)
    assert metrics == pytest.approx(expected_metrics)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_shared_preparation_preserves_existing_diffusion_canvas(processors):
    _, diffusion, _, _ = processors
    targets = torch.tensor([[1, 2, 3, 0]])
    context = torch.tensor([[4, 5, 1, 2, 3]])
    data = BatchedDataDict(
        input_ids=targets,
        target_ids=targets,
        clean_input_ids=context,
        masked_indices=torch.tensor([[True, False, True, False]]),
        prompt_lengths=torch.tensor([2]),
        position_ids=torch.arange(4)[None],
    )
    prepared = diffusion.prepare_diffusion_microbatch(
        TinyProcessedMicrobatch(data_dict=data), mask_token_id=100
    )
    torch.testing.assert_close(
        prepared.input_ids, torch.tensor([[100, 2, 100, 0, 4, 5, 1, 2, 3]])
    )
    torch.testing.assert_close(
        prepared.position_ids, torch.tensor([[2, 3, 4, 5, 0, 1, 2, 3, 4]])
    )
    assert prepared.data_dict is data


@pytest.mark.parametrize("with_targets", [False, True])
def test_processor_shares_and_clears_metadata_for_training_and_generation(
    processors, with_targets
):
    _, diffusion, _, _ = processors
    model = torch.nn.Sequential(TinyAttentionPolicy(), TinyAttentionPolicy())
    data = BatchedDataDict(
        input_ids=torch.zeros(2, 4, dtype=torch.long),
        clean_input_ids=torch.zeros(2, 7, dtype=torch.long),
        prompt_lengths=torch.tensor([3, 2]),
        response_lengths=torch.tensor([4, 4]),
        noisy_valid_lengths=torch.tensor([4, 4]),
        clean_lengths=torch.tensor([7, 6]),
    )
    if with_targets:
        data["target_ids"] = torch.zeros(2, 8, dtype=torch.long)
    processor = diffusion.DiffusionMicrobatchProcessor(model=model, mask_token_id=100)
    processor.set_asymmetric_metadata(data)
    assert model[0].metadata is model[1].metadata
    assert model[0].metadata["noisy_length"] == (8 if with_targets else 4)
    assert model[0].metadata["clean_length"] == 7
    processor.clear_asymmetric_metadata()
    assert model[0].metadata is model[1].metadata is None
