# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Positive selection, denoising targets, gradients, and streaming normalization."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, ValidationError

from ar_grpo.positive_diffusion import (
    PositiveDiffusionConfig,
    PositiveDiffusionLoss,
    build_positive_diffusion_batch,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def trajectories() -> BatchedDataDict:
    return BatchedDataDict(
        input_ids=torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 0, 0],
                [13, 14, 15, 16, 0, 0, 0],
                [1, 2, 3, 4, 5, 6, 0],
            ]
        ),
        input_lengths=torch.tensor([7, 5, 4, 6]),
        # A tool feedback span between two assistant spans in row zero.
        token_mask=torch.tensor(
            [
                [0, 0, 1, 1, 0, 1, 1],
                [0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
            ],
            dtype=torch.float,
        ),
        sample_mask=torch.tensor([1.0, 1.0, 1.0, 0.0]),
        advantages=torch.tensor([[1.0] * 7, [-1.0] * 7, [0.0] * 7, [1.0] * 7]),
        rewards=torch.tensor([1.0, 1.0, 0.0, 1.0]),
    )


def config(**kwargs) -> PositiveDiffusionConfig:
    return PositiveDiffusionConfig(mask_token_id=100, block_size=4, **kwargs)


def build(data, cfg, seed=42):
    return build_positive_diffusion_batch(
        data,
        cfg,
        pad_token_id=0,
        pad_multiple=4,
        generator=torch.Generator().manual_seed(seed),
    )


@pytest.mark.parametrize(
    "positive_samples, selected_rows", [("advantage", [0]), ("reward", [0, 1])]
)
def test_only_positive_response_targets_contribute_to_diffusion_loss(
    positive_samples, selected_rows
):
    data = trajectories()
    batch = build(
        data,
        config(
            positive_samples=positive_samples,
            mask_probability_min=1.0,
            mask_probability_max=1.0,
        ),
    )
    expected = torch.zeros(4, 8, dtype=torch.bool)
    expected[selected_rows, :7] = data["token_mask"][selected_rows].bool()
    assert torch.equal(batch["diffusion_loss_mask"], expected)
    tail = torch.zeros_like(expected)
    tail[0, 7] = True
    tail[1, 5:8] = True
    assert torch.equal(batch["masked_indices"], expected | tail)
    assert torch.equal(batch["target_ids"][:, :7], data["input_ids"])
    assert batch["input_ids"] is data["input_ids"]
    assert batch["clean_input_ids"] is data["input_ids"]
    assert batch["target_ids"].shape == (4, 8)
    assert batch["clean_input_ids"].shape == (4, 7)
    assert torch.equal(batch["noisy_valid_lengths"], torch.tensor([8, 8, 4, 8]))
    assert (batch["prompt_lengths"] == 0).all()
    assert batch["token_mask"] is data["token_mask"]
    assert batch["advantages"] is data["advantages"]
    torch.testing.assert_close(batch["input_lengths"], data["input_lengths"])
    assert not batch["diffusion_loss_mask"][:, 7:].any()
    # Building an auxiliary view must not mutate the rollout batch.
    assert data["input_ids"].shape == (4, 7)


def test_partial_masks_are_reproducible_and_leave_visible_response_tokens():
    data = trajectories()
    data["input_ids"] = torch.arange(256).repeat(4, 1)
    data["token_mask"] = torch.ones(4, 256)
    data["token_mask"][:, :10] = 0
    data["advantages"] = torch.tensor([1.0, -1.0, 0.0, 1.0])
    data["input_lengths"] = torch.full((4,), 256)
    cfg = config(mask_probability_min=0.5, mask_probability_max=0.5)
    first, second = build(data, cfg), build(data, cfg)
    assert torch.equal(first["masked_indices"], second["masked_indices"])
    assert 0 < first["masked_indices"].sum() < 246
    assert not first["masked_indices"][1:].any()
    assert not first["masked_indices"][:, :10].any()
    torch.testing.assert_close(
        first["diffusion_mask_probability"], torch.full((4,), 0.5)
    )


def test_same_position_ce_gradient_and_inverse_probability_weighting():
    cfg = config(weight=0.2, mask_probability_min=1.0, mask_probability_max=1.0)
    batch = build(trajectories(), cfg)
    batch["token_mask"] = batch["diffusion_loss_mask"]
    batch["diffusion_mask_probability"].fill_(0.5)
    logits = torch.randn(4, 8, 32, requires_grad=True)
    selected = (
        logits.log_softmax(-1).gather(-1, batch["target_ids"].unsqueeze(-1)).squeeze(-1)
    )
    denominator = torch.tensor(10.0)
    loss, metrics = PositiveDiffusionLoss(cfg)(
        batch,
        torch.tensor(3.0),
        denominator,
        next_token_logprobs=selected,
        shift_labels=False,
    )
    expected = 0.2 * (-selected[0, [2, 3, 5, 6]]).sum() / 0.5 / 10
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, [2, 3, 5, 6]].abs().sum() > 0
    assert not logits.grad[1:].any()
    assert not logits.grad[0, [0, 1, 4, 7]].any()
    assert all(isinstance(value, (int, float)) for value in metrics.values())
    assert metrics["diffusion_aux_masked_tokens"] == 4
    assert metrics["diffusion_aux_positive_samples"] == 1


def test_empty_positive_batch_has_finite_zero_loss_and_gradient():
    data = trajectories()
    data["advantages"].zero_()
    batch = build(data, config())
    batch["token_mask"] = batch["diffusion_loss_mask"]
    scores = torch.randn(4, 8, requires_grad=True)
    loss, metrics = PositiveDiffusionLoss(config())(
        batch,
        torch.tensor(0.0),
        torch.tensor(0.0),
        next_token_logprobs=scores,
        shift_labels=False,
    )
    assert loss == 0 and torch.isfinite(loss)
    loss.backward()
    assert not scores.grad.any()
    assert metrics["diffusion_aux_positive_samples"] == 0


def test_sum_of_streaming_chunk_gradients_matches_whole_batch():
    cfg = config(mask_probability_min=1.0, mask_probability_max=1.0)
    data = trajectories()
    batch = build(data, cfg)
    batch["token_mask"] = batch["diffusion_loss_mask"]
    denominator = (data["token_mask"][:, 1:] * data["sample_mask"][:, None]).sum()
    full_scores = torch.randn(4, 8, requires_grad=True)
    full_loss, _ = PositiveDiffusionLoss(cfg)(
        batch,
        torch.tensor(3.0),
        denominator,
        next_token_logprobs=full_scores,
        shift_labels=False,
    )
    full_loss.backward()
    chunk_scores = full_scores.detach().clone().requires_grad_()
    for start in range(4):
        chunk = batch.slice(start, start + 1)
        raw, _ = PositiveDiffusionLoss(cfg)(
            chunk,
            torch.tensor(1.0),
            torch.tensor(1.0),
            next_token_logprobs=chunk_scores[start : start + 1],
            shift_labels=False,
        )
        (raw / denominator).backward()
    torch.testing.assert_close(chunk_scores.grad, full_scores.grad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight": -1},
        {"mask_probability_min": 0},
        {"mask_probability_min": 0.8, "mask_probability_max": 0.2},
        {"positive_samples": "best"},
        {"weight": float("nan")},
    ],
)
def test_invalid_auxiliary_configuration_fails(kwargs):
    with pytest.raises(ValidationError):
        config(**kwargs)


def test_recipe_preserves_causal_rollouts_and_selects_new_worker():
    register_omegaconf_resolvers()
    cfg = load_config(
        Path(__file__).resolve().parents[2] / "configs/positive_diffusion.yaml"
    )
    PositiveDiffusionConfig.model_validate(
        OmegaConf.to_container(cfg.policy.diffusion_aux_loss)
    )
    assert (
        cfg.policy.worker_extension_cls_fqn
        == "ar_grpo.positive_diffusion_policy.PositiveDiffusionARPolicy"
    )
    assert cfg.policy.generation.vllm_kwargs.diffusion_config is None
    assert list(cfg.policy.generation.vllm_kwargs.hf_overrides.architectures) == [
        "NemotronLabsDiffusionForCausalLM"
    ]
    assert cfg.loss_fn.token_level_loss


def test_auxiliary_config_survives_controller_policy_validation():
    class ControllerConfig(BaseModel, extra="allow"):
        policy: PolicyConfig

    register_omegaconf_resolvers()
    cfg = load_config(
        Path(__file__).resolve().parents[2] / "configs/positive_diffusion.yaml"
    )
    validated = ControllerConfig(
        policy=OmegaConf.to_container(cfg.policy, resolve=True)
    )
    parsed = PositiveDiffusionConfig.model_validate(
        validated.policy["diffusion_aux_loss"]
    )
    assert parsed.weight == 0.1
    assert not validated.policy["draft"].enabled
