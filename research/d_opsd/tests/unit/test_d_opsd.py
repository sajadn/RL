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
"""Independent loss oracles and trajectory/conditioning invariants."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from d_opsd.algorithm import (
    DOPSD,
    DOPSDConfig,
    DOPSDSchedule,
    prepare_teacher_targets,
    select_correct_responses,
)
from d_opsd.config import validate_config
from d_opsd.loss import DOPSDLoss
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def config(**overrides):
    sampling = dict(
        temperature=1.0,
        block_size=4,
        max_steps=4,
        selection_policy="leftmost",
        threshold=0.9,
        returns_reveal_steps=True,
        returns_entropy=False,
        emit_full_blocks=True,
    )
    values = dict(
        schedule=dict(mask_token_id=31, block_size=4, seed_base=42),
        runtime="megatron",
        generation_python=None,
        max_generation_kl=0.05,
        distributed=True,
        sampling=sampling,
        validation_sampling=sampling,
        teacher_retain_ratio=0.5,
        reward_threshold=1.0,
        correct_only=True,
        selection_source="teacher",
        pointwise_clip=0.05,
    )
    values.update(overrides)
    return DOPSDConfig(**values)


def batch(*, stops=(), rewards=(1.0, 0.0), group_size=1):
    # One response has 2 + 3 transitions across its two blocks, the other 4 + 1.
    ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    )
    steps = torch.tensor([[0, 0, 1, 1, 0, 1, 1, 2], [0, 1, 2, 3, 0, 0, 0, 0]])
    mask = torch.ones_like(ids)
    mask[:, :2] = 0
    data = BatchedDataDict(
        input_ids=ids,
        input_lengths=torch.tensor([10, 10]),
        token_mask=mask,
        sample_mask=torch.ones(2),
        total_rewards=torch.tensor(rewards),
    )
    logs = [
        [
            dict(role="user", token_ids=ids[row, :2]),
            dict(role="assistant", token_ids=ids[row, 2:], reveal_steps=steps[row]),
        ]
        for row in range(2)
    ]
    DOPSD(
        config(), stop_token_ids=list(stops), group_size=group_size
    ).prepare_training_data(data, logs, 3)
    return data


def targets(schedule, *, selection="student"):
    generator = torch.Generator().manual_seed(27)
    for level in schedule.iter_levels():
        logits = torch.randn(*level["input_ids"].shape, 32, generator=generator)
        yield prepare_teacher_targets(
            level, logits, block_size=4, selection_source=selection
        )


def test_first_correct_selection_and_no_advantage_dependence():
    rewards = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    masks = torch.ones(8)
    chosen = select_correct_responses(
        rewards, masks, group_size=4, threshold=1.0, correct_only=True
    )
    torch.testing.assert_close(
        chosen, torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    masks[1] = 0
    chosen = select_correct_responses(
        rewards, masks, group_size=4, threshold=1.0, correct_only=True
    )
    assert chosen[2] == 1
    torch.testing.assert_close(
        select_correct_responses(
            rewards, masks, group_size=4, threshold=1.0, correct_only=False
        ),
        masks,
    )


@pytest.mark.parametrize("rewards", [torch.tensor([float("nan"), 1.0]), torch.ones(3)])
def test_invalid_verification_rewards(rewards):
    with pytest.raises(ValueError):
        select_correct_responses(
            rewards,
            torch.ones_like(rewards),
            group_size=2,
            threshold=1.0,
            correct_only=True,
        )


def test_step_weights_sum_to_one_per_accepted_response():
    data = batch(rewards=(1.0, 1.0))
    schedule = DOPSDSchedule(data, config(), pad_token_id=0)
    torch.testing.assert_close(schedule.transition_counts, torch.tensor([5, 5]))
    total = sum(t["token_mask"].sum(1) for t in targets(schedule))
    torch.testing.assert_close(total, torch.ones(2))


def test_teacher_is_nested_deterministic_and_keeps_k_masks():
    schedule = DOPSDSchedule(batch(rewards=(1.0, 1.0)), config(), pad_token_id=0)
    for index in range(schedule.num_steps):
        level = schedule.generate_single_trajectory(index)
        other = schedule.generate_single_trajectory(index)
        torch.testing.assert_close(
            level["dopsd_teacher_mask"], other["dopsd_teacher_mask"]
        )
        assert not (level["dopsd_teacher_mask"] & ~level["masked_indices"]).any()
        for start in (0, 4):
            count = level["dopsd_committed"][:, start : start + 4].sum(1)
            assert (level["dopsd_eligible"][:, start : start + 4].sum(1) >= count).all()
    first = schedule.generate_single_trajectory(0)
    assert (first["masked_indices"] & ~first["dopsd_teacher_mask"]).any()


def test_eos_trims_loss_but_preserves_committed_tail_context():
    data = batch(stops=(5,), rewards=(1.0, 1.0))
    assert data["dopsd_loss_mask"][:, :5].sum() == 6  # response tokens 3,4,5
    assert not data["dopsd_loss_mask"][:, 5:].any()
    schedule = DOPSDSchedule(data, config(), pad_token_id=0)
    level = schedule.generate_single_trajectory(2)
    assert not level["masked_indices"][0, 4:7].any()
    for target in targets(schedule, selection="teacher"):
        assert not target["token_mask"][:, 3:].any()
    torch.testing.assert_close(
        sum(t["token_mask"].sum(1) for t in targets(schedule)), torch.ones(2)
    )


def test_rejected_responses_have_zero_loss_weight():
    schedule = DOPSDSchedule(batch(), config(), pad_token_id=0)
    for target in targets(schedule):
        assert not target["token_mask"][1].any()
        assert not target["dopsd_target_weights"][1].any()


def test_teacher_confidence_uses_probabilities_and_excludes_revealed_positions():
    schedule = DOPSDSchedule(
        batch(rewards=(1.0, 1.0)), config(teacher_retain_ratio=0.0), pad_token_id=0
    )
    level = schedule.generate_single_trajectory(0)
    logits = torch.zeros(2, 8, 32)
    # Raw logits are large but uniformly distributed: confidence must remain low.
    logits[:, 0] = 100
    logits[:, 1:, 0] = 5
    result = prepare_teacher_targets(
        level, logits, block_size=4, selection_source="teacher"
    )
    assert not result["token_mask"][:, 0].any()
    assert not (result["token_mask"].bool() & ~level["dopsd_eligible"]).any()
    assert result["dopsd_teacher_logits"].shape == (2, 8, 32)


@pytest.mark.parametrize("clip", [None, 0.05])
def test_reverse_kl_and_gradients_match_vocabulary_entry_oracle(clip):
    student = torch.tensor([[[1.0, -1.0, 0.0], [0.0, 3.0, -2.0]]], requires_grad=True)
    teacher = torch.tensor([[[0.0, 1.0, -2.0], [1.0, -2.0, 3.0]]], requires_grad=True)
    data = BatchedDataDict(
        sample_mask=torch.ones(1),
        dopsd_target_weights=torch.tensor([[0.4, 0.6]]),
        dopsd_teacher_logits=teacher,
    )
    fn = DOPSDLoss(pointwise_clip=clip)
    loss, metrics = fn(data, torch.tensor(1.0), torch.tensor(1.0), logits=student)
    p = student.softmax(-1)
    q = teacher.detach().softmax(-1)
    terms = p * (p.log() - q.log())
    expected = (
        (terms if clip is None else terms.clamp(max=clip)).sum(-1)
        * data["dopsd_target_weights"]
    ).sum()
    torch.testing.assert_close(loss, expected)
    expected_grad = torch.autograd.grad(expected, student, retain_graph=True)[0]
    loss.backward()
    torch.testing.assert_close(student.grad, expected_grad)
    assert teacher.grad is None
    # The GRPO controller aggregates worker metrics through NumPy, including on CUDA.
    assert all(isinstance(value, float) for value in metrics.values())
    assert np.isfinite(np.asarray(list(metrics.values()), dtype=float)).all()


def test_empty_loss_is_differentiable_zero_even_with_nan_unused_targets():
    student = torch.randn(2, 4, 3, requires_grad=True)
    data = BatchedDataDict(
        sample_mask=torch.ones(2),
        dopsd_target_weights=torch.zeros(2, 4),
        dopsd_teacher_logits=torch.full((2, 4, 3), float("nan")),
    )
    loss, _ = DOPSDLoss(pointwise_clip=0.05)(
        data, torch.tensor(0.0), torch.tensor(0.0), logits=student
    )
    loss.backward()
    assert loss == 0
    assert student.grad is not None and not student.grad.any()


def test_streamed_gradients_equal_independent_per_block_step_average():
    torch.manual_seed(3)
    student = torch.nn.Parameter(torch.randn(2, 8, 32))
    schedule = DOPSDSchedule(batch(rewards=(1.0, 1.0)), config(), pad_token_id=0)
    records = list(targets(schedule))
    fn = DOPSDLoss(pointwise_clip=None)
    streamed = sum(
        fn(t, torch.tensor(2.0), torch.tensor(2.0), logits=student)[0] for t in records
    )
    reference = student.sum() * 0
    for row in range(2):
        step_losses = []
        for target in records:
            p = student[row].log_softmax(-1)
            q = target["dopsd_teacher_logits"][row].log_softmax(-1)
            divergence = (p.exp() * (p - q)).sum(-1)
            for start in (0, 4):
                committed = target["dopsd_committed"][row, start : start + 4]
                if committed.any():
                    step_losses.append(divergence[start : start + 4][committed].mean())
        reference = reference + torch.stack(step_losses).mean() / 2
    torch.testing.assert_close(streamed, reference)
    g1 = torch.autograd.grad(streamed, student, retain_graph=True)[0]
    g2 = torch.autograd.grad(reference, student)[0]
    torch.testing.assert_close(g1, g2)


@pytest.mark.parametrize(
    "recipe",
    [
        "d_opsd.yaml",
        "recipes/d_opsd-sudoku6x6-1n8g-megatron-smoke.yaml",
        "recipes/d_opsd-sudoku6x6-4n8g-megatron-vllm.yaml",
        "recipes/d_opsd-deepscaler-3b-4n8g-megatron-vllm-dualval-long.yaml",
        "recipes/d_opsd-deepscaler-3b-1n8g-megatron-vllm-dualval-smoke.yaml",
    ],
)
def test_recipes_and_reject_unsupported_modes(recipe):
    register_omegaconf_resolvers()
    base = load_config(Path(__file__).resolve().parents[2] / "configs" / recipe)
    OmegaConf.resolve(base)
    assert validate_config(base).teacher_retain_ratio == 0.25
    # The upstream controller constructs its GRPO loss before our worker replaces it.
    from nemo_rl.algorithms.loss.loss_functions import (
        ClippedPGLossConfig,
        ClippedPGLossFn,
    )

    ClippedPGLossFn(
        ClippedPGLossConfig(**OmegaConf.to_container(base.loss_fn, resolve=True))
    )
    for key, value in [
        ("policy.megatron_cfg.tensor_model_parallel_size", 2),
        ("grpo.async_grpo.enabled", True),
        ("loss_fn.force_on_policy_ratio", False),
        ("loss_fn.use_importance_sampling_correction", True),
        ("loss_fn.truncated_importance_sampling_type", "tis"),
    ]:
        invalid = copy.deepcopy(base)
        OmegaConf.update(invalid, key, value)
        with pytest.raises(ValueError):
            validate_config(invalid)


def test_generated_trajectory_drives_student_update_with_frozen_teacher():
    from test_schedule import TinyDiffusion
    from test_megatron_generation import TinyDoubled
    from block_diffusion.generation.megatron_generation import sample_batch

    torch.manual_seed(91)
    student = TinyDiffusion()
    teacher = copy.deepcopy(student)
    teacher_state = copy.deepcopy(teacher.state_dict())
    settings = config()
    prompts = torch.tensor([[1, 2], [1, 2]])
    with torch.no_grad():
        responses = sample_batch(
            TinyDoubled(student),
            prompts,
            sampling=settings.sampling,
            max_new_tokens=8,
            max_sequence_length=12,
            mask_token_id=31,
            stop_token_ids=[],
            generator=torch.Generator().manual_seed(42),
        )
    ids = torch.tensor(
        [
            prompt.tolist() + response["token_ids"]
            for prompt, response in zip(prompts, responses)
        ]
    )
    mask = torch.ones_like(ids)
    mask[:, :2] = 0
    data = BatchedDataDict(
        input_ids=ids,
        input_lengths=torch.tensor([10, 10]),
        token_mask=mask,
        sample_mask=torch.ones(2),
        total_rewards=torch.ones(2),
    )
    messages = [
        [
            dict(role="user", token_ids=prompt),
            dict(
                role="assistant",
                token_ids=torch.tensor(response["token_ids"]),
                reveal_steps=torch.tensor(response["reveal_steps"]),
            ),
        ]
        for prompt, response in zip(prompts, responses)
    ]
    DOPSD(settings, stop_token_ids=[], group_size=2).prepare_training_data(
        data, messages, 1
    )
    assert data["sample_mask"].tolist() == [1.0, 0.0]
    schedule = DOPSDSchedule(data, settings, pad_token_id=0)

    def logits(model, view):
        noisy = view["input_ids"].masked_fill(view["masked_indices"], 31)
        clean = view["clean_input_ids"]
        positions = torch.cat(
            [
                view["position_ids"] + view["prompt_lengths"][:, None],
                torch.arange(clean.shape[1])[None].expand_as(clean),
            ],
            1,
        )
        return TinyDoubled(model)(torch.cat([noisy, clean], 1), positions)[
            :, : noisy.shape[1]
        ]

    objective = student.head.weight.sum() * 0
    for level in schedule.iter_levels():
        teacher_view = BatchedDataDict(level)
        teacher_view["masked_indices"] = level["dopsd_teacher_mask"]
        with torch.no_grad():
            teacher_logits = logits(teacher, teacher_view)
        target = prepare_teacher_targets(
            level, teacher_logits, block_size=4, selection_source="teacher"
        )
        loss, _ = DOPSDLoss(pointwise_clip=0.05)(
            target, torch.tensor(1.0), torch.tensor(1.0), logits=logits(student, level)
        )
        objective = objective + loss
    assert torch.isfinite(objective) and objective > 0
    objective.backward()
    torch.optim.SGD(student.parameters(), lr=0.05).step()
    assert any(
        not torch.equal(student.state_dict()[key], teacher_state[key])
        for key in teacher_state
    )
    for key in teacher_state:
        torch.testing.assert_close(teacher.state_dict()[key], teacher_state[key])
    assert all(parameter.grad is None for parameter in teacher.parameters())
