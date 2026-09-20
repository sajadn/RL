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

"""Tests that the worker scores each quadrature point against its own mask.

These run on CPU with no distributed backend: the model forward is stubbed so
the test can observe exactly what each quadrature point was handed.
"""

import contextlib
import types
from dataclasses import dataclass

import pytest
import torch
from gdpo import SdmcElboEstimator, SdmcLikelihoodConfig
from gdpo.models.policy.workers.gdpo_worker import DTensorGDPOPolicyWorker

# The worker is a Ray actor class; the plain class carrying the methods sits
# behind __ray_metadata__ and is what we bind the method under test from.
_WORKER_CLS = DTensorGDPOPolicyWorker.__ray_metadata__.modified_class

MASK_ID = 99
PAD_ID = 0


class FakeTokenizer:
    pad_token_id = PAD_ID


@dataclass
class FakeProcessedInputs:
    input_ids: torch.Tensor
    position_ids: torch.Tensor | None = None


@dataclass
class FakeMicrobatch:
    processed_inputs: FakeProcessedInputs
    data_dict: dict


def make_worker(quadrature="gauss-3", mc_samples=1):
    """A stub carrying only what _gdpo_elbo_logprobs actually touches."""
    worker = types.SimpleNamespace(
        model=object(),
        device_mesh=None,
        cp_size=1,
        tokenizer=FakeTokenizer(),
        allow_flash_attn_args=False,
        sampling_params=None,
        elbo_estimator=SdmcElboEstimator(
            SdmcLikelihoodConfig(quadrature=quadrature, mc_samples=mc_samples),
            MASK_ID,
        ),
        _autocast_context=lambda: contextlib.nullcontext(),
    )
    worker._gdpo_elbo_logprobs = types.MethodType(
        _WORKER_CLS._gdpo_elbo_logprobs, worker
    )
    return worker


@pytest.fixture
def batch():
    # Row layout: two prompt tokens then four scorable completion tokens.
    input_ids = torch.tensor([[5, 6, 11, 12, 13, 14], [7, 8, 21, 22, 23, 24]])
    token_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]], dtype=torch.bool
    )
    return FakeMicrobatch(FakeProcessedInputs(input_ids), {"token_mask": token_mask})


@pytest.fixture
def spy(monkeypatch):
    """Record the input_ids each forward is prepared with."""
    seen = []

    def fake_prepare(model, processed_inputs, **kwargs):
        seen.append(processed_inputs.input_ids.clone())
        return types.SimpleNamespace(
            model_context_factory=lambda: contextlib.nullcontext()
        )

    def fake_forward(**kwargs):
        mb = kwargs["processed_mb"]
        return (
            torch.zeros_like(mb.processed_inputs.input_ids, dtype=torch.float32),
            {},
            mb,
        )

    monkeypatch.setattr(
        "gdpo.models.policy.workers.gdpo_worker.prepare_model_forward", fake_prepare
    )
    monkeypatch.setattr(
        "gdpo.models.policy.workers.gdpo_worker.forward_with_post_processing_fn",
        fake_forward,
    )
    return seen


class TestQuadratureIsolation:
    def test_one_forward_is_prepared_per_quadrature_point(self, batch, spy):
        make_worker("gauss-3")._gdpo_elbo_logprobs(
            processed_mb=batch, post_processing_fn=None, sequence_dim=1
        )
        assert len(spy) == 3

    def test_each_point_is_prepared_against_its_own_masked_view(self, batch, spy):
        """Each point must see its own mask.

        Hoisting prepare_model_forward out of score_fn would snapshot the clean
        sequence for every point.
        """
        make_worker("gauss-3")._gdpo_elbo_logprobs(
            processed_mb=batch, post_processing_fn=None, sequence_dim=1
        )
        clean = torch.tensor([[5, 6, 11, 12, 13, 14], [7, 8, 21, 22, 23, 24]])
        for prepared_ids in spy:
            assert not torch.equal(prepared_ids, clean), (
                "a quadrature point was prepared against the unmasked sequence"
            )
            assert (prepared_ids == MASK_ID).any()

    def test_prompt_positions_are_never_masked(self, batch, spy):
        make_worker("gauss-3")._gdpo_elbo_logprobs(
            processed_mb=batch, post_processing_fn=None, sequence_dim=1
        )
        for prepared_ids in spy:
            assert (prepared_ids[:, :2] != MASK_ID).all()

    def test_more_points_prepare_more_forwards(self, batch, spy):
        make_worker("gauss-5")._gdpo_elbo_logprobs(
            processed_mb=batch, post_processing_fn=None, sequence_dim=1
        )
        assert len(spy) == 5

    def test_mc_samples_multiply_the_forward_count(self, batch, spy):
        make_worker("gauss-2", mc_samples=3)._gdpo_elbo_logprobs(
            processed_mb=batch, post_processing_fn=None, sequence_dim=1
        )
        assert len(spy) == 6


class TestInputRestoration:
    def test_the_microbatch_is_left_clean_afterwards(self, batch, spy):
        before = batch.processed_inputs.input_ids.clone()
        make_worker()._gdpo_elbo_logprobs(
            processed_mb=batch, post_processing_fn=None, sequence_dim=1
        )
        assert torch.equal(batch.processed_inputs.input_ids, before)
        assert batch.processed_inputs.position_ids is None

    def test_inputs_are_restored_even_when_the_forward_raises(self, batch, monkeypatch):
        before = batch.processed_inputs.input_ids.clone()

        monkeypatch.setattr(
            "gdpo.models.policy.workers.gdpo_worker.prepare_model_forward",
            lambda *a, **k: types.SimpleNamespace(
                model_context_factory=lambda: contextlib.nullcontext()
            ),
        )

        def boom(**kwargs):
            raise RuntimeError("forward exploded")

        monkeypatch.setattr(
            "gdpo.models.policy.workers.gdpo_worker.forward_with_post_processing_fn",
            boom,
        )
        with pytest.raises(RuntimeError, match="forward exploded"):
            make_worker()._gdpo_elbo_logprobs(
                processed_mb=batch, post_processing_fn=None, sequence_dim=1
            )
        assert torch.equal(batch.processed_inputs.input_ids, before)
        assert batch.processed_inputs.position_ids is None


class TestOutput:
    def test_the_result_keeps_the_batch_shape(self, batch, spy):
        out = make_worker()._gdpo_elbo_logprobs(
            processed_mb=batch, post_processing_fn=None, sequence_dim=1
        )
        assert out.shape == batch.processed_inputs.input_ids.shape


def test_actual_previous_and_training_paths_share_masked_scoring(monkeypatch):
    """Exercise get_logprobs, scorer construction, and the real loss/backward API."""
    from gdpo.algorithms.loss.gdpo import GDPOLossFn

    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch
    from nemo_rl.models.policy.workers import dtensor_policy_worker_v2 as base_worker

    class CpuData(BatchedDataDict):
        def to(self, device):
            # The test substitutes CPU placement, not the model/scoring paths.
            assert device == "cuda"
            return self

    class TinyDiffusionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Embedding(100, 100)
            self.seen = []

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            # Deliberately has no position_ids argument, like LLaDA.
            self.seen.append(input_ids.clone())
            return types.SimpleNamespace(logits=self.table(input_ids))

    clean = torch.tensor([[5, 6, 11, 12, 13, 14]])
    token_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
    data = CpuData(
        {
            "input_ids": clean,
            "input_lengths": torch.tensor([6]),
            "token_mask": token_mask,
            "sample_mask": torch.ones(1),
            "advantages": token_mask,
            "generation_logprobs": torch.full_like(token_mask, float("nan")),
        }
    )
    mb = ProcessedMicrobatch(
        data_dict=data,
        processed_inputs=ProcessedInputs(
            input_ids=clean,
            seq_len=6,
            attention_mask=torch.ones_like(clean),
            position_ids=torch.arange(6).unsqueeze(0),
        ),
        original_batch_size=1,
        original_seq_len=6,
    )
    worker = object.__new__(_WORKER_CLS)
    worker.model = TinyDiffusionModel()
    worker.cfg = {"logprob_batch_size": 1, "logprob_chunk_size": None}
    worker.cp_size = 1
    worker.device_mesh = None
    worker.dp_mesh = None
    worker.enable_seq_packing = False
    worker.sampling_params = None
    worker.tokenizer = FakeTokenizer()
    worker.allow_flash_attn_args = False
    worker.elbo_estimator = SdmcElboEstimator(
        SdmcLikelihoodConfig(quadrature="gauss-3"), MASK_ID
    )
    worker.timer = types.SimpleNamespace(start=lambda _: None, stop=lambda _: None)
    worker._autocast_context = contextlib.nullcontext
    monkeypatch.setattr(
        base_worker, "get_microbatch_iterator", lambda *a, **k: (iter([mb]), 1)
    )

    previous = worker.get_logprobs(data)["logprobs"]
    current = worker._gdpo_train_elbo_scorer(1)(mb)
    torch.testing.assert_close(current, previous)
    assert current.requires_grad
    assert not previous.requires_grad
    assert len(worker.model.seen) == 6
    for first, second in zip(worker.model.seen[:3], worker.model.seen[3:]):
        torch.testing.assert_close(first, second)
        assert (first == MASK_ID).any()
    torch.testing.assert_close(mb.processed_inputs.input_ids, clean)
    assert mb.processed_inputs.position_ids is not None

    loss_fn = GDPOLossFn(
        ClippedPGLossConfig(
            reference_policy_kl_penalty=0,
            position_aligned_logprobs=True,
            sequence_level_importance_ratios=True,
            token_level_loss=False,
        )
    )
    data["prev_logprobs"] = previous
    post_processor = worker._make_loss_post_processor(
        cfg=worker.cfg,
        loss_fn=loss_fn,
        cp_mesh=None,
        cp_size=1,
        dp_size=1,
    )
    results = worker._forward_backward(
        data_iterator=iter([mb]),
        post_processing_fn=post_processor,
        sequence_dim=1,
        forward_only=False,
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(4.0),
        dp_size=1,
        cp_size=1,
        num_global_batches=1,
        train_context_fn=None,
        num_valid_microbatches=1,
        on_microbatch_start=None,
    )
    assert results[0][1]["probs_ratio"] == pytest.approx(1.0)
    assert worker.model.table.weight.grad.abs().sum() > 0


def test_generate_respects_each_prompts_remaining_sequence_budget(monkeypatch):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attended_lengths = []

        def forward(self, input_ids, attention_mask):
            self.attended_lengths.append(attention_mask.sum(dim=1).tolist())
            logits = torch.zeros(*input_ids.shape, 128)
            logits[..., 42] = 1
            return types.SimpleNamespace(logits=logits)

    worker = make_worker()
    worker.model = Model()
    worker.tokenizer.eos_token_id = 98
    worker.cfg = {
        "max_total_sequence_length": 6,
        "generation": {
            "max_new_tokens": 4,
            "temperature": 1.0,
            "top_k": None,
            "top_p": 1.0,
            "stop_token_ids": None,
        },
    }
    worker.dp_mesh = types.SimpleNamespace(get_local_rank=lambda: 0)
    worker.denoise_cfg = types.SimpleNamespace(
        diffusion_steps=4,
        block_length=2,
        cfg_scale=0.0,
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [[5, 6, 0, 0, 0, 0], [5, 6, 7, 8, 9, 0], [5, 6, 7, 8, 9, 10]]
            ),
            "input_lengths": torch.tensor([2, 5, 6]),
        }
    )
    output = _WORKER_CLS.generate(worker, data, greedy=True, seed=0)
    assert output["generation_lengths"].tolist() == [4, 1, 0]
    assert output["unpadded_sequence_lengths"].tolist() == [6, 6, 6]
    assert output["truncated"].tolist() == [True, True, True]
    assert worker.model.attended_lengths == [[6, 6, 6]] * 4
    worker.cfg["max_total_sequence_length"] = 5
    with pytest.raises(ValueError, match="prompt exceeds"):
        _WORKER_CLS.generate(worker, data, greedy=True, seed=0)
