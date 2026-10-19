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
"""Require saved/resumed checkpoints, finite metrics, and initial replay parity."""

import argparse
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.distributed.checkpoint import FileSystemReader


def completed_step(run: Path, minimum: int) -> dict:
    checkpoints = [
        p
        for p in (run / "checkpoints").glob("step_*")
        if p.name.removeprefix("step_").isdigit()
    ]
    if not checkpoints:
        raise RuntimeError(f"No finalized checkpoint under {run}")
    latest = max(checkpoints, key=lambda p: int(p.name.removeprefix("step_")))
    info = json.loads((latest / "training_info.json").read_text())
    if info["total_steps"] < minimum:
        raise RuntimeError(f"{latest} has only {info['total_steps']} updates")
    # Native Megatron stores optimizer tensors in the weights DCP, even though
    # Policy.save_checkpoint also receives an optimizer_path argument.
    weights = latest / "policy" / "weights"
    iteration = int((weights / "latest_checkpointed_iteration.txt").read_text())
    path = weights / f"iter_{iteration:07d}"
    metadata = FileSystemReader(path).read_metadata()
    names = metadata.state_dict_metadata
    if "embedding.word_embeddings.weight" not in names:
        raise RuntimeError(f"Missing model tensors: {path}")
    for moment in ("exp_avg", "exp_avg_sq"):
        if not any(
            name.startswith("optimizer.") and name.endswith("." + moment)
            for name in names
        ):
            raise RuntimeError(f"Missing optimizer {moment}: {path}")
    required_sizes: dict[str, int] = {}
    for entry in metadata.storage_data.values():
        required_sizes[entry.relative_path] = max(
            required_sizes.get(entry.relative_path, 0), entry.offset + entry.length
        )
    for filename, size in required_sizes.items():
        if (path / filename).stat().st_size < size:
            raise RuntimeError(f"Incomplete checkpoint shard: {path / filename}")
    return {"checkpoint": str(latest), "total_steps": info["total_steps"]}


def verify(run: Path, *, auxiliary: bool) -> dict:
    report = completed_step(run, 3)
    scalars: dict[str, list[float]] = {}
    for logs in (run, run.with_name(run.name + "_resume")):
        for path in logs.rglob("events.out.tfevents.*"):
            accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
            accumulator.Reload()
            for tag in accumulator.Tags()["scalars"]:
                scalars.setdefault(tag, []).extend(
                    e.value for e in accumulator.Scalars(tag)
                )
    for tag in (
        "train/loss",
        "train/grad_norm",
        "train/gen_kl_error",
        "validation/accuracy",
    ):
        values = scalars.get(tag)
        if not values or not all(math.isfinite(v) for v in values):
            raise RuntimeError(f"Missing/nonfinite smoke metric {tag}")
    if max(abs(v) for v in scalars["train/gen_kl_error"]) > 0.1:
        raise RuntimeError("Smoke generation/training KL exceeds 0.1")
    if max(scalars["train/grad_norm"]) <= 0:
        raise RuntimeError("Smoke did not exercise a nonzero policy gradient")
    if auxiliary:
        for tag in ("train/diffusion_aux_loss", "train/diffusion_aux_masked_tokens"):
            values = scalars.get(tag)
            if (
                not values
                or not all(math.isfinite(v) for v in values)
                or max(values) <= 0
            ):
                raise RuntimeError(f"Auxiliary objective was not exercised: {tag}")
    records = list(run.rglob("train_data_step1.jsonl"))
    if len(records) != 1:
        raise RuntimeError("Expected one first-update rollout log for replay parity")
    errors = []
    for line in records[0].read_text().splitlines():
        row = json.loads(line)
        if row["sample_loss_mask"][0] <= 0:
            continue
        for previous, generated, mask in zip(
            row["prev_logprobs"][0],
            row["generation_logprobs"][0],
            row["token_loss_mask"][0],
            strict=True,
        ):
            if mask:
                errors.append(abs(previous - generated))
    if not errors or not all(math.isfinite(v) for v in errors):
        raise RuntimeError("Missing/nonfinite first-update replay log probabilities")
    mean_error, max_error = sum(errors) / len(errors), max(errors)
    if mean_error > 0.1 or max_error > 1.0:
        raise RuntimeError(
            f"Generation/replay mismatch: mean={mean_error}, max={max_error}"
        )
    report.update(
        mean_abs_initial_logprob_error=mean_error,
        max_abs_initial_logprob_error=max_error,
        metrics={
            k: v for k, v in scalars.items() if k.startswith(("train/", "validation/"))
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--checkpoint-only", choices=["justgrpo", "diffaux"])
    args = parser.parse_args()
    if args.checkpoint_only:
        print(
            json.dumps(completed_step(args.root / ("smoke_" + args.checkpoint_only), 2))
        )
        return
    result = {
        name: verify(args.root / ("smoke_" + name), auxiliary=name == "diffaux")
        for name in ("justgrpo", "diffaux")
    }
    (args.root / "sanity_report.json").write_text(json.dumps(result, indent=2))
    print("LLaDA smoke and resume checks passed for both variants", flush=True)


if __name__ == "__main__":
    main()
