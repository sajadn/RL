---
name: gsm8k-checkpoint-eval
description: Evaluate NeMo-RL/Nemotron diffusion or Ministral3 RL checkpoints on GSM8K by first converting Megatron checkpoints to Hugging Face format, then running eval_3b_checkpoint.sh, including sbatch and Ministral3 conversion details.
---

# GSM8K Checkpoint Evaluation

Use this workflow when evaluating a saved NeMo-RL Megatron checkpoint on the SGLang GSM8K benchmark. The checkpoint must be converted to Hugging Face format before evaluation.

## Workflow

1. Work from the JustGRPO worktree:

```bash
cd /home/snorouzi/diffusion_RL/JustGRPO-worktree
```

2. Convert the checkpoint to HF format with `convert_checkpoint.sh`.

For native Nemotron diffusion checkpoints:

```bash
STEP_DIR=/path/to/run/checkpoints/step_N \
OUT=/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/<name>_hf \
./convert_checkpoint.sh
```

For `Ministral3ForCausalLM` / `ministral3_ar` checkpoints, pass `--ministral3` so the Megatron-Bridge runtime patch is added:

```bash
STEP_DIR=/path/to/run/checkpoints/step_N \
OUT=/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/<name>_hf \
./convert_checkpoint.sh --ministral3
```

The `--ministral3` flag is needed because the causal `Ministral3ForCausalLM` bridge is registered through the runtime patch directory.

3. Evaluate the converted HF checkpoint with `eval_3b_checkpoint.sh`.

```bash
MODEL=/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/<name>_hf \
OUTDIR=/lustre/fsw/portfolios/coreai/users/snorouzi/eval_results/<name>_gsm8k \
./eval_3b_checkpoint.sh
```

To submit the same evaluation as a Slurm batch job, add `--sbatch`:

```bash
MODEL=/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/<name>_hf \
OUTDIR=/lustre/fsw/portfolios/coreai/users/snorouzi/eval_results/<name>_gsm8k \
./eval_3b_checkpoint.sh --sbatch
```

## Notes

- `STEP_DIR` must point at a checkpoint step directory containing `config.yaml` and `policy/weights/iter_0000000`.
- `OUT` is the converted HF model directory passed later as `MODEL`.
- `eval_3b_checkpoint.sh` accepts `MODEL` and `OUTDIR` as environment variables.
- For diffusion-style evaluation, the converted HF checkpoint should use a diffusion-compatible HF template/base model and the eval script should run with its diffusion settings.
