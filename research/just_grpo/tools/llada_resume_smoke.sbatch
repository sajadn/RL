#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
set -euo pipefail
: "${PAIR_ROOT:?}" "${VARIANT:?}" "${REPO_DIR:?}" "${POLICY_PYTHON:?}"
"$POLICY_PYTHON" "$REPO_DIR/research/just_grpo/tools/verify_llada_smoke.py" "$PAIR_ROOT" --checkpoint-only "$VARIANT"
exec bash "$REPO_DIR/research/just_grpo/tools/llada_distributed.sbatch"
