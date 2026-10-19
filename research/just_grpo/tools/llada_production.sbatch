#!/usr/bin/env bash
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
set -euo pipefail
: "${PAIR_ROOT:?}" "${REPO_DIR:?}" "${POLICY_PYTHON:?}"
"$POLICY_PYTHON" "$REPO_DIR/research/just_grpo/tools/verify_llada_smoke.py" "$PAIR_ROOT"
if [[ "${TRAINING_CONTINUATION:-0}" == 1 ]]; then
  python3 - <<'PY'
import json,os
from pathlib import Path
root=Path(os.environ["CHECKPOINT_ROOT"])
checkpoints=[p for p in root.glob("step_*") if p.name.removeprefix("step_").isdigit()]
if not checkpoints:
    raise SystemExit("Continuation requires a finalized production checkpoint")
latest=max(checkpoints,key=lambda p:int(p.name.removeprefix("step_")))
info=json.loads((latest/"training_info.json").read_text())
if info["total_steps"] < 1:
    raise SystemExit("Continuation needs at least one completed production update")
if info["total_steps"] >= 125:
    (root.parent/"training_complete").touch()
PY
  if [[ -f "${CHECKPOINT_ROOT%/*}/training_complete" ]]; then
    echo "125 updates completed; no continuation required."
    exit 0
  fi
fi
exec bash "$REPO_DIR/research/just_grpo/tools/llada_distributed.sbatch"
