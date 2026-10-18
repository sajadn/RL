#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_DIR"
# Pyxis may replace the submission environment library path.
export LD_LIBRARY_PATH="/lustre/fsw/portfolios/coreai/users/snorouzi/vllm_runtimes/cuda_compat_13.1/lib.real${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD=/lustre/fsw/portfolios/coreai/users/snorouzi/vllm_runtimes/nemotron_dllm_792ab07/lib/python3.13/site-packages/nvidia/nccl/lib/libnccl.so.2
export FLASHINFER_WORKSPACE_BASE="/tmp/ar_sc_${RUN_NAME}_${SLURM_JOB_ID:-local}/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="/tmp/ar_sc_${RUN_NAME}_${SLURM_JOB_ID:-local}/inductor"
export TRITON_CACHE_DIR="/tmp/ar_sc_${RUN_NAME}_${SLURM_JOB_ID:-local}/triton"
export WANDB_DIR="$RUNDIR/wandb" WANDB_CACHE_DIR="$RUNDIR/wandb_cache"
mkdir -p "$WANDB_DIR"
if [[ -z "${WANDB_API_KEY:-}" && -f /home/snorouzi/wandb_api_key.txt ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '\n\r' < /home/snorouzi/wandb_api_key.txt)"
fi
if [[ "${CASCADE2_SWE_RUN:-0}" == 1 ]]; then
  # The Gym preflight uses its own dependencies; training overlays require policy-only packages.
  PYTHONPATH="$REPO_DIR:$GYM_SOURCE_DIR" \
    "$GYM_SOURCE_DIR/.venv/bin/python" research/ar_grpo/preflight_cascade2_swe.py --config "$CONFIG" --check-apptainer
fi
# Fail if the provisioned interpreter imports the old RL checkout.
"$POLICY_PYTHON" - <<'CHECK'
import ctypes
import os
from pathlib import Path

import nemo_rl
import transfer_queue

version = ctypes.c_int()
status = ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(version))
assert status == 0 and version.value >= 13000, (status, version.value)
print("AR_SC_CUDA_DRIVER", version.value, flush=True)
repo_dir = Path(os.environ["REPO_DIR"]).resolve()
assert Path(nemo_rl.__file__).resolve().is_relative_to(repo_dir)
print("AR_SC_IMPORT", nemo_rl.__file__, transfer_queue.__file__, flush=True)

# Load GPU dependencies only after checking CUDA compatibility.
from megatron.bridge.diffusion.models.common import nemotron_labs_diffusion_attention as cp_attention
from omegaconf import OmegaConf

from nemo_rl.models.megatron.data import _get_pack_sequence_parameters_for_megatron
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

bridge_source = repo_dir / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src"
assert Path(cp_attention.__file__).resolve().is_relative_to(bridge_source.resolve())
assert hasattr(cp_attention.NemotronLabsDiffusionAttention, "_packed_causal_forward")
print("AR_SC_BRIDGE", cp_attention.__file__, flush=True)
register_omegaconf_resolvers()
policy_cfg = OmegaConf.to_container(load_config(os.environ["CONFIG"]), resolve=True)["policy"]
packing = _get_pack_sequence_parameters_for_megatron(
    policy_cfg["megatron_cfg"],
    policy_cfg["make_sequence_length_divisible_by"],
    policy_cfg["max_total_sequence_length"],
)
print("AR_SC_PACKING_PREFLIGHT", packing, flush=True)
CHECK
if [[ "${RUN_AR_UNIT_TESTS:-0}" == 1 ]]; then
  bridge_dir="$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
  (
    cd "$bridge_dir"
    export PYTHONPATH="/lustre/fsw/portfolios/coreai/users/snorouzi/ar_sc_test_overlay_20261001:$PYTHONPATH"
    /home/snorouzi/.local/bin/uv --no-cache run --no-project --python "$POLICY_PYTHON" python -m pytest \
      --confcutdir=tests/unit_tests/diffusion/model/nemotron_labs_diffusion \
      tests/unit_tests/diffusion/model/nemotron_labs_diffusion/test_nemotron_labs_diffusion_attention.py \
      -k 'TestLazyDiffusionMask or TestNemotronLabsDiffusionAttentionInferenceForward or TestNemotronLabsDiffusionAttentionInit' -q
  )
fi
exec "$POLICY_PYTHON" "${RUN_SCRIPT:-research/ar_grpo/run_ar_grpo.py}" --config "$CONFIG"
