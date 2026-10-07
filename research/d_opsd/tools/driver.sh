#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail
cd "${REPO_DIR}"
export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD=/lustre/fsw/portfolios/coreai/users/snorouzi/vllm_runtimes/nemotron_dllm_792ab07/lib/python3.13/site-packages/nvidia/nccl/lib/libnccl.so.2
export FLASHINFER_WORKSPACE_BASE="/tmp/d_opsd_${DOPSD_JOB_ID}/flashinfer"
"${POLICY_PYTHON}" - <<'PY'
from pathlib import Path
import inspect, os, subprocess
import torch, nemo_rl, d_opsd, block_diffusion, megatron.core
from omegaconf import OmegaConf
from d_opsd.config import validate_config
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from megatron.bridge.diffusion.models.common.nemotron_labs_diffusion_attention import NemotronLabsDiffusionAttention
root = Path(os.environ["REPO_DIR"]).resolve()
register_omegaconf_resolvers()
config = load_config(os.environ["CONFIG"])
OmegaConf.resolve(config)
validate_config(config)
assert config.cluster.num_nodes == int(os.environ["DOPSD_NUM_NODES"])
assert torch.cuda.is_available()
for package in (nemo_rl, d_opsd, block_diffusion):
    assert Path(package.__file__).resolve().is_relative_to(root), package.__file__
    print("DOPSD_IMPORT", package.__file__, flush=True)
bridge = Path("/lustre/fsw/portfolios/coreai/users/snorouzi/Megatron-Bridge-asymmetric-rl").resolve()
assert Path(inspect.getfile(NemotronLabsDiffusionAttention)).resolve().is_relative_to(bridge)
assert callable(NemotronLabsDiffusionAttention.build_asymmetric_ar_metadata)
assert Path(megatron.core.__file__).resolve().is_relative_to(bridge / "3rdparty/Megatron-LM")
print("DOPSD_COMMIT", subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(), flush=True)
print("DOPSD_PREFLIGHT", config.data.train.dataset_name, list(config.policy.generation.vllm_val_dllm_variants), flush=True)
PY
exec "${POLICY_PYTHON}" research/d_opsd/run_d_opsd.py --config "${CONFIG}"
