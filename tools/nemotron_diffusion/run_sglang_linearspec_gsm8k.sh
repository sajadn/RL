#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Run the SGLang fork's OpenAI-compatible GSM8K evaluator against a
# Nemotron-Diffusion dLLM server.

set -euo pipefail

SGLANG_REPO="${SGLANG_REPO:-/home/snorouzi/code/sglang-nemotron-dllm-lora-eval}"
VENV="${VENV:-/lustre/fsw/portfolios/coreai/users/snorouzi/sglang_nemotron_torch291_cu129_uvpy312_venv}"
MODEL="${MODEL:?MODEL must point to a HF checkpoint directory}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
LORA_PATH="${LORA_PATH:-}"
OUTDIR="${OUTDIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/eval_results/sglang_linearspec_gsm8k}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
CONCURRENT="${CONCURRENT:-1}"
PORT="${PORT:-31001}"
HOST="${HOST:-127.0.0.1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
DLLM_ALGORITHM="${DLLM_ALGORITHM:-LinearSpec}"
MAX_STEPS="${MAX_STEPS:-}"
THRESHOLD="${THRESHOLD:-}"
DTYPE="${DTYPE:-bfloat16}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS-1 2 4 8 16 32 64 128}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
DISABLE_PIECEWISE_CUDA_GRAPH="${DISABLE_PIECEWISE_CUDA_GRAPH:-0}"
INCLUDE_STATS_FILE="${INCLUDE_STATS_FILE:-1}"
DRY_RUN="${DRY_RUN:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUTDIR}"
SERVER_LOG="${OUTDIR}/server.log"
CLIENT_LOG="${OUTDIR}/client.log"
RESULTS_JSON="${OUTDIR}/results.json"
DLLM_CONFIG="${OUTDIR}/dllm_config.yaml"

export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"
export PYTHONPATH="${SGLANG_REPO}/python:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES

{
  echo "algorithm: ${DLLM_ALGORITHM}"
  echo "causal_context: true"
  if [[ "${DLLM_ALGORITHM}" != "AR" ]]; then
    echo "block_size: ${BLOCK_SIZE}"
    if [[ -n "${LORA_PATH}" ]]; then
      if [[ "${DLLM_ALGORITHM}" != "LinearSpec" ]]; then
        echo "WARNING: lora_path is currently consumed by LinearSpec only." >&2
      fi
      echo "lora_path: ${LORA_PATH}"
    fi
    if [[ -n "${MAX_STEPS}" ]]; then
      echo "max_steps: ${MAX_STEPS}"
    fi
    if [[ -n "${THRESHOLD}" ]]; then
      echo "threshold: ${THRESHOLD}"
    fi
  fi
  if [[ "${INCLUDE_STATS_FILE}" == "1" ]]; then
    echo "stats_file: ${OUTDIR}/stats.jsonl"
  fi
} >"${DLLM_CONFIG}"

SERVER_ARGS=(
  --model-path "${MODEL}"
  --tokenizer-path "${TOKENIZER}"
  --trust-remote-code
  --host "${HOST}"
  --port "${PORT}"
  --tp-size 1
  --dtype "${DTYPE}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --attention-backend flashinfer
  --dllm-algorithm "${DLLM_ALGORITHM}"
  --dllm-algorithm-config "${DLLM_CONFIG}"
)

if [[ -n "${CUDA_GRAPH_BS}" ]]; then
  read -r -a CUDA_GRAPH_BS_ARGS <<<"${CUDA_GRAPH_BS}"
  SERVER_ARGS+=(--cuda-graph-bs "${CUDA_GRAPH_BS_ARGS[@]}")
fi
if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
  SERVER_ARGS+=(--disable-cuda-graph)
fi
if [[ "${DISABLE_PIECEWISE_CUDA_GRAPH}" == "1" ]]; then
  SERVER_ARGS+=(--disable-piecewise-cuda-graph)
fi

{
  echo "SGLANG_REPO=${SGLANG_REPO}"
  echo "MODEL=${MODEL}"
  echo "TOKENIZER=${TOKENIZER}"
  echo "LORA_PATH=${LORA_PATH:-<none>}"
  echo "DLLM_ALGORITHM=${DLLM_ALGORITHM}"
  echo "OUTDIR=${OUTDIR}"
  echo "NUM_SAMPLES=${NUM_SAMPLES}"
  echo "MAX_TOKENS=${MAX_TOKENS}"
  echo "CONCURRENT=${CONCURRENT}"
  echo "HOST=${HOST}"
  echo "PORT=${PORT}"
  echo "MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC}"
  echo "MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}"
  echo "CUDA_GRAPH_BS=${CUDA_GRAPH_BS}"
  echo "DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH}"
  echo "DISABLE_PIECEWISE_CUDA_GRAPH=${DISABLE_PIECEWISE_CUDA_GRAPH}"
  echo "INCLUDE_STATS_FILE=${INCLUDE_STATS_FILE}"
  echo "DRY_RUN=${DRY_RUN}"
  echo "DLLM_CONFIG:"
  cat "${DLLM_CONFIG}"
} | tee "${CLIENT_LOG}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf "SERVER_COMMAND:"
  printf " %q" "${VENV}/bin/python" -m sglang.launch_server "${SERVER_ARGS[@]}"
  printf "\n"
  exit 0
fi

"${VENV}/bin/python" -m sglang.launch_server "${SERVER_ARGS[@]}" \
  >"${SERVER_LOG}" 2>&1 &

SERVER_PID=$!
cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 420); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "SERVER_EXITED" | tee -a "${CLIENT_LOG}"
    tail -n 200 "${SERVER_LOG}" | tee -a "${CLIENT_LOG}"
    exit 1
  fi
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "SERVER_READY after ${i}s" | tee -a "${CLIENT_LOG}"
    break
  fi
  if [[ "${i}" == "420" ]]; then
    echo "SERVER_TIMEOUT" | tee -a "${CLIENT_LOG}"
    tail -n 200 "${SERVER_LOG}" | tee -a "${CLIENT_LOG}"
    exit 1
  fi
  sleep 1
done

"${VENV}/bin/python" "${SGLANG_REPO}/benchmark/gsm8k/eval_sglang.py" \
  --benchmark gsm8k \
  --base_url "http://${HOST}:${PORT}/v1" \
  --no_thinking \
  --prompt_style v2 \
  --max_tokens "${MAX_TOKENS}" \
  --concurrent "${CONCURRENT}" \
  --num_samples "${NUM_SAMPLES}" \
  --output "${RESULTS_JSON}" \
  2>&1 | tee -a "${CLIENT_LOG}"

echo "Server log: ${SERVER_LOG}" | tee -a "${CLIENT_LOG}"
echo "Client log: ${CLIENT_LOG}" | tee -a "${CLIENT_LOG}"
echo "Results: ${RESULTS_JSON}" | tee -a "${CLIENT_LOG}"
