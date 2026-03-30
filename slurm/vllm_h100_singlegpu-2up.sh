#!/bin/bash
#SBATCH -J vllm-h100-singlegpu-2up
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(cd "$SUBMIT_DIR" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

WORK_USER="${USER:-$(whoami)}"

CURRENT_BRANCH="$(git branch --show-current)"
if [ -z "$CURRENT_BRANCH" ]; then
  CURRENT_BRANCH="detached-$(git rev-parse --short HEAD)"
fi

CANDIDATE_WORK_DIR="/home/team-005/work/${WORK_USER}/${CURRENT_BRANCH}"

if [ -d "$CANDIDATE_WORK_DIR" ]; then
  WORK_DIR="$CANDIDATE_WORK_DIR"
  MODE="worktree"
else
  WORK_DIR="$REPO_ROOT"
  MODE="repo-root"
fi

echo "=========================================="
echo "JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "HOSTNAME=$(hostname)"
echo "SUBMIT_DIR=$SUBMIT_DIR"
echo "REPO_ROOT=$REPO_ROOT"
echo "WORK_USER=$WORK_USER"
echo "CURRENT_BRANCH=$CURRENT_BRANCH"
echo "MODE=$MODE"
echo "WORK_DIR=$WORK_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "=========================================="

cd "$WORK_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# キャッシュ
export UV_CACHE_DIR="$HOME/.cache/uv"
export HF_HOME="$HOME/data/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" logs

# 依存
uv sync --frozen || uv sync

# モデル設定
MODEL_A="${MODEL_A:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_B="${MODEL_B:-meta-llama/Llama-3.1-8B-Instruct}"

PORT_A=8000
PORT_B=8001

GPU_UTIL_A="${GPU_UTIL_A:-0.45}"
GPU_UTIL_B="${GPU_UTIL_B:-0.45}"

MAX_LEN_A="${MAX_LEN_A:-4096}"
MAX_LEN_B="${MAX_LEN_B:-4096}"

# Slurm がジョブに見せている GPU をそのまま使う
VISIBLE_GPU="${CUDA_VISIBLE_DEVICES:-0}"

cleanup() {
  echo
  echo "---- Cleanup ----"
  for pid in ${PID_A:-} ${PID_B:-}; do
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
    fi
  done
}
trap cleanup EXIT INT TERM

echo
echo "---- Starting vLLM servers on the same GPU ----"
echo "VISIBLE_GPU=$VISIBLE_GPU"

env CUDA_VISIBLE_DEVICES="$VISIBLE_GPU" \
  uv run vllm serve "$MODEL_A" \
    --host 0.0.0.0 \
    --port "$PORT_A" \
    --served-model-name model-a \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_LEN_A" \
    --gpu-memory-utilization "$GPU_UTIL_A" \
    --enable-prefix-caching \
    > "logs/vllm_${SLURM_JOB_ID}_a.log" 2>&1 &
PID_A=$!

env CUDA_VISIBLE_DEVICES="$VISIBLE_GPU" \
  uv run vllm serve "$MODEL_B" \
    --host 0.0.0.0 \
    --port "$PORT_B" \
    --served-model-name model-b \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_LEN_B" \
    --gpu-memory-utilization "$GPU_UTIL_B" \
    --enable-prefix-caching \
    > "logs/vllm_${SLURM_JOB_ID}_b.log" 2>&1 &
PID_B=$!

echo "PIDs: A=$PID_A, B=$PID_B"

echo
echo "---- Waiting for servers ----"
READY_A=0
READY_B=0

for i in $(seq 1 180); do
  if [ "$READY_A" -eq 0 ] && curl -fsS "http://127.0.0.1:${PORT_A}/v1/models" >/dev/null 2>&1; then
    READY_A=1
    echo "Server A is up: http://127.0.0.1:${PORT_A}"
  fi

  if [ "$READY_B" -eq 0 ] && curl -fsS "http://127.0.0.1:${PORT_B}/v1/models" >/dev/null 2>&1; then
    READY_B=1
    echo "Server B is up: http://127.0.0.1:${PORT_B}"
  fi

  if [ "$READY_A" -eq 1 ] && [ "$READY_B" -eq 1 ]; then
    break
  fi

  sleep 1
done

echo
echo "---- /v1/models (A) ----"
curl -s "http://127.0.0.1:${PORT_A}/v1/models" || true

echo
echo "---- /v1/models (B) ----"
curl -s "http://127.0.0.1:${PORT_B}/v1/models" || true

echo
echo "---- Tail logs ----"
tail -n 40 "logs/vllm_${SLURM_JOB_ID}_a.log" || true
tail -n 40 "logs/vllm_${SLURM_JOB_ID}_b.log" || true

echo
echo "---- Keep running (stop with: scancel ${SLURM_JOB_ID}) ----"
wait