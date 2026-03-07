#!/bin/bash
#SBATCH -J vllm-qwen3vl-light
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
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

# worktree運用ならここにあるはず、なければ通常運用としてREPO_ROOTを使う
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

# キャッシュ（作業者単位で永続化）
export UV_CACHE_DIR="$HOME/.cache/uv"
export HF_HOME="$HOME/data/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

# 依存を揃える
uv sync --frozen || uv sync

MODEL="Qwen/Qwen3-VL-4B-Instruct"
PORT=8001

echo
echo "---- Starting vLLM server ----"
uv run vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  > "vllm_${SLURM_JOB_ID}.log" 2>&1 &
# --reasoning-parser qwen3 は Qwen3 thinking モデル用。必要なら上記に追加。

VLLM_PID=$!
echo "vLLM PID=$VLLM_PID"

echo
echo "---- Waiting for server to be ready ----"
for i in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "Server is up! (http://127.0.0.1:${PORT})"
    break
  fi
  sleep 1
done

echo
echo "---- /v1/models ----"
curl -s "http://127.0.0.1:${PORT}/v1/models" || true

echo
echo "---- Tail vLLM log ----"
tail -n 60 "vllm_${SLURM_JOB_ID}.log" || true

echo
echo "---- Keep running (stop with: scancel ${SLURM_JOB_ID}) ----"
wait "$VLLM_PID"