#!/bin/bash
#SBATCH -J vllm-h100-mig-402020-direct
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:h100_3g.40gb:1,gpu:h100_1g.20gb:2
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

# ===== モデル設定 =====
MODEL_40G="${MODEL_40G:-Qwen/Qwen3-VL-4B-Instruct}"
MODEL_20G_A="${MODEL_20G_A:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_20G_B="${MODEL_20G_B:-Qwen/Qwen2.5-3B-Instruct}"

PORT_40G=8000
PORT_20G_A=8001
PORT_20G_B=8002

GPU_UTIL_40G="${GPU_UTIL_40G:-0.90}"
GPU_UTIL_20G_A="${GPU_UTIL_20G_A:-0.90}"
GPU_UTIL_20G_B="${GPU_UTIL_20G_B:-0.90}"

MAX_LEN_40G="${MAX_LEN_40G:-8192}"
MAX_LEN_20G_A="${MAX_LEN_20G_A:-4096}"
MAX_LEN_20G_B="${MAX_LEN_20G_B:-4096}"

# ===== Slurm が batch job に渡した GPU/MIG を 3 個に分解 =====
ALLOCATED_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [ -z "$ALLOCATED_CUDA_DEVICES" ]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES is empty."
  exit 1
fi

IFS=',' read -r DEV_40G DEV_20G_A DEV_20G_B EXTRA <<< "$ALLOCATED_CUDA_DEVICES"

if [ -n "${EXTRA:-}" ]; then
  echo "WARN: more than 3 devices are visible: $ALLOCATED_CUDA_DEVICES"
fi

if [ -z "${DEV_40G:-}" ] || [ -z "${DEV_20G_A:-}" ] || [ -z "${DEV_20G_B:-}" ]; then
  echo "ERROR: expected 3 allocated devices, got: $ALLOCATED_CUDA_DEVICES"
  exit 1
fi

echo
echo "---- Visible MIG assignments ----"
echo "40G   -> $DEV_40G"
echo "20G-A -> $DEV_20G_A"
echo "20G-B -> $DEV_20G_B"

echo
echo "---- nvidia-smi -L ----"
nvidia-smi -L || true

cleanup() {
  echo
  echo "---- Cleanup ----"
  for pid in ${PID_40G:-} ${PID_20G_A:-} ${PID_20G_B:-}; do
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
    fi
  done
}
trap cleanup EXIT INT TERM

echo
echo "---- Starting vLLM servers ----"

# Server 1 (40G MIG)
env CUDA_VISIBLE_DEVICES="$DEV_40G" \
  uv run vllm serve "$MODEL_40G" \
    --host 0.0.0.0 \
    --port "$PORT_40G" \
    --served-model-name model-40g \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_LEN_40G" \
    --gpu-memory-utilization "$GPU_UTIL_40G" \
    --enable-prefix-caching \
    > "logs/vllm_${SLURM_JOB_ID}_40g.log" 2>&1 &
PID_40G=$!

# Server 2 (20G MIG)
env CUDA_VISIBLE_DEVICES="$DEV_20G_A" \
  uv run vllm serve "$MODEL_20G_A" \
    --host 0.0.0.0 \
    --port "$PORT_20G_A" \
    --served-model-name model-20g-a \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_LEN_20G_A" \
    --gpu-memory-utilization "$GPU_UTIL_20G_A" \
    --enable-prefix-caching \
    > "logs/vllm_${SLURM_JOB_ID}_20g_a.log" 2>&1 &
PID_20G_A=$!

# Server 3 (20G MIG)
env CUDA_VISIBLE_DEVICES="$DEV_20G_B" \
  uv run vllm serve "$MODEL_20G_B" \
    --host 0.0.0.0 \
    --port "$PORT_20G_B" \
    --served-model-name model-20g-b \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_LEN_20G_B" \
    --gpu-memory-utilization "$GPU_UTIL_20G_B" \
    --enable-prefix-caching \
    > "logs/vllm_${SLURM_JOB_ID}_20g_b.log" 2>&1 &
PID_20G_B=$!

echo "PIDs: 40G=$PID_40G, 20G-A=$PID_20G_A, 20G-B=$PID_20G_B"

echo
echo "---- Waiting for all servers to be ready ----"
READY_40=0
READY_20A=0
READY_20B=0

for i in $(seq 1 180); do
  if [ "$READY_40" -eq 0 ] && curl -fsS "http://127.0.0.1:${PORT_40G}/v1/models" >/dev/null 2>&1; then
    READY_40=1
    echo "40G server is up:   http://127.0.0.1:${PORT_40G}"
  fi

  if [ "$READY_20A" -eq 0 ] && curl -fsS "http://127.0.0.1:${PORT_20G_A}/v1/models" >/dev/null 2>&1; then
    READY_20A=1
    echo "20G-A server is up: http://127.0.0.1:${PORT_20G_A}"
  fi

  if [ "$READY_20B" -eq 0 ] && curl -fsS "http://127.0.0.1:${PORT_20G_B}/v1/models" >/dev/null 2>&1; then
    READY_20B=1
    echo "20G-B server is up: http://127.0.0.1:${PORT_20G_B}"
  fi

  if [ "$READY_40" -eq 1 ] && [ "$READY_20A" -eq 1 ] && [ "$READY_20B" -eq 1 ]; then
    break
  fi

  sleep 1
done

echo
echo "---- /v1/models (40G) ----"
curl -s "http://127.0.0.1:${PORT_40G}/v1/models" || true

echo
echo "---- /v1/models (20G-A) ----"
curl -s "http://127.0.0.1:${PORT_20G_A}/v1/models" || true

echo
echo "---- /v1/models (20G-B) ----"
curl -s "http://127.0.0.1:${PORT_20G_B}/v1/models" || true

echo
echo "---- Tail logs ----"
for f in \
  "logs/vllm_${SLURM_JOB_ID}_40g.log" \
  "logs/vllm_${SLURM_JOB_ID}_20g_a.log" \
  "logs/vllm_${SLURM_JOB_ID}_20g_b.log"
do
  echo "===== $f ====="
  tail -n 40 "$f" || true
done

echo
echo "---- Keep running (stop with: scancel ${SLURM_JOB_ID}) ----"
wait
```

コア部分だけ抜くと、やりたい形はこれです。

```bash
IFS=',' read -r DEV_40G DEV_20G_A DEV_20G_B <<< "$CUDA_VISIBLE_DEVICES"

CUDA_VISIBLE_DEVICES="$DEV_40G" \
  vllm serve "$MODEL_40G" --port 8000 --gpu-memory-utilization 0.90 &

CUDA_VISIBLE_DEVICES="$DEV_20G_A" \
  vllm serve "$MODEL_20G_A" --port 8001 --gpu-memory-utilization 0.90 &

CUDA_VISIBLE_DEVICES="$DEV_20G_B" \
  vllm serve "$MODEL_20G_B" --port 8002 --gpu-memory-utilization 0.90 &
