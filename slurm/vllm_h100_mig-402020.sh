#!/bin/bash
#SBATCH -J vllm-h100-mig-402020
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

# GPUの見え方を安定化
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# キャッシュ（作業者単位で永続化）
export UV_CACHE_DIR="$HOME/.cache/uv"
export HF_HOME="$HOME/data/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" logs

# 依存を揃える
uv sync --frozen || uv sync

# ===== モデル・ポート設定 =====
# 40GB MIG に載せるモデル
MODEL_40G="${MODEL_40G:-Qwen/Qwen3-VL-4B-Instruct}"
# 20GB MIG に載せるモデル（必要に応じて小さめ/量子化モデルへ変更）
MODEL_20G_A="${MODEL_20G_A:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_20G_B="${MODEL_20G_B:-Qwen/Qwen2.5-3B-Instruct}"

PORT_40G=8000
PORT_20G_A=8001
PORT_20G_B=8002

# MIGごとに少し余裕を残して調整
GPU_UTIL_40G="${GPU_UTIL_40G:-0.90}"
GPU_UTIL_20G_A="${GPU_UTIL_20G_A:-0.90}"
GPU_UTIL_20G_B="${GPU_UTIL_20G_B:-0.90}"

MAX_LEN_40G="${MAX_LEN_40G:-8192}"
MAX_LEN_20G_A="${MAX_LEN_20G_A:-4096}"
MAX_LEN_20G_B="${MAX_LEN_20G_B:-4096}"

cleanup() {
  echo
  echo "---- Cleanup ----"
  jobs -pr | xargs -r kill || true
}
trap cleanup EXIT INT TERM

echo
echo "---- Starting vLLM servers on MIGs ----"

# 40GB MIG
srun --exclusive -N1 -n1 -c8 --gres=gpu:h100_3g.40gb:1 \
  bash -lc "
    set -euo pipefail
    cd '$WORK_DIR'
    export UV_CACHE_DIR='$UV_CACHE_DIR'
    export HF_HOME='$HF_HOME'
    export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    echo '[40G] HOSTNAME='\"\$(hostname)\"
    echo '[40G] CUDA_VISIBLE_DEVICES='\"\${CUDA_VISIBLE_DEVICES:-<not set>}\"
    exec uv run vllm serve '$MODEL_40G' \
      --host 0.0.0.0 \
      --port '$PORT_40G' \
      --served-model-name model-40g \
      --tensor-parallel-size 1 \
      --max-model-len '$MAX_LEN_40G' \
      --gpu-memory-utilization '$GPU_UTIL_40G' \
      --enable-prefix-caching
  " > "logs/vllm_${SLURM_JOB_ID}_40g.log" 2>&1 &

PID_40G=$!

# 20GB MIG #1
srun --exclusive -N1 -n1 -c8 --gres=gpu:h100_1g.20gb:1 \
  bash -lc "
    set -euo pipefail
    cd '$WORK_DIR'
    export UV_CACHE_DIR='$UV_CACHE_DIR'
    export HF_HOME='$HF_HOME'
    export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    echo '[20G-A] HOSTNAME='\"\$(hostname)\"
    echo '[20G-A] CUDA_VISIBLE_DEVICES='\"\${CUDA_VISIBLE_DEVICES:-<not set>}\"
    exec uv run vllm serve '$MODEL_20G_A' \
      --host 0.0.0.0 \
      --port '$PORT_20G_A' \
      --served-model-name model-20g-a \
      --tensor-parallel-size 1 \
      --max-model-len '$MAX_LEN_20G_A' \
      --gpu-memory-utilization '$GPU_UTIL_20G_A' \
      --enable-prefix-caching
  " > "logs/vllm_${SLURM_JOB_ID}_20g_a.log" 2>&1 &

PID_20G_A=$!

# 20GB MIG #2
srun --exclusive -N1 -n1 -c8 --gres=gpu:h100_1g.20gb:1 \
  bash -lc "
    set -euo pipefail
    cd '$WORK_DIR'
    export UV_CACHE_DIR='$UV_CACHE_DIR'
    export HF_HOME='$HF_HOME'
    export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    echo '[20G-B] HOSTNAME='\"\$(hostname)\"
    echo '[20G-B] CUDA_VISIBLE_DEVICES='\"\${CUDA_VISIBLE_DEVICES:-<not set>}\"
    exec uv run vllm serve '$MODEL_20G_B' \
      --host 0.0.0.0 \
      --port '$PORT_20G_B' \
      --served-model-name model-20g-b \
      --tensor-parallel-size 1 \
      --max-model-len '$MAX_LEN_20G_B' \
      --gpu-memory-utilization '$GPU_UTIL_20G_B' \
      --enable-prefix-caching
  " > "logs/vllm_${SLURM_JOB_ID}_20g_b.log" 2>&1 &

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