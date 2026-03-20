#!/bin/bash
#SBATCH -J qwen-tts-server
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
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

# 環境変数の読み込み
set -a
[ -f .env ] && source .env
set +a

# 依存を揃える
uv sync --frozen || uv sync

# 空きメモリが最多の GPU を選択
BEST_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
  | awk 'BEGIN{max=-1; idx=0} {if($1>max){max=$1; idx=NR-1}} END{print idx}')
export CUDA_VISIBLE_DEVICES="$BEST_GPU"
echo "Selected GPU: $BEST_GPU (most free memory)"

MODEL="${TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
PORT=8010

echo
echo "---- Starting Qwen TTS server ----"
vllm-omni serve "$MODEL" \
  --omni \
  --host 0.0.0.0 \
  --port "$PORT" \
  --task-type CustomVoice \
  > "qwen_tts_server_${SLURM_JOB_ID:-local}.log" 2>&1 &

SERVER_PID=$!
echo "TTS server PID=$SERVER_PID"

echo
echo "---- Waiting for server to be ready ----"
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "Server is up! (http://127.0.0.1:${PORT})"
    break
  fi
  sleep 2
done

echo
echo "---- /health ----"
curl -s "http://127.0.0.1:${PORT}/health" || true

echo
echo "---- Tail server log ----"
tail -n 40 "qwen_tts_server_${SLURM_JOB_ID:-local}.log" || true

echo
echo "---- Keep running (stop with: scancel ${SLURM_JOB_ID:-N/A}) ----"
wait "$SERVER_PID"
