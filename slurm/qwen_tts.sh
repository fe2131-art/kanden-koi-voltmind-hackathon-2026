#!/bin/bash
#SBATCH -J qwen-tts
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
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
nvidia-smi

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

# 空きメモリが最多の GPU を選択（SLURM 割当が占有済みの場合でも正しい GPU を使う）
BEST_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
  | awk 'BEGIN{max=-1; idx=0} {if($1>max){max=$1; idx=NR-1}} END{print idx}')
export CUDA_VISIBLE_DEVICES="$BEST_GPU"
echo "Selected GPU: $BEST_GPU (most free memory)"

# TTS モデルを事前ダウンロード（未キャッシュの場合のみ）
TTS_MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
echo
echo "---- Downloading TTS model (if not cached) ----"
uv run python - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download, constants

model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
cache_dir = os.path.join(os.environ.get("HF_HOME", constants.HF_HUB_CACHE), "hub")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

try:
    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        token=token,
    )
    print(f"モデルキャッシュ確認完了: {local_path}", flush=True)
except Exception as e:
    print(f"ERROR: モデルのダウンロードに失敗しました: {e}", file=sys.stderr, flush=True)
    sys.exit(1)
PYEOF

echo
echo "---- Starting Qwen TTS synthesis ----"
uv run python src/tts/synthesize.py \
  --input  data/perception_results.json \
  --outdir data/voice \
  --config configs/default.yaml
