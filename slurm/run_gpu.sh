#!/bin/bash
#SBATCH -J safety-view-agent-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o slurm-%j.out
#SBATCH --container-image=/home/team-005/nvidia+pytorch+25.11-py3.sqsh
#SBATCH --container-mounts=/home/team-005:/home/team-005

# フェイルセーフ（エラー発生時に即座に処理を停止）
set -euo pipefail

# パス設定とワークツリー判定
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(cd "$SUBMIT_DIR" && git rev-parse --show-toplevel)"
WORK_USER="${USER:-$(whoami)}"

CURRENT_BRANCH="$(cd "$SUBMIT_DIR" && git branch --show-current)"
if [ -z "$CURRENT_BRANCH" ]; then
  CURRENT_BRANCH="detached-$(cd "$SUBMIT_DIR" && git rev-parse --short HEAD)"
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
nvidia-smi

cd "$WORK_DIR"

# キャッシュの永続化とマウント先への指定
export UV_CACHE_DIR="/home/team-005/work/${WORK_USER}/.cache/uv"
export HF_HOME="/home/team-005/work/${WORK_USER}/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

# 環境変数の読み込み
set -a
[ -f .env ] && source .env
set +a

# 依存関係の確実な同期
echo "---- Syncing dependencies ----"
uv sync --frozen || uv sync

# スクリプト実行
echo "---- Starting Agent ----"
uv run python src/run.py