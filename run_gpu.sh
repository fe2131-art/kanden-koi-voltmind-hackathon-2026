#!/bin/bash
#SBATCH -J safety-view-agent-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o slurm-%j.out
#SBATCH --container-image=/home/team-005/nvidia+pytorch+25.11-py3.sqsh
#SBATCH --container-mounts=/home/team-005:/home/team-005

# デバッグ情報出力
echo "=========================================="
echo "JOB_ID=$SLURM_JOB_ID"
echo "=========================================="
hostname
nvidia-smi

# ワーキングディレクトリに移動（作業者名とブランチをカレントディレクトリから動的に取得）
WORK_USER=$(basename $(dirname $(pwd)))
CURRENT_BRANCH=$(git branch --show-current)
WORK_DIR="/home/team-005/work/${WORK_USER}/${CURRENT_BRANCH}"

echo "Working directory: $WORK_DIR"
cd "$WORK_DIR" || exit 1

# 環境変数を読み込み
set -a
source .env
set +a

# Python スクリプット実行
uv run python src/run.py > "agent_output_${SLURM_JOB_ID}.log" 2>&1
