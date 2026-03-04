#!/bin/bash
#SBATCH -J pyver-check
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o slurm-%j.out

set -euo pipefail

JOB_ID="${SLURM_JOB_ID:-N/A}"

echo "=========================================="
echo "JOB_ID=$JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "=========================================="

echo
echo "---- python -V ----"
python -V || true

echo
echo "---- which python ----"
which python || true

echo
echo "---- uv --version / which uv ----"
uv --version || true
which uv || true

echo
echo "---- uv run python -V ----"
uv run python -V

echo
echo "---- uv run which python ----"
uv run which python

echo
echo "---- uv run: sys.executable ----"
uv run python -c "import sys; print(sys.executable)"

echo
echo "---- (optional) import check: torch / vllm ----"
uv run python -c "import torch; print('torch', torch.__version__)" || true
uv run python -c "import vllm; print('vllm', vllm.__version__)" || true