#!/bin/bash
#SBATCH -J gpu-2-check
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -t 00:05:00
#SBATCH -o slurm-%j.out

set -euo pipefail

echo "=========================================="
echo "JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "=========================================="

echo
echo "---- CUDA_VISIBLE_DEVICES ----"
echo "${CUDA_VISIBLE_DEVICES:-<not set>}"

echo
echo "---- Host RAM (free -h) ----"
free -h || true

echo
echo "---- Torch GPU Test + Memory Stats ----"
python -u - <<'EOF'
import os
import time
import resource
import torch

def mb(x): return x / 1024**2

print("[info] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[info] torch version =", torch.__version__)
print("[info] cuda available =", torch.cuda.is_available())
print("[info] torch sees device_count =", torch.cuda.device_count(), flush=True)

# Host RAM (process RSS)
ru = resource.getrusage(resource.RUSAGE_SELF)
print(f"[host] max RSS = {ru.ru_maxrss/1024:.1f} MB (Linux ru_maxrss is KB)", flush=True)

n = torch.cuda.device_count()
if n == 0:
    raise SystemExit("No CUDA devices visible to torch.")

# まず “見えているGPU” の空き/総量を表示
for i in range(n):
    free_b, total_b = torch.cuda.mem_get_info(i)
    props = torch.cuda.get_device_properties(i)
    print(f"[gpu {i}] {props.name}  free={mb(free_b):.0f}MB / total={mb(total_b):.0f}MB", flush=True)

# 計算を軽めにして確実に進むように（必要なら大きくしてOK）
M = 2048
print(f"\n[test] allocating & matmul with M={M}", flush=True)

for i in range(n):
    print(f"\n--- Testing GPU {i} ---", flush=True)
    torch.cuda.set_device(i)

    # peak stats reset
    try:
        torch.cuda.reset_peak_memory_stats(i)
    except Exception:
        pass

    # before
    free0, total0 = torch.cuda.mem_get_info(i)
    print(f"  before: free={mb(free0):.0f}MB total={mb(total0):.0f}MB", flush=True)

    t0 = time.time()
    x = torch.randn(M, M, device=f"cuda:{i}", dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize(i)
    dt = time.time() - t0

    alloc = torch.cuda.memory_allocated(i)
    rsvd  = torch.cuda.memory_reserved(i)
    peakA = torch.cuda.max_memory_allocated(i)
    peakR = torch.cuda.max_memory_reserved(i)

    # after
    free1, total1 = torch.cuda.mem_get_info(i)
    print(f"  matmul time: {dt:.3f}s", flush=True)
    print(f"  allocated : {mb(alloc):.1f} MB", flush=True)
    print(f"  reserved  : {mb(rsvd):.1f} MB", flush=True)
    print(f"  peak alloc: {mb(peakA):.1f} MB", flush=True)
    print(f"  peak rsvd : {mb(peakR):.1f} MB", flush=True)
    print(f"  after: free={mb(free1):.0f}MB total={mb(total1):.0f}MB", flush=True)

print("\nAll visible GPUs tested successfully.", flush=True)
EOF