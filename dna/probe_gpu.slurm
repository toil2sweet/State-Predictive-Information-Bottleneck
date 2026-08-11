#!/usr/bin/env bash
#SBATCH --job-name=spib-gpu-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00

set -euo pipefail

PROJECT_ROOT="${DNA_PROJECT_ROOT:-/mnt/rna01/lidp/spib-project}"
ENV_PREFIX="${DNA_ENV_PREFIX:-${PROJECT_ROOT}/envs/spib-cu118}"

echo "===== DNA GPU probe ====="
date
echo "host=$(hostname)"
echo "partition=${SLURM_JOB_PARTITION:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
    "${ENV_PREFIX}/bin/python" - <<'PY'
import sys
import torch

print("python=", sys.version.replace("\n", " "))
print("torch=", torch.__version__)
print("torch_cuda_build=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated GPU")
print("gpu=", torch.cuda.get_device_name(0))
x = torch.ones(32, device="cuda")
print("cuda_smoke_sum=", x.sum().item())
PY
else
    echo "Environment is not installed yet: ${ENV_PREFIX}"
fi
