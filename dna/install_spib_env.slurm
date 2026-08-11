#!/usr/bin/env bash
#SBATCH --job-name=spib-env
#SBATCH --partition=Normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT="${DNA_PROJECT_ROOT:-/mnt/rna01/lidp/spib-project}"
ENV_PREFIX="${DNA_ENV_PREFIX:-${PROJECT_ROOT}/envs/spib-cu118}"
PYTHON_BIN="${DNA_PYTHON_BIN:-/mnt/apps/python-3.11.0/bin/python3.11}"
COMPLETE_MARKER="${ENV_PREFIX}/.spib-environment-complete"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing DNA Python: ${PYTHON_BIN}" >&2; exit 1; }
if [[ -f "${COMPLETE_MARKER}" ]]; then
    echo "SPIB environment is already complete: ${ENV_PREFIX}"
    exit 0
fi
[[ ! -e "${ENV_PREFIX}" ]] || {
    echo "An incomplete environment already exists: ${ENV_PREFIX}" >&2
    echo "Inspect and move it aside explicitly before rebuilding it." >&2
    exit 1
}

mkdir -p "${PROJECT_ROOT}/envs" "${PROJECT_ROOT}/logs/env"
exec > >(tee -a "${PROJECT_ROOT}/logs/env/spib-env-${SLURM_JOB_ID:-manual}.log") 2>&1

echo "===== DNA SPIB environment installation ====="
date
echo "host=$(hostname)"
echo "python=${PYTHON_BIN}"
echo "environment=${ENV_PREFIX}"

"${PYTHON_BIN}" -m venv "${ENV_PREFIX}"
"${ENV_PREFIX}/bin/python" -m pip install --upgrade "pip==25.1.1"
"${ENV_PREFIX}/bin/python" -m pip install \
    "numpy==1.26.4" \
    "matplotlib==3.8.4"
"${ENV_PREFIX}/bin/python" -m pip install \
    "torch==2.7.1" \
    --index-url https://download.pytorch.org/whl/cu118

"${ENV_PREFIX}/bin/python" - <<'PY'
import matplotlib
import numpy
import torch

print("numpy=", numpy.__version__)
print("matplotlib=", matplotlib.__version__)
print("torch=", torch.__version__)
print("torch_cuda_build=", torch.version.cuda)
PY

"${ENV_PREFIX}/bin/python" -m pip freeze > "${PROJECT_ROOT}/logs/env/spib-cu118-pip-freeze.txt"
touch "${COMPLETE_MARKER}"
echo "Installed ${ENV_PREFIX}"
