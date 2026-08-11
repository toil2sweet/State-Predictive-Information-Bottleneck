#!/usr/bin/env bash
#SBATCH --job-name=spib-env
#SBATCH --partition=Normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

CODE_ROOT="${DNA_CODE_ROOT:-/mnt/rna01/lidp/State-Predictive-Information-Bottleneck}"
PROJECT_ROOT="${DNA_PROJECT_ROOT:-/mnt/rna01/lidp/spib-project}"
MINIFORGE_ROOT="${DNA_MINIFORGE_ROOT:-/mnt/rna01/lidp/miniforge3}"
ENV_PREFIX="${DNA_ENV_PREFIX:-${MINIFORGE_ROOT}/envs/spib}"
ENV_SPEC="${DNA_SPIB_ENV_SPEC:-${CODE_ROOT}/dna/spib-environment.yml}"
MAMBA_BIN="${MINIFORGE_ROOT}/bin/mamba"
TORCH_VERSION="${DNA_TORCH_VERSION:-2.7.1}"
TORCH_INDEX_URL="${DNA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
COMPLETE_MARKER="${ENV_PREFIX}/.spib-environment-complete"

[[ -x "${MAMBA_BIN}" ]] || { echo "Missing DNA Miniforge/Mamba: ${MAMBA_BIN}" >&2; exit 1; }
[[ -f "${ENV_SPEC}" ]] || {
    echo "Missing environment specification: ${ENV_SPEC}" >&2
    exit 1
}
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
echo "miniforge=${MINIFORGE_ROOT}"
echo "environment=${ENV_PREFIX}"
echo "environment_spec=${ENV_SPEC}"
echo "torch=${TORCH_VERSION}"
echo "torch_index=${TORCH_INDEX_URL}"

"${MAMBA_BIN}" env create --yes \
    --prefix "${ENV_PREFIX}" \
    --file "${ENV_SPEC}"
"${ENV_PREFIX}/bin/python" -m pip install \
    "torch==${TORCH_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

"${ENV_PREFIX}/bin/python" - <<'PY'
import matplotlib
import numpy
import torch

print("numpy=", numpy.__version__)
print("matplotlib=", matplotlib.__version__)
print("torch=", torch.__version__)
print("torch_cuda_build=", torch.version.cuda)
PY

"${ENV_PREFIX}/bin/python" -m pip freeze > "${PROJECT_ROOT}/logs/env/spib-pip-freeze.txt"
"${MINIFORGE_ROOT}/bin/conda" list --explicit --prefix "${ENV_PREFIX}" \
    > "${PROJECT_ROOT}/logs/env/spib-conda-explicit.txt"
touch "${COMPLETE_MARKER}"
echo "Installed ${ENV_PREFIX}"
