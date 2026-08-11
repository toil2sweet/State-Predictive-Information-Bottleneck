#!/usr/bin/env bash
#SBATCH --job-name=spib
#SBATCH --partition=GPUA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

CODE_ROOT="${DNA_CODE_ROOT:-/mnt/rna01/lidp/State-Predictive-Information-Bottleneck}"
PROJECT_ROOT="${DNA_PROJECT_ROOT:-/mnt/rna01/lidp/spib-project}"
DATA_ROOT="${DNA_DATA_ROOT:-${PROJECT_ROOT}/data}"
ENV_PREFIX="${DNA_ENV_PREFIX:-${PROJECT_ROOT}/envs/spib-cu118}"
CONFIG="${CONFIG:-examples/Four_Well_hsic_config.ini}"
COMMIT="${SPIB_GIT_COMMIT:?SPIB_GIT_COMMIT must be a full Git commit}"
RUN_ID="${SLURM_JOB_ID:?This script must run through sbatch}"
SHORT_COMMIT="${COMMIT:0:12}"
SCRATCH_ROOT="/tmp/${USER}/spib/${RUN_ID}"
CODE_SNAPSHOT="${SCRATCH_ROOT}/code"
SCRATCH_OUTPUT="${SCRATCH_ROOT}/output"
STAGED_CONFIG="${SCRATCH_ROOT}/config/run.ini"
OUTPUT_ROOT="${PROJECT_ROOT}/results/spib/${RUN_ID}-${SHORT_COMMIT}"
LOG_ROOT="${PROJECT_ROOT}/logs/spib"
LOG_FILE="${LOG_ROOT}/${RUN_ID}.log"

[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid full Git commit: ${COMMIT}" >&2; exit 1; }
[[ "${CONFIG}" != /* ]] || { echo "CONFIG must be tracked and relative to the repository" >&2; exit 1; }
[[ -d "${CODE_ROOT}/.git" ]] || { echo "Missing DNA checkout: ${CODE_ROOT}" >&2; exit 1; }
[[ -x "${ENV_PREFIX}/bin/python" ]] || { echo "Missing SPIB environment: ${ENV_PREFIX}" >&2; exit 1; }
[[ -f "${ENV_PREFIX}/.spib-environment-complete" ]] || {
    echo "SPIB environment installation is incomplete: ${ENV_PREFIX}" >&2
    exit 1
}
git -C "${CODE_ROOT}" cat-file -e "${COMMIT}^{commit}"
git -C "${CODE_ROOT}" cat-file -e "${COMMIT}:${CONFIG}"

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}" "${CODE_SNAPSHOT}" "${SCRATCH_OUTPUT}"
exec >>"${LOG_FILE}" 2>&1

sync_outputs() {
    if [[ -d "${SCRATCH_OUTPUT}" ]]; then
        rsync -a "${SCRATCH_OUTPUT}/" "${OUTPUT_ROOT}/"
    fi
    if [[ -f "${STAGED_CONFIG}" ]]; then
        cp -f "${STAGED_CONFIG}" "${OUTPUT_ROOT}/run.ini"
    fi
}

finish() {
    status=$?
    trap - EXIT
    set +e
    sync_outputs
    echo "exit_status=${status}"
    date
    if [[ "${status}" -eq 0 ]]; then
        case "${SCRATCH_ROOT}" in
            /tmp/"${USER}"/spib/"${RUN_ID}") rm -rf -- "${SCRATCH_ROOT}" ;;
            *) echo "Refusing to remove unexpected scratch path: ${SCRATCH_ROOT}" >&2 ;;
        esac
    else
        echo "Preserving failed-job scratch when the node permits it: ${SCRATCH_ROOT}" >&2
    fi
    exit "${status}"
}
trap finish EXIT

echo "===== DNA SPIB run ====="
date
echo "host=$(hostname)"
echo "job=${RUN_ID}"
echo "partition=${SLURM_JOB_PARTITION:-unknown}"
echo "commit=${COMMIT}"
echo "config=${CONFIG}"
echo "environment=${ENV_PREFIX}"
echo "data_root=${DATA_ROOT}"
echo "scratch=${SCRATCH_ROOT}"
echo "output=${OUTPUT_ROOT}"

git -C "${CODE_ROOT}" archive "${COMMIT}" | tar -xf - -C "${CODE_SNAPSHOT}"
"${ENV_PREFIX}/bin/python" "${CODE_SNAPSHOT}/dna/stage_inputs.py" \
    --config "${CODE_SNAPSHOT}/${CONFIG}" \
    --code-root "${CODE_SNAPSHOT}" \
    --data-root "${DATA_ROOT}" \
    --scratch-data "${SCRATCH_ROOT}/data" \
    --output-config "${STAGED_CONFIG}"

export PYTHONPATH="${CODE_SNAPSHOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SPIB_OUTPUT_DIR="${SCRATCH_OUTPUT}/SPIB"
export SPIB_FIG_DIR="${SCRATCH_OUTPUT}/fig"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib"

{
    echo "commit=${COMMIT}"
    echo "config=${CONFIG}"
    echo "environment=${ENV_PREFIX}"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "node=${SLURM_JOB_NODELIST:-unknown}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    "${ENV_PREFIX}/bin/python" -m pip freeze
} > "${SCRATCH_OUTPUT}/run-metadata.txt"

nvidia-smi
"${ENV_PREFIX}/bin/python" - <<'PY'
import sys
import torch

print("python=", sys.version.replace("\n", " "))
print("torch=", torch.__version__)
print("torch_cuda_build=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the Slurm-allocated GPU")
print("gpu=", torch.cuda.get_device_name(0))
PY

cd "${CODE_SNAPSHOT}"
"${ENV_PREFIX}/bin/python" -u test_model_advanced.py -config "${STAGED_CONFIG}"
sync_outputs
