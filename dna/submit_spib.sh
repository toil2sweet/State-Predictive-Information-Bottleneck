#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 [GPUA100|GPUA40] [tracked-config.ini]" >&2
}

PARTITION="${1:-GPUA100}"
CONFIG="${2:-examples/Four_Well_hsic_config.ini}"
PROJECT_ROOT="${DNA_PROJECT_ROOT:-/mnt/rna01/lidp/spib-project}"
ENV_PREFIX="${DNA_ENV_PREFIX:-/mnt/rna01/lidp/miniforge3/envs/spib}"
CODE_ROOT="$(git rev-parse --show-toplevel)"

if [[ $# -gt 2 ]]; then
    usage
    exit 2
fi
case "${PARTITION}" in
    GPUA100|GPUA40) ;;
    *) echo "Unsupported partition: ${PARTITION}" >&2; usage; exit 2 ;;
esac
[[ "${CONFIG}" != /* && "${CONFIG}" != *,* && "${CONFIG}" != *$'\n'* ]] || {
    echo "CONFIG must be a safe repository-relative path" >&2
    exit 2
}
[[ -x "${ENV_PREFIX}/bin/python" ]] || {
    echo "Missing SPIB environment: ${ENV_PREFIX}" >&2
    echo "Submit dna/install_spib_env.slurm first." >&2
    exit 1
}
[[ -f "${ENV_PREFIX}/.spib-environment-complete" ]] || {
    echo "SPIB environment installation is incomplete: ${ENV_PREFIX}" >&2
    exit 1
}

cd "${CODE_ROOT}"
[[ "$(git symbolic-ref --quiet --short HEAD || true)" == "hsic-spib" ]] || {
    echo "DNA checkout must be on hsic-spib" >&2
    exit 1
}
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "DNA checkout has tracked local changes; refusing to submit." >&2
    git status --short --untracked-files=no
    exit 1
fi
git ls-files --error-unmatch -- "${CONFIG}" >/dev/null
COMMIT="$(git rev-parse HEAD^{commit})"
git cat-file -e "${COMMIT}:${CONFIG}"

mkdir -p "${PROJECT_ROOT}/logs/slurm" "${PROJECT_ROOT}/results/spib" "${PROJECT_ROOT}/data"
JOB_ID="$(sbatch --parsable \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=32G \
    --gres=gpu:1 \
    --time=04:00:00 \
    --output="${PROJECT_ROOT}/logs/slurm/spib-%j.out" \
    --export="ALL,DNA_CODE_ROOT=${CODE_ROOT},DNA_PROJECT_ROOT=${PROJECT_ROOT},DNA_ENV_PREFIX=${ENV_PREFIX},SPIB_GIT_COMMIT=${COMMIT},CONFIG=${CONFIG}" \
    "${CODE_ROOT}/dna/run_spib_advanced.slurm")"

echo "job=${JOB_ID}"
echo "partition=${PARTITION}"
echo "commit=${COMMIT}"
echo "config=${CONFIG}"
echo "log=${PROJECT_ROOT}/logs/spib/${JOB_ID}.log"
echo "output=${PROJECT_ROOT}/results/spib/${JOB_ID}-${COMMIT:0:12}"
