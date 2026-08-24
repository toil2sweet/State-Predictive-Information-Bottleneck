#!/usr/bin/env bash

# Shared target defaults for ASPIRE2A (A100) and ASPIRE2A+ (H100).
# Source this file, then call: nscc_load_target [a100|h100]

nscc_load_target() {
    local requested="${1:-${NSCC_TARGET:-a100}}"
    case "${requested}" in
        a100|A100|aspire2a|ASPIRE2A) requested=a100 ;;
        h100|H100|aspire2a+|ASPIRE2A+|aspire2p|ASPIRE2P) requested=h100 ;;
        *) echo "Unknown NSCC target: ${requested} (use a100 or h100)" >&2; return 2 ;;
    esac

    NSCC_TARGET="${requested}"
    NSCC_RUNTIME_CODE_ROOT="${NSCC_RUNTIME_CODE_ROOT:-/home/users/nus/depeng/State-Predictive-Information-Bottleneck}"
    NSCC_QUEUE="${NSCC_QUEUE:-normal}"
    NSCC_INTERACTIVE_WALLTIME="${NSCC_INTERACTIVE_WALLTIME:-06:00:00}"
    NSCC_BATCH_WALLTIME="${NSCC_BATCH_WALLTIME:-02:00:00}"

    if [[ "${requested}" == a100 ]]; then
        NSCC_LABEL="ASPIRE2A-A100"
        NSCC_SSH_HOST="${NSCC_SSH_HOST:-nscc-a100}"
        NSCC_PROJECT_ID="${NSCC_PROJECT_ID:-11004454}"
        NSCC_PROJECT_ROOT="${NSCC_PROJECT_ROOT:-/home/project/11004454/depeng}"
        NSCC_INTERACTIVE_SESSION="${NSCC_INTERACTIVE_SESSION:-spib-a100}"
        NSCC_INTERACTIVE_SELECT="${NSCC_INTERACTIVE_SELECT:-1:ngpus=1}"
        NSCC_BATCH_SELECT="${NSCC_BATCH_SELECT:-1:ngpus=1}"
        NSCC_MINIFORGE_MODULE="${NSCC_MINIFORGE_MODULE:-miniforge3/25.3.1}"
    else
        NSCC_LABEL="ASPIRE2A+-H100"
        NSCC_SSH_HOST="${NSCC_SSH_HOST:-nscc}"
        NSCC_PROJECT_ID="${NSCC_PROJECT_ID:-11014454}"
        NSCC_PROJECT_ROOT="${NSCC_PROJECT_ROOT:-/data/projects/11014454/depeng}"
        NSCC_INTERACTIVE_SESSION="${NSCC_INTERACTIVE_SESSION:-spib-gpu}"
        NSCC_INTERACTIVE_SELECT="${NSCC_INTERACTIVE_SELECT:-1:ncpus=4:mem=16gb:ngpus=1}"
        NSCC_BATCH_SELECT="${NSCC_BATCH_SELECT:-1:ncpus=8:mem=32gb:ngpus=1}"
        NSCC_MINIFORGE_MODULE="${NSCC_MINIFORGE_MODULE:-miniforge3/24.3.0}"
    fi
    NSCC_ENV_PREFIX="${NSCC_ENV_PREFIX:-${NSCC_PROJECT_ROOT}/envs/spib}"
    export NSCC_TARGET NSCC_LABEL NSCC_SSH_HOST NSCC_PROJECT_ID NSCC_PROJECT_ROOT
    export NSCC_RUNTIME_CODE_ROOT NSCC_QUEUE NSCC_INTERACTIVE_SESSION
    export NSCC_INTERACTIVE_SELECT NSCC_INTERACTIVE_WALLTIME NSCC_BATCH_SELECT
    export NSCC_BATCH_WALLTIME NSCC_MINIFORGE_MODULE NSCC_ENV_PREFIX
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    nscc_load_target "${1:-a100}"
    printf 'target=%s\nlabel=%s\nssh=%s\nproject_id=%s\nproject_root=%s\nenv=%s\n' \
        "${NSCC_TARGET}" "${NSCC_LABEL}" "${NSCC_SSH_HOST}" "${NSCC_PROJECT_ID}" \
        "${NSCC_PROJECT_ROOT}" "${NSCC_ENV_PREFIX}"
fi
