#!/usr/bin/env bash

set -euo pipefail
TARGET="${1:-${NSCC_TARGET:-a100}}"
LOCAL_ROOT="$(git rev-parse --show-toplevel)"
source "${LOCAL_ROOT}/nscc/target_config.sh"
nscc_load_target "${TARGET}"
bash "${LOCAL_ROOT}/nscc/sync_code.sh" "${NSCC_TARGET}"

quote_arg() { printf '%q' "$1"; }
ssh "${NSCC_SSH_HOST}" \
    "cd $(quote_arg "${NSCC_RUNTIME_CODE_ROOT}") && qsub -P $(quote_arg "${NSCC_PROJECT_ID}") -q $(quote_arg "${NSCC_QUEUE}") -l select=$(quote_arg "${NSCC_INTERACTIVE_SELECT}") -l walltime=00:05:00 -v NSCC_TARGET=$(quote_arg "${NSCC_TARGET}"),NSCC_PROJECT_ROOT=$(quote_arg "${NSCC_PROJECT_ROOT}"),NSCC_ENV_PREFIX=$(quote_arg "${NSCC_ENV_PREFIX}"),NSCC_MINIFORGE_MODULE=$(quote_arg "${NSCC_MINIFORGE_MODULE}") nscc/probe_gpu.pbs"
