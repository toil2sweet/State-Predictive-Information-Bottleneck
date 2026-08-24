#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: nscc/submit_working.sh [config.ini] [a100|h100]

Sync, freeze, and submit the current local working tree through the single NSCC
runtime mirror. This does not stage, commit, tag, or push anything.
EOF
}

CONFIG="${1:-examples/Four_Well_hsic_config.ini}"
TARGET="${2:-${NSCC_TARGET:-a100}}"
[[ $# -le 2 ]] || { usage >&2; exit 2; }
case "${CONFIG}" in -h|--help) usage; exit 0 ;; esac
[[ "${CONFIG}" != /* && "${CONFIG}" != *,* && "${CONFIG}" != *$'\n'* && "${CONFIG}" != *[[:space:]]* ]] || { echo "Invalid CONFIG." >&2; exit 2; }
[[ "${CONFIG}" != .. && "${CONFIG}" != ../* && "${CONFIG}" != */../* && "${CONFIG}" != */.. ]] || { echo "CONFIG must not escape the repository." >&2; exit 2; }

LOCAL_ROOT="$(git rev-parse --show-toplevel)"
source "${LOCAL_ROOT}/nscc/target_config.sh"
nscc_load_target "${TARGET}"
SSH_HOST="${NSCC_SSH_HOST}"
REMOTE_ROOT="${NSCC_RUNTIME_CODE_ROOT}"
PROJECT_ROOT="${NSCC_PROJECT_ROOT}"
ENV_PREFIX="${NSCC_ENV_PREFIX}"

cd "${LOCAL_ROOT}"
[[ -f "${CONFIG}" ]] || { echo "Missing local config: ${LOCAL_ROOT}/${CONFIG}" >&2; exit 1; }
bash "${LOCAL_ROOT}/nscc/sync_code.sh" "${NSCC_TARGET}"

quote_args() { printf '%q ' "$@"; }
REMOTE_COMMAND="cd $(quote_args "${REMOTE_ROOT}")&& NSCC_TARGET=$(quote_args "${NSCC_TARGET}")NSCC_SOURCE_CODE_ROOT=$(quote_args "${REMOTE_ROOT}")NSCC_PROJECT_ROOT=$(quote_args "${PROJECT_ROOT}")NSCC_ENV_PREFIX=$(quote_args "${ENV_PREFIX}")NSCC_PROJECT_ID=$(quote_args "${NSCC_PROJECT_ID}")NSCC_QUEUE=$(quote_args "${NSCC_QUEUE}")NSCC_BATCH_SELECT=$(quote_args "${NSCC_BATCH_SELECT}")NSCC_BATCH_WALLTIME=$(quote_args "${NSCC_BATCH_WALLTIME}")NSCC_MINIFORGE_MODULE=$(quote_args "${NSCC_MINIFORGE_MODULE}")bash nscc/submit_job.sh --working $(quote_args "${CONFIG}")"
ssh "${SSH_HOST}" "${REMOTE_COMMAND}"
