#!/usr/bin/env bash

# Copy persistent SPIB input data between NSCC targets through a temporary local
# staging directory. Environments, logs, results, and code are never copied.

set -euo pipefail
SOURCE_TARGET="${1:-h100}"
DEST_TARGET="${2:-a100}"
[[ "${SOURCE_TARGET}" != "${DEST_TARGET}" ]] || { echo "Source and destination targets must differ." >&2; exit 2; }
LOCAL_ROOT="$(git rev-parse --show-toplevel)"
source "${LOCAL_ROOT}/nscc/target_config.sh"

nscc_load_target "${SOURCE_TARGET}"
SOURCE_HOST="${NSCC_SSH_HOST}"
SOURCE_ROOT="${NSCC_PROJECT_ROOT}"

# Clear inherited target values before loading the second target.
unset NSCC_TARGET NSCC_LABEL NSCC_SSH_HOST NSCC_PROJECT_ID NSCC_PROJECT_ROOT
unset NSCC_RUNTIME_CODE_ROOT NSCC_QUEUE NSCC_INTERACTIVE_SESSION NSCC_INTERACTIVE_SELECT
unset NSCC_BATCH_SELECT NSCC_MINIFORGE_MODULE NSCC_ENV_PREFIX
nscc_load_target "${DEST_TARGET}"
DEST_HOST="${NSCC_SSH_HOST}"
DEST_ROOT="${NSCC_PROJECT_ROOT}"

STAGING="$(mktemp -d "${TMPDIR:-/tmp}/spib-data-sync.XXXXXX")"
cleanup() { rm -rf -- "${STAGING}"; }
trap cleanup EXIT

mkdir -p "${STAGING}/data"
rsync -az --progress "${SOURCE_HOST}:${SOURCE_ROOT%/}/data/" "${STAGING}/data/"
ssh "${DEST_HOST}" "mkdir -p $(printf '%q' "${DEST_ROOT%/}/data")"
rsync -az --progress "${STAGING}/data/" "${DEST_HOST}:${DEST_ROOT%/}/data/"
echo "data_sync=complete"
echo "source=${SOURCE_TARGET}:${SOURCE_ROOT}/data"
echo "destination=${DEST_TARGET}:${DEST_ROOT}/data"
