#!/usr/bin/env bash

# Initialize target-side project directories and report available software.
# This is intentionally idempotent and does not submit or cancel any job.

set -euo pipefail
TARGET="${1:-${NSCC_TARGET:-a100}}"
LOCAL_ROOT="$(git rev-parse --show-toplevel)"
source "${LOCAL_ROOT}/nscc/target_config.sh"
nscc_load_target "${TARGET}"

ssh "${NSCC_SSH_HOST}" bash -s -- "${NSCC_PROJECT_ROOT}" "${NSCC_PROJECT_ID}" <<'REMOTE'
set -euo pipefail
project_root="$1"
project_id="$2"
project_parent="${project_root%/depeng}"
[[ -d "${project_parent}" ]] || {
    echo "Project filesystem is not mounted at ${project_parent}." >&2
    echo "Check Project ${project_id} membership/path with NSCC support." >&2
    exit 1
}
mkdir -p "${project_root}"/{envs,data,logs/spib,results/spib,run_sources/spib}
echo "project_root=${project_root}"
echo "quota_begin"
quota -s 2>/dev/null || true
echo "quota_end"
echo "miniforge_modules_begin"
module avail miniforge3 2>&1 || true
echo "miniforge_modules_end"
REMOTE

bash "${LOCAL_ROOT}/nscc/sync_code.sh" "${NSCC_TARGET}"
echo "bootstrap=complete"
echo "target=${NSCC_TARGET}"
