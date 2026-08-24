#!/usr/bin/env bash

# Shared defaults for the temporary HKU CS GPU farm target.
# Source this file, then call: hku_load_target

hku_load_target() {
    HKU_LABEL="${HKU_LABEL:-HKU-TS}"
    HKU_SSH_HOST="${HKU_SSH_HOST:-HKUCDS_GPU_Farm}"
    HKU_WORKSPACE_ROOT="${HKU_WORKSPACE_ROOT:-/userhome/cs3/lidepeng/TS}"
    HKU_RUNTIME_CODE_ROOT="${HKU_RUNTIME_CODE_ROOT:-${HKU_WORKSPACE_ROOT}/State-Predictive-Information-Bottleneck}"
    HKU_PROJECT_ROOT="${HKU_PROJECT_ROOT:-${HKU_RUNTIME_CODE_ROOT}}"
    HKU_CONDA_BASE="${HKU_CONDA_BASE:-/userhome/cs3/lidepeng/anaconda3}"
    HKU_ENV_PREFIX="${HKU_ENV_PREFIX:-${HKU_CONDA_BASE}}"
    HKU_INTERACTIVE_SESSION="${HKU_INTERACTIVE_SESSION:-spib-hku}"
    export HKU_LABEL HKU_SSH_HOST HKU_WORKSPACE_ROOT HKU_RUNTIME_CODE_ROOT
    export HKU_PROJECT_ROOT HKU_CONDA_BASE HKU_ENV_PREFIX HKU_INTERACTIVE_SESSION
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    hku_load_target
    printf 'label=%s\nssh=%s\nworkspace=%s\nruntime=%s\nenv=%s\n' \
        "${HKU_LABEL}" "${HKU_SSH_HOST}" "${HKU_WORKSPACE_ROOT}" \
        "${HKU_RUNTIME_CODE_ROOT}" "${HKU_ENV_PREFIX}"
fi
