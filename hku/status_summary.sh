#!/usr/bin/env bash

# Print a compact, user-facing summary of the HKU interactive GPU session.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/target_config.sh"
hku_load_target
SSH_HOST="${HKU_SSH_HOST}"
LABEL="${HKU_LABEL}"
SESSION="${HKU_INTERACTIVE_SESSION}"

if ! ssh -o BatchMode=yes -o ConnectTimeout=12 "${SSH_HOST}" bash -s -- "${LABEL}" "${SESSION}" <<'REMOTE'
set -uo pipefail
label="$1"
session="$2"

parts=()
if tmux has-session -t "${session}" 2>/dev/null; then
    state="$(tmux show-option -qv -t "${session}" @spib_auto_init_state 2>/dev/null || echo unknown)"
    parts+=("interactive ${session} ${state}")
fi

if command -v squeue >/dev/null 2>&1; then
    while IFS= read -r line; do
        [[ -n "${line}" ]] || continue
        job="$(awk '{print $1}' <<<"${line}")"
        state="$(awk '{print $5}' <<<"${line}")"
        node="$(awk '{print $8}' <<<"${line}")"
        time="$(awk '{print $6}' <<<"${line}")"
        parts+=("slurm ${job} ${state} · ${node} · ${time}")
    done < <(squeue -h -u "${USER}" -o '%i %j %T %M %N' 2>/dev/null)
fi

if ((${#parts[@]} == 0)); then
    echo "${label}：无活动作业"
else
    joined="${parts[0]}"
    for ((i=1; i<${#parts[@]}; i++)); do joined+="；${parts[i]}"; done
    echo "${label}：${joined}"
fi
REMOTE
then
    echo "${LABEL}：状态未知（SSH 不可用）"
fi
