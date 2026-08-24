#!/usr/bin/env bash

# Print a compact, user-facing summary of active NSCC PBS allocations.

set -uo pipefail

TARGET="${1:-${NSCC_TARGET:-a100}}"
[[ $# -le 1 ]] || { echo "Usage: nscc/status_summary.sh [a100|h100|all]" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${TARGET}" == all ]]; then
    bash "$0" a100
    bash "$0" h100
    exit 0
fi
source "${SCRIPT_DIR}/target_config.sh"
nscc_load_target "${TARGET}"
SSH_HOST="${NSCC_SSH_HOST}"
LABEL="${NSCC_LABEL}"

if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "${SSH_HOST}" bash -s -- "${LABEL}" <<'REMOTE'
set -uo pipefail
label="$1"

qstat_bin="$(command -v qstat || true)"
if [[ -z "${qstat_bin}" && -x /opt/pbs/bin/qstat ]]; then
    qstat_bin=/opt/pbs/bin/qstat
fi
[[ -n "${qstat_bin}" && -x "${qstat_bin}" ]] || exit 2

mapfile -t jobs < <("${qstat_bin}" -u "${USER}" 2>/dev/null | awk '$1 ~ /^[0-9]+[.]/ {print $1}')
if ((${#jobs[@]} == 0)); then
    echo "${label}：无活动作业"
    exit 0
fi

to_seconds() {
    local value="$1" h m s
    IFS=: read -r h m s <<<"${value}"
    [[ "${h}" =~ ^[0-9]+$ && "${m}" =~ ^[0-9]+$ && "${s}" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$((10#${h} * 3600 + 10#${m} * 60 + 10#${s}))"
}

format_duration() {
    local seconds="$1" h m
    ((seconds < 0)) && seconds=0
    h=$((seconds / 3600))
    m=$(((seconds % 3600 + 59) / 60))
    if ((m == 60)); then h=$((h + 1)); m=0; fi
    if ((h > 0)); then printf '%dh%02dm' "${h}" "${m}"; else printf '%dm' "${m}"; fi
}

parts=()
for job in "${jobs[@]}"; do
    info="$("${qstat_bin}" -f "${job}" 2>/dev/null || true)"
    [[ -n "${info}" ]] || continue
    name="$(awk '$1 == "Job_Name" {print $3; exit}' <<<"${info}")"
    state="$(awk '$1 == "job_state" {print $3; exit}' <<<"${info}")"
    [[ "${state}" == E ]] && continue
    limit="$(awk '$1 == "Resource_List.walltime" {print $3; exit}' <<<"${info}")"
    used="$(awk '$1 == "resources_used.walltime" {print $3; exit}' <<<"${info}")"
    host="$(awk '$1 == "exec_host" {print $3; exit}' <<<"${info}")"
    host="${host%%/*}"
    mode=batch
    [[ "${name}" == STDIN ]] && mode=interactive

    remaining=等待
    if [[ "${state}" == R ]]; then
        if limit_s="$(to_seconds "${limit}")" && used_s="$(to_seconds "${used:-00:00:00}")"; then
            remaining="剩余$(format_duration "$((limit_s - used_s))")"
        fi
    fi
    node=""
    [[ -n "${host}" ]] && node=" · ${host}"
    parts+=("${mode} ${job} ${state}${node} · ${remaining}")
done

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
