#!/usr/bin/env bash

# Run on the local Mac. Keep one SSH connection open, report PBS state changes,
# stream new persistent log lines, and finish when the job leaves the queue.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: nscc/watch_job.sh JOB_ID [a100|h100]

Environment overrides:
  Target defaults to a100; append h100 to select ASPIRE2A+.
  NSCC_POLL_SECONDS  Poll interval from 2 to 60 seconds (default: 5)
  SPIB_AUTO_FETCH_FIGURES  Fetch remote fig/ after completion: 1 or 0 (default: 1)
  SPIB_LOCAL_FIG_DIR       Local figure root (default: <repository>/fig/<target>)
EOF
}

JOB_ID="${1:-}"
TARGET="${2:-${NSCC_TARGET:-a100}}"
[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
case "${JOB_ID}" in -h|--help) usage; exit 0 ;; esac
[[ "${JOB_ID}" =~ ^[0-9]+([.][A-Za-z0-9._-]+)?$ ]] || {
    echo "Invalid PBS job ID: ${JOB_ID}" >&2
    exit 2
}

LOCAL_ROOT="$(git rev-parse --show-toplevel)"
source "${LOCAL_ROOT}/nscc/target_config.sh"
nscc_load_target "${TARGET}"
SSH_HOST="${NSCC_SSH_HOST}"
PROJECT_ROOT="${NSCC_PROJECT_ROOT}"
POLL_SECONDS="${NSCC_POLL_SECONDS:-5}"
AUTO_FETCH_FIGURES="${SPIB_AUTO_FETCH_FIGURES:-1}"
[[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] && ((POLL_SECONDS >= 2 && POLL_SECONDS <= 60)) || {
    echo "NSCC_POLL_SECONDS must be an integer from 2 to 60." >&2
    exit 2
}
[[ "${AUTO_FETCH_FIGURES}" == 0 || "${AUTO_FETCH_FIGURES}" == 1 ]] || {
    echo "SPIB_AUTO_FETCH_FIGURES must be 0 or 1." >&2
    exit 2
}

if ssh "${SSH_HOST}" bash -s -- "${JOB_ID}" "${PROJECT_ROOT}" "${POLL_SECONDS}" <<'REMOTE'
set -uo pipefail

job_id="$1"
project_root="$2"
poll_seconds="$3"
log_file="${project_root}/logs/spib/${job_id}.log"
result_root="${project_root}/results/spib/${job_id}"
last_line=0
last_state=""
missing_count=0
terminal_seen=0

qstat_bin="$(command -v qstat || true)"
if [[ -z "${qstat_bin}" && -x /opt/pbs/bin/qstat ]]; then
    qstat_bin=/opt/pbs/bin/qstat
fi
[[ -n "${qstat_bin}" && -x "${qstat_bin}" ]] || {
    echo "Cannot find an executable qstat." >&2
    exit 1
}

flush_log() {
    local line_count
    [[ -f "${log_file}" ]] || return 0
    line_count="$(wc -l <"${log_file}" | tr -d '[:space:]')"
    [[ "${line_count}" =~ ^[0-9]+$ ]] || return 0
    if ((line_count > last_line)); then
        sed -n "$((last_line + 1)),${line_count}p" "${log_file}"
        last_line="${line_count}"
    fi
}

echo "watch_job=${job_id}"
echo "log=${log_file}"
echo "result=${result_root}"

while :; do
    state="$("${qstat_bin}" -f "${job_id}" 2>/dev/null | awk '$1 == "job_state" {print $3; exit}')"
    if [[ -n "${state}" ]]; then
        missing_count=0
        if [[ "${state}" != "${last_state}" ]]; then
            echo "pbs_state=${state}"
            last_state="${state}"
        fi
        # PBS may move a launch failure through E and then retry it in R. Do
        # not treat E as terminal; wait for a final state or disappearance.
        case "${state}" in
            F|C|X) terminal_seen=1 ;;
        esac
    else
        missing_count=$((missing_count + 1))
    fi

    flush_log

    if ((terminal_seen == 1 || missing_count >= 3)); then
        break
    fi
    sleep "${poll_seconds}"
done

exit_status=""
for _ in {1..15}; do
    flush_log
    if [[ -f "${log_file}" ]]; then
        exit_status="$(sed -n 's/^exit_status=//p' "${log_file}" | tail -1)"
    fi
    [[ "${exit_status}" =~ ^[0-9]+$ ]] && break
    sleep 2
done

echo "pbs_state=finished"
echo "exit_status=${exit_status:-unknown}"
history="$("${qstat_bin}" -xf "${job_id}" 2>/dev/null || true)"
scheduler_exit_status="$(awk '$1 == "Exit_status" {print $3; exit}' <<<"${history}")"
run_count="$(awk '$1 == "run_count" {print $3; exit}' <<<"${history}")"
[[ -n "${scheduler_exit_status}" ]] && echo "scheduler_exit_status=${scheduler_exit_status}"
[[ -n "${run_count}" ]] && echo "run_count=${run_count}"
if [[ -d "${result_root}" ]]; then
    echo "result_files_begin"
    find "${result_root}" -maxdepth 3 -type f -printf '%s\t%p\n' | LC_ALL=C sort -k2
    echo "result_files_end"
else
    echo "result_directory_missing=${result_root}"
fi

if [[ "${exit_status}" =~ ^[0-9]+$ ]]; then
    ((exit_status == 0)) && exit 0
    exit "${exit_status}"
fi
exit 3
REMOTE
then
    remote_status=0
else
    remote_status=$?
fi

# A PBS job cannot write directly to the local Mac. Fetch only its fig/
# subtree after monitoring ends, and retain the actual job exit status even
# when the network transfer itself fails.
if [[ "${AUTO_FETCH_FIGURES}" == 1 ]]; then
    echo "figures_fetch=starting"
    if NSCC_SSH_HOST="${SSH_HOST}" NSCC_PROJECT_ROOT="${PROJECT_ROOT}" \
        SPIB_LOCAL_FIG_DIR="${SPIB_LOCAL_FIG_DIR:-}" \
        bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fetch_figures.sh" "${JOB_ID}" "${NSCC_TARGET}"
    then
        :
    else
        echo "figures_fetch=failed (remote results remain available)" >&2
    fi
fi

exit "${remote_status}"
