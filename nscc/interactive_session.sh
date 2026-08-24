#!/usr/bin/env bash

# Manage a reconnectable NSCC interactive GPU allocation from the local Mac.
# The qsub -I client runs inside tmux on the NSCC login node, so disconnecting
# the local SSH client does not immediately destroy the interactive allocation.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: nscc/interactive_session.sh start|status|attach [a100|h100]

Actions:
  start   Sync, start/reuse a GPU allocation, and auto-activate SPIB when ready.
  status  Show the persistent tmux pane and the user's active PBS jobs.
  attach  Attach to the session; normally opens at an active (spib) prompt.

Environment overrides:
Target defaults to a100. Use h100 as the optional second argument.
NSCC_INTERACTIVE_WALLTIME defaults to 06:00:00.
EOF
}

ACTION="${1:-}"
TARGET="${2:-${NSCC_TARGET:-a100}}"
[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
case "${ACTION}" in
    start|status|attach) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

LOCAL_ROOT="$(git rev-parse --show-toplevel)"
source "${LOCAL_ROOT}/nscc/target_config.sh"
nscc_load_target "${TARGET}"
SSH_HOST="${NSCC_SSH_HOST}"
SESSION="${NSCC_INTERACTIVE_SESSION}"
WALLTIME="${NSCC_INTERACTIVE_WALLTIME}"
SELECT="${NSCC_INTERACTIVE_SELECT}"
CODE_ROOT="${NSCC_RUNTIME_CODE_ROOT}"

[[ "${SESSION}" =~ ^[A-Za-z0-9_-]+$ ]] || {
    echo "Invalid tmux session name: ${SESSION}" >&2
    exit 2
}
[[ "${WALLTIME}" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || {
    echo "NSCC_INTERACTIVE_WALLTIME must use HH:MM:SS." >&2
    exit 2
}
[[ "${SELECT}" =~ ^[A-Za-z0-9:=+._-]+$ ]] || { echo "Invalid PBS select specification." >&2; exit 2; }
[[ "${NSCC_QUEUE}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid PBS queue." >&2; exit 2; }
[[ "${NSCC_PROJECT_ID}" =~ ^[0-9]+$ ]] || { echo "Invalid NSCC project ID." >&2; exit 2; }
if [[ "${ACTION}" == start ]]; then
    cd "${LOCAL_ROOT}"
    bash "${LOCAL_ROOT}/nscc/sync_code.sh" "${NSCC_TARGET}"
fi

ssh "${SSH_HOST}" bash -s -- \
    "${ACTION}" "${SESSION}" "${WALLTIME}" "${SELECT}" "${CODE_ROOT}" \
    "${NSCC_QUEUE}" "${NSCC_PROJECT_ID}" "${NSCC_PROJECT_ROOT}" "${NSCC_ENV_PREFIX}" \
    "${NSCC_MINIFORGE_MODULE}" "${NSCC_TARGET}" <<'REMOTE'
set -euo pipefail

action="$1"
session="$2"
walltime="$3"
select_spec="$4"
code_root="$5"
queue="$6"
project_id="$7"
project_root="$8"
env_prefix="$9"
miniforge_module="${10}"
target="${11}"

command -v tmux >/dev/null 2>&1 || {
    echo "tmux is unavailable on the NSCC login node." >&2
    exit 1
}

qsub_bin="$(command -v qsub || true)"
if [[ -z "${qsub_bin}" && -x /opt/pbs/bin/qsub ]]; then
    qsub_bin=/opt/pbs/bin/qsub
fi
qstat_bin="$(command -v qstat || true)"
if [[ -z "${qstat_bin}" && -x /opt/pbs/bin/qstat ]]; then
    qstat_bin=/opt/pbs/bin/qstat
fi
[[ -n "${qsub_bin}" && -x "${qsub_bin}" ]] || {
    echo "Cannot find an executable qsub." >&2
    exit 1
}
[[ -n "${qstat_bin}" && -x "${qstat_bin}" ]] || {
    echo "Cannot find an executable qstat." >&2
    exit 1
}

if [[ "${action}" == start ]]; then
    if tmux has-session -t "${session}" 2>/dev/null; then
        echo "interactive_session=reused"
    else
        command="exec ${qsub_bin} -I -q ${queue} -l select=${select_spec} -l walltime=${walltime} -P ${project_id}"
        tmux new-session -d -s "${session}" "${command}"
        echo "interactive_session=started"
    fi
fi

if ! tmux has-session -t "${session}" 2>/dev/null; then
    echo "interactive_session=missing"
    exit 1
fi

auto_script="${code_root}/nscc/interactive_auto_init.sh"
auto_log="/tmp/${USER}-${session}-spib-auto-init.log"
auto_state="$(tmux show-option -qv -t "${session}" @spib_auto_init_state 2>/dev/null || true)"
case "${auto_state}" in
    monitoring|initializing|ready) ;;
    *)
        [[ -x "${auto_script}" ]] || {
            echo "Missing executable auto-initializer: ${auto_script}" >&2
            exit 1
        }
        tmux set-option -q -t "${session}" @spib_auto_init_state monitoring
        nohup bash "${auto_script}" "${session}" "${code_root}" "${project_root}" \
            "${env_prefix}" "${miniforge_module}" "${target}" \
            >"${auto_log}" 2>&1 </dev/null &
        auto_state=monitoring
        ;;
esac

if [[ "${action}" == attach ]]; then
    echo "auto_init=${auto_state}"
    exit 0
fi

echo "session=${session}"
echo "target=${target}"
echo "walltime=${walltime}"
echo "auto_init=$(tmux show-option -qv -t "${session}" @spib_auto_init_state 2>/dev/null || echo unknown)"
echo "attach=bash nscc/interactive_session.sh attach"
echo "pane_begin"
tmux capture-pane -p -t "${session}" -S -30 || true
echo "pane_end"
echo "pbs_jobs_begin"
"${qstat_bin}" -u "${USER}" 2>&1 || true
echo "pbs_jobs_end"
REMOTE

if [[ "${ACTION}" == attach ]]; then
    exec ssh -tt "${SSH_HOST}" "tmux attach-session -t '${SESSION}'"
fi
