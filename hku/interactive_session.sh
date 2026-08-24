#!/usr/bin/env bash

# Manage a reconnectable HKU gpu-interactive allocation from the local Mac.
# The gpu-interactive client runs inside tmux on gpu3gate1, so disconnecting
# the local SSH client does not immediately destroy the GPU shell.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: hku/interactive_session.sh start|status|attach

Actions:
  start   Sync, start/reuse gpu-interactive, and auto-activate SPIB when ready.
  status  Show the persistent tmux pane and the user's Slurm jobs.
  attach  Attach to the session; normally opens at an active (base) prompt.
EOF
}

ACTION="${1:-}"
[[ $# -eq 1 ]] || { usage >&2; exit 2; }
case "${ACTION}" in
    start|status|attach) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

LOCAL_ROOT="$(git rev-parse --show-toplevel)"
source "${LOCAL_ROOT}/hku/target_config.sh"
hku_load_target
SSH_HOST="${HKU_SSH_HOST}"
SESSION="${HKU_INTERACTIVE_SESSION}"
CODE_ROOT="${HKU_RUNTIME_CODE_ROOT}"

[[ "${SESSION}" =~ ^[A-Za-z0-9_-]+$ ]] || {
    echo "Invalid tmux session name: ${SESSION}" >&2
    exit 2
}
if [[ "${ACTION}" == start ]]; then
    cd "${LOCAL_ROOT}"
    bash "${LOCAL_ROOT}/hku/sync_code.sh"
fi

ssh "${SSH_HOST}" bash -s -- \
    "${ACTION}" "${SESSION}" "${CODE_ROOT}" "${HKU_PROJECT_ROOT}" \
    "${HKU_ENV_PREFIX}" "${HKU_CONDA_BASE}" <<'REMOTE'
set -euo pipefail

action="$1"
session="$2"
code_root="$3"
project_root="$4"
env_prefix="$5"
conda_base="$6"

command -v tmux >/dev/null 2>&1 || {
    echo "tmux is unavailable on the HKU login node." >&2
    exit 1
}
command -v gpu-interactive >/dev/null 2>&1 || {
    echo "gpu-interactive is unavailable on the HKU login node." >&2
    exit 1
}

if [[ "${action}" == start ]]; then
    if tmux has-session -t "${session}" 2>/dev/null; then
        echo "interactive_session=reused"
    else
        tmux new-session -d -s "${session}" "exec gpu-interactive"
        echo "interactive_session=started"
    fi
fi

if ! tmux has-session -t "${session}" 2>/dev/null; then
    echo "interactive_session=missing"
    exit 1
fi

auto_script="${code_root}/hku/interactive_auto_init.sh"
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
            "${env_prefix}" "${conda_base}" \
            >"${auto_log}" 2>&1 </dev/null &
        auto_state=monitoring
        ;;
esac

if [[ "${action}" == attach ]]; then
    echo "auto_init=${auto_state}"
    exit 0
fi

echo "session=${session}"
echo "target=hku"
echo "auto_init=$(tmux show-option -qv -t "${session}" @spib_auto_init_state 2>/dev/null || echo unknown)"
echo "attach=bash hku/interactive_session.sh attach"
echo "pane_begin"
tmux capture-pane -p -t "${session}" -S -30 || true
echo "pane_end"
echo "slurm_jobs_begin"
squeue -u "${USER}" 2>&1 || true
echo "slurm_jobs_end"
REMOTE

if [[ "${ACTION}" == attach ]]; then
    exec ssh -tt "${SSH_HOST}" "tmux attach-session -t '${SESSION}'"
fi
