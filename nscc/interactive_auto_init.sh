#!/usr/bin/env bash

# Run on the NSCC login node. Watch the tmux pane that owns qsub -I and inject
# the SPIB setup command only after a compute-node prompt is ready.

set -euo pipefail

SESSION="${1:-}"
CODE_ROOT="${2:-/home/users/nus/depeng/State-Predictive-Information-Bottleneck}"
PROJECT_ROOT="${3:-}"
ENV_PREFIX="${4:-}"
MINIFORGE_MODULE="${5:-}"
TARGET="${6:-a100}"
[[ $# -ge 1 && $# -le 6 ]] || exit 2
[[ "${SESSION}" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ "${CODE_ROOT}" == /home/users/nus/depeng/State-Predictive-Information-Bottleneck ]] || exit 2

STATE_OPTION=@spib_auto_init_state
tmux has-session -t "${SESSION}" 2>/dev/null || exit 0
tmux set-option -q -t "${SESSION}" "${STATE_OPTION}" monitoring

init_sent=0
sent_at=0
for ((attempt=1; attempt<=17280; attempt++)); do
    tmux has-session -t "${SESSION}" 2>/dev/null || exit 0
    pane="$(tmux capture-pane -p -t "${SESSION}" -S -100 2>/dev/null || true)"

    if grep -Fq 'SPIB interactive environment ready' <<<"${pane}" || \
       grep -Eq '^\(spib\) .*@[^:]+:.*[$#][[:space:]]*$' <<<"${pane}"; then
        tmux set-option -q -t "${SESSION}" "${STATE_OPTION}" ready
        exit 0
    fi

    if ((init_sent == 0)) && \
       grep -Eq 'qsub: job [^[:space:]]+ ready' <<<"${pane}" && \
       grep -Eq '^[^[:space:]]+@[^:]+:.*[$#][[:space:]]*$' <<<"${pane}"; then
        printf -v init_command 'cd %q && NSCC_TARGET=%q NSCC_PROJECT_ROOT=%q NSCC_ENV_PREFIX=%q NSCC_MINIFORGE_MODULE=%q source %q' \
            "${CODE_ROOT}" "${TARGET}" "${PROJECT_ROOT}" "${ENV_PREFIX}" "${MINIFORGE_MODULE}" \
            "${CODE_ROOT}/nscc/interactive_setup.sh"
        tmux set-option -q -t "${SESSION}" "${STATE_OPTION}" initializing
        tmux send-keys -t "${SESSION}" "${init_command}" C-m
        init_sent=1
        sent_at="${attempt}"
    elif ((init_sent == 1 && attempt - sent_at >= 120)); then
        tmux set-option -q -t "${SESSION}" "${STATE_OPTION}" failed
        exit 1
    fi

    sleep 5
done

tmux has-session -t "${SESSION}" 2>/dev/null && \
    tmux set-option -q -t "${SESSION}" "${STATE_OPTION}" timed-out
exit 1
