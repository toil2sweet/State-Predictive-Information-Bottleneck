#!/usr/bin/env bash

# Run on the HKU login node. Watch the tmux pane that owns gpu-interactive and
# inject SPIB setup only after a compute-node prompt is ready.

set -euo pipefail

SESSION="${1:-}"
CODE_ROOT="${2:-/userhome/cs3/lidepeng/TS/State-Predictive-Information-Bottleneck}"
PROJECT_ROOT="${3:-${CODE_ROOT}}"
ENV_PREFIX="${4:-/userhome/cs3/lidepeng/anaconda3}"
CONDA_BASE="${5:-/userhome/cs3/lidepeng/anaconda3}"
[[ $# -ge 1 && $# -le 5 ]] || exit 2
[[ "${SESSION}" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ "${CODE_ROOT}" == /userhome/cs3/lidepeng/TS/State-Predictive-Information-Bottleneck ]] || exit 2

STATE_OPTION=@spib_auto_init_state
tmux has-session -t "${SESSION}" 2>/dev/null || exit 0
tmux set-option -q -t "${SESSION}" "${STATE_OPTION}" monitoring

init_sent=0
sent_at=0
for ((attempt=1; attempt<=360; attempt++)); do
    tmux has-session -t "${SESSION}" 2>/dev/null || exit 0
    pane="$(tmux capture-pane -p -t "${SESSION}" -S -100 2>/dev/null || true)"
    tail_pane="$(tmux capture-pane -p -t "${SESSION}" -S -8 2>/dev/null || true)"

    if grep -Fq 'SPIB interactive environment ready' <<<"${pane}" || \
       grep -Eq '^\(base\) .*@[^:]+:.*[$#][[:space:]]*$' <<<"${pane}"; then
        tmux set-option -q -t "${SESSION}" "${STATE_OPTION}" ready
        exit 0
    fi

    if grep -Fq 'You already have a GPU session running' <<<"${pane}"; then
        tmux set-option -q -t "${SESSION}" "${STATE_OPTION}" failed
        exit 1
    fi

    if ((init_sent == 0)) && \
       grep -Eq '^[^[:space:]]+@[^:]+:.*[$#][[:space:]]*$' <<<"${tail_pane}" && \
       ! grep -Eq '@gpu3gate1:' <<<"${tail_pane}"; then
        printf -v init_command 'cd %q && HKU_PROJECT_ROOT=%q HKU_ENV_PREFIX=%q HKU_CONDA_BASE=%q HKU_RUNTIME_CODE_ROOT=%q source %q' \
            "${CODE_ROOT}" "${PROJECT_ROOT}" "${ENV_PREFIX}" "${CONDA_BASE}" \
            "${CODE_ROOT}" "${CODE_ROOT}/hku/interactive_setup.sh"
        tmux set-option -q -t "${SESSION}" "${STATE_OPTION}" initializing
        tmux send-keys -t "${SESSION}" "${init_command}" C-m
        init_sent=1
        sent_at="${attempt}"
    elif ((init_sent == 1 && attempt - sent_at >= 60)); then
        tmux set-option -q -t "${SESSION}" "${STATE_OPTION}" failed
        exit 1
    fi

    sleep 2
done

tmux has-session -t "${SESSION}" 2>/dev/null && \
    tmux set-option -q -t "${SESSION}" "${STATE_OPTION}" timed-out
exit 1
