#!/usr/bin/env bash

# From the local Mac: sync, reuse/start gpu-interactive, run one config, then
# download any produced figures into this checkout's fig/ directory.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: hku/run.sh [CONFIG]

Synchronize the local tree, start or reuse the HKU interactive GPU session,
run CONFIG with spib_run, and copy any PNG/PDF/SVG figures into fig/.
CONFIG defaults to examples/Four_Well_hsic_hku_config.ini.
EOF
}

CONFIG="${1:-examples/Four_Well_hsic_hku_config.ini}"
case "${CONFIG}" in
    -h|--help) usage; exit 0 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 2; }
[[ "${CONFIG}" != /* && "${CONFIG}" != .. && "${CONFIG}" != ../* && "${CONFIG}" != */../* ]] || {
    echo "Config must be a repository-relative path." >&2
    exit 2
}

LOCAL_ROOT="$(git rev-parse --show-toplevel)"
source "${LOCAL_ROOT}/hku/target_config.sh"
hku_load_target
SSH_HOST="${HKU_SSH_HOST}"
SESSION="${HKU_INTERACTIVE_SESSION}"
LAST_RUN="${HKU_PROJECT_ROOT%/}/results/spib/.hku-last-run"

[[ -f "${LOCAL_ROOT}/${CONFIG}" ]] || {
    echo "Missing local config: ${LOCAL_ROOT}/${CONFIG}" >&2
    exit 1
}

cd "${LOCAL_ROOT}"
bash "${LOCAL_ROOT}/hku/interactive_session.sh" start

previous="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "${SSH_HOST}" \
    "sed -n 's/^run_id=//p' '${LAST_RUN}' 2>/dev/null | head -1" || true)"

ready=0
for ((attempt=1; attempt<=180; attempt++)); do
    status_out="$(bash "${LOCAL_ROOT}/hku/interactive_session.sh" status || true)"
    if grep -Eq '^auto_init=ready$' <<<"${status_out}" || \
       grep -Fq 'SPIB interactive environment ready' <<<"${status_out}"; then
        ready=1
        break
    fi
    if grep -Eq 'auto_init=(failed|timed-out)' <<<"${status_out}"; then
        echo "${status_out}" >&2
        echo "HKU interactive setup failed." >&2
        exit 1
    fi
    sleep 2
done
[[ "${ready}" == 1 ]] || {
    echo "Timed out waiting for gpu-interactive / (base) setup." >&2
    exit 1
}

printf -v run_cmd 'spib_run %q' "${CONFIG}"
ssh "${SSH_HOST}" "tmux send-keys -t '${SESSION}' $(printf '%q' "${run_cmd}") C-m"

finished=0
run_id=""
for ((attempt=1; attempt<=43200; attempt++)); do
    meta="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "${SSH_HOST}" \
        "cat '${LAST_RUN}' 2>/dev/null" || true)"
    run_id="$(sed -n 's/^run_id=//p' <<<"${meta}" | head -1)"
    state="$(sed -n 's/^state=//p' <<<"${meta}" | head -1)"
    if [[ -n "${run_id}" && "${run_id}" != "${previous}" && "${state}" == finished ]]; then
        finished=1
        break
    fi
    sleep 5
done
[[ "${finished}" == 1 ]] || {
    echo "Timed out waiting for ${CONFIG} to finish on HKU." >&2
    exit 1
}

echo "interactive_run=${run_id}"
bash "${LOCAL_ROOT}/hku/fetch_figures.sh" "${run_id}"
