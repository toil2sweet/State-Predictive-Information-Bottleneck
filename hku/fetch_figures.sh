#!/usr/bin/env bash

# Run on the local Mac. Copy one HKU run's figures into this checkout's fig/.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: hku/fetch_figures.sh [RUN_ID]

Copy PNG/PDF figures from the HKU runtime checkout into this repository's fig/
directory. With no RUN_ID, use the latest finished run recorded by spib_run.
If that run produced no figures, print figures=none and exit 0.

Environment overrides:
  SPIB_LOCAL_FIG_DIR  Local figure directory (default: <repository>/fig)
EOF
}

RUN_ID="${1:-}"
case "${RUN_ID}" in
    -h|--help) usage; exit 0 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 2; }
if [[ -n "${RUN_ID}" && ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid run ID: ${RUN_ID}" >&2
    exit 2
fi

LOCAL_ROOT="$(git rev-parse --show-toplevel)"
source "${LOCAL_ROOT}/hku/target_config.sh"
hku_load_target
SSH_HOST="${HKU_SSH_HOST}"
PROJECT_ROOT="${HKU_PROJECT_ROOT}"
LOCAL_FIG_DIR="${SPIB_LOCAL_FIG_DIR:-${LOCAL_ROOT}/fig}"
LAST_RUN="${PROJECT_ROOT%/}/results/spib/.hku-last-run"

RESOLVED="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "${SSH_HOST}" bash -s -- \
    "${PROJECT_ROOT%/}" "${RUN_ID:-latest}" "${LAST_RUN}" <<'REMOTE'
set -euo pipefail
project_root="$1"
run_id="$2"
last_run="$3"
[[ "${run_id}" != latest ]] || run_id=""

if [[ -z "${run_id}" ]]; then
    [[ -f "${last_run}" ]] || { echo "missing_pointer"; exit 0; }
    run_id="$(sed -n 's/^run_id=//p' "${last_run}" | head -1)"
    state="$(sed -n 's/^state=//p' "${last_run}" | head -1)"
    fig_dir="$(sed -n 's/^fig_dir=//p' "${last_run}" | head -1)"
    echo "pointer_state=${state:-unknown}"
else
    fig_dir="${project_root}/results/spib/${run_id}/fig"
    echo "pointer_state=explicit"
fi
echo "run_id=${run_id:-}"
echo "fig_dir=${fig_dir:-}"

if [[ -z "${run_id}" || -z "${fig_dir}" ]]; then
    echo "probe=missing_pointer"
    exit 0
fi
if [[ ! -d "${fig_dir}" ]]; then
    echo "probe=missing"
    exit 0
fi
if find "${fig_dir}" \( -name '*.png' -o -name '*.pdf' -o -name '*.svg' \) \
    -type f -print -quit | grep -q .; then
    echo "probe=present"
else
    echo "probe=empty"
fi
REMOTE
)" || {
    echo "Could not inspect HKU figures." >&2
    exit 1
}

pointer_state="$(sed -n 's/^pointer_state=//p' <<<"${RESOLVED}" | head -1)"
run_id="$(sed -n 's/^run_id=//p' <<<"${RESOLVED}" | head -1)"
remote_fig="$(sed -n 's/^fig_dir=//p' <<<"${RESOLVED}" | head -1)"
probe="$(sed -n 's/^probe=//p' <<<"${RESOLVED}" | head -1)"

if [[ "${pointer_state}" == running ]]; then
    echo "figures=pending"
    echo "run_id=${run_id}"
    echo "remote=${remote_fig}"
    exit 0
fi

case "${probe}" in
    present) ;;
    missing|empty|missing_pointer|"")
        echo "figures=none"
        echo "run_id=${run_id}"
        echo "remote=${remote_fig}"
        exit 0
        ;;
    *)
        echo "Unexpected remote figure probe result: ${probe}" >&2
        exit 1
        ;;
esac

mkdir -p "${LOCAL_FIG_DIR}"
rsync -a --compress --exclude='.DS_Store' \
    "${SSH_HOST}:${remote_fig}/" "${LOCAL_FIG_DIR}/"

echo "figures=downloaded"
echo "run_id=${run_id}"
echo "remote=${remote_fig}"
echo "local=${LOCAL_FIG_DIR}"
