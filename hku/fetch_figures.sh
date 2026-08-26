#!/usr/bin/env bash

# Run on the local Mac. Copy one HKU run's figures into a short per-run
# directory under this checkout's fig/.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: hku/fetch_figures.sh [RUN_ID]

Copy PNG/PDF/SVG figures from the HKU runtime checkout into
fig/<system>-<job>-<version>-<MMDDTHHMM>/, e.g.
fig/double-well-184423-0.3.0-0824T0800.
<system> comes from the run config (double-well, four-well, muller, ...).
With no RUN_ID, use the latest finished run recorded by spib_run.
If that run produced no figures, print figures=none and exit 0.

Environment overrides:
  SPIB_LOCAL_FIG_DIR  Local figure root (default: <repository>/fig)
EOF
}

hku_run_label() {
    local ts_potential="$1"
    local config="$2"
    local ts config_base
    ts="$(printf '%s' "${ts_potential}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
    case "${ts}" in
        double_well|dw) printf '%s' "double-well"; return 0 ;;
        four_well|fw) printf '%s' "four-well"; return 0 ;;
        muller|muller_brown|mb) printf '%s' "muller"; return 0 ;;
        trpcage|trp_cage|protein) printf '%s' "trpcage"; return 0 ;;
    esac
    config_base="$(basename "${config:-}" | tr '[:upper:]' '[:lower:]')"
    case "${config_base}" in
        *double*well*) printf '%s' "double-well" ;;
        *four*well*) printf '%s' "four-well" ;;
        *muller*) printf '%s' "muller" ;;
        *trpcage*|*trp*cage*) printf '%s' "trpcage" ;;
        *) printf '%s' "run" ;;
    esac
}

hku_local_fig_subdir() {
    local run_id="$1"
    local label="$2"
    local job version stamp
    if [[ "${run_id}" =~ ^interactive-([0-9]+)-(.+)-([0-9]{8}T[0-9]{6}Z)$ ]]; then
        job="${BASH_REMATCH[1]}"
        version="${BASH_REMATCH[2]}"
        stamp="${BASH_REMATCH[3]}"
        printf '%s-%s-%s-%s' "${label}" "${job}" "${version}" "${stamp:4:4}T${stamp:9:4}"
        return 0
    fi
    printf '%s-%s' "${label}" "${run_id#interactive-}"
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
LOCAL_FIG_ROOT="${SPIB_LOCAL_FIG_DIR:-${LOCAL_ROOT}/fig}"
LAST_RUN="${PROJECT_ROOT%/}/results/spib/.hku-last-run"

RESOLVED="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "${SSH_HOST}" bash -s -- \
    "${PROJECT_ROOT%/}" "${RUN_ID:-latest}" "${LAST_RUN}" <<'REMOTE'
set -euo pipefail
project_root="$1"
run_id="$2"
last_run="$3"
[[ "${run_id}" != latest ]] || run_id=""
config=""
ts_potential=""

if [[ -z "${run_id}" ]]; then
    [[ -f "${last_run}" ]] || { echo "missing_pointer"; exit 0; }
    run_id="$(sed -n 's/^run_id=//p' "${last_run}" | head -1)"
    state="$(sed -n 's/^state=//p' "${last_run}" | head -1)"
    fig_dir="$(sed -n 's/^fig_dir=//p' "${last_run}" | head -1)"
    config="$(sed -n 's/^config=//p' "${last_run}" | head -1)"
    echo "pointer_state=${state:-unknown}"
else
    fig_dir="${project_root}/results/spib/${run_id}/fig"
    echo "pointer_state=explicit"
fi
echo "run_id=${run_id:-}"
echo "fig_dir=${fig_dir:-}"

run_dir="${project_root}/results/spib/${run_id:-}"
if [[ -n "${run_id}" && -f "${run_dir}/submitted-config.ini" ]]; then
    ts_potential="$(sed -n 's/^[[:space:]]*ts_potential[[:space:]]*=[[:space:]]*//p' \
        "${run_dir}/submitted-config.ini" | head -1 | awk '{print $1}')"
    if [[ -z "${config}" ]]; then
        config="${run_dir}/submitted-config.ini"
    fi
fi
echo "config=${config:-}"
echo "ts_potential=${ts_potential:-}"

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
config="$(sed -n 's/^config=//p' <<<"${RESOLVED}" | head -1)"
ts_potential="$(sed -n 's/^ts_potential=//p' <<<"${RESOLVED}" | head -1)"

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

label="$(hku_run_label "${ts_potential}" "${config}")"
DESTINATION="${LOCAL_FIG_ROOT%/}/$(hku_local_fig_subdir "${run_id}" "${label}")"
mkdir -p "${DESTINATION}"
rsync -a --compress --exclude='.DS_Store' \
    "${SSH_HOST}:${remote_fig}/" "${DESTINATION}/"

echo "figures=downloaded"
echo "run_id=${run_id}"
echo "system=${label}"
echo "remote=${remote_fig}"
echo "local=${DESTINATION}"
