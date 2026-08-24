#!/usr/bin/env bash

# Run on the local Mac. Copy one NSCC run's figure directory into the local
# checkout without retrieving trajectories, checkpoints, or other result data.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: nscc/fetch_figures.sh RUN_ID [a100|h100]

Copy the selected target's results/spib/RUN_ID/fig/ into this checkout's
fig/<target>/<job-id>-<version>/ directory.
The destination can be overridden with SPIB_LOCAL_FIG_DIR.

Environment overrides:
  Target defaults to a100; append h100 to select ASPIRE2A+.
  SPIB_LOCAL_FIG_DIR  Local figure root (default: <repository>/fig/<target>)
EOF
}

RUN_ID="${1:-}"
TARGET="${2:-${NSCC_TARGET:-a100}}"
[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
case "${RUN_ID}" in -h|--help) usage; exit 0 ;; esac
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "Invalid run ID: ${RUN_ID}" >&2
    exit 2
}

LOCAL_ROOT="$(git rev-parse --show-toplevel)"
source "${LOCAL_ROOT}/nscc/target_config.sh"
nscc_load_target "${TARGET}"
SSH_HOST="${NSCC_SSH_HOST}"
PROJECT_ROOT="${NSCC_PROJECT_ROOT}"
LOCAL_FIG_ROOT="${SPIB_LOCAL_FIG_DIR:-${LOCAL_ROOT}/fig/${NSCC_TARGET}}"
REMOTE_FIG_DIR="${PROJECT_ROOT%/}/results/spib/${RUN_ID}/fig"

# Resolve the code version from the immutable run metadata. Interactive runs
# include a timestamp in RUN_ID, but the local destination intentionally uses
# only the PBS job id and version so repeated fetches land in one predictable
# directory.
code_version="$(ssh "${SSH_HOST}" bash -s -- "${PROJECT_ROOT%/}/results/spib/${RUN_ID}/.nscc-source-info" <<'REMOTE'
set -euo pipefail
metadata="$1"
sed -n 's/^code_version=//p' "${metadata}" | head -1
REMOTE
)" || code_version=""
[[ -n "${code_version}" ]] || code_version="unversioned"
code_version="${code_version//[^0-9A-Za-z._-]/_}"
job_id="${RUN_ID}"
if [[ "${job_id}" == interactive-* ]]; then
    job_id="${job_id#interactive-}"
    job_id="${job_id%%-*}"
fi
DESTINATION="${LOCAL_FIG_ROOT%/}/${job_id}-${code_version}"

probe="$(ssh "${SSH_HOST}" bash -s -- "${REMOTE_FIG_DIR}" <<'REMOTE'
set -euo pipefail
figure_dir="$1"
if [[ ! -d "${figure_dir}" ]]; then
    echo missing
elif find "${figure_dir}" -type f -print -quit | grep -q .; then
    echo present
else
    echo empty
fi
REMOTE
)" || {
    echo "Could not inspect remote figure directory: ${REMOTE_FIG_DIR}" >&2
    exit 1
}

case "${probe}" in
    present) ;;
    missing|empty)
        echo "figures=none"
        echo "remote=${REMOTE_FIG_DIR}"
        exit 0
        ;;
    *)
        echo "Unexpected remote figure probe result: ${probe}" >&2
        exit 1
        ;;
esac

mkdir -p "${DESTINATION}"
rsync -a --compress --exclude='.DS_Store' \
    "${SSH_HOST}:${REMOTE_FIG_DIR}/" "${DESTINATION}/"

echo "figures=downloaded"
echo "target=${NSCC_TARGET}"
echo "remote=${REMOTE_FIG_DIR}"
echo "local=${DESTINATION}"
