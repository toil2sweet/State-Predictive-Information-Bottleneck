#!/usr/bin/env bash

# Copy locally prepared Trp-cage npy arrays into NSCC project storage.
# Code sync deliberately omits *.npy; training reads them via a runtime symlink.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: nscc/sync_trpcage_data.sh [a100|h100]

Copy trpcage/traj_data.npy and the matching mean/std/TICA-kmeans labels into
<project>/data/trpcage/ on the selected NSCC target.
EOF
}

TARGET="${1:-${NSCC_TARGET:-a100}}"
case "${TARGET}" in
    -h|--help) usage; exit 0 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 2; }

LOCAL_ROOT="$(git rev-parse --show-toplevel)"
source "${LOCAL_ROOT}/nscc/target_config.sh"
nscc_load_target "${TARGET}"

SRC="${LOCAL_ROOT}/trpcage"
DEST="${NSCC_PROJECT_ROOT%/}/data/trpcage"
FILES=(
    traj_data.npy
    data_mean.npy
    data_std.npy
    init_label_tica_kmeans200_lag200.npy
)

missing=0
for f in "${FILES[@]}"; do
    if [[ ! -f "${SRC}/${f}" ]]; then
        echo "Missing local array: ${SRC}/${f}" >&2
        missing=1
    fi
done
[[ "${missing}" == 0 ]] || exit 1

ssh "${NSCC_SSH_HOST}" "mkdir -p $(printf '%q' "${DEST}")"
rsync -avP -- "${FILES[@]/#/${SRC}/}" "${NSCC_SSH_HOST}:${DEST}/"

echo "trpcage_data=synced"
echo "target=${NSCC_TARGET}"
echo "remote=${NSCC_SSH_HOST}:${DEST}"
