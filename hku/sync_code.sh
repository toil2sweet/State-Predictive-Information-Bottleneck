#!/usr/bin/env bash

# Mirror the local working tree to the dedicated HKU runtime checkout under TS.
# Never sync onto /userhome/cs3/lidepeng/TS itself; that directory holds other
# projects (v1/v2/v3, ts-dar, CTClustering) that must stay untouched.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: hku/sync_code.sh [--dry-run]

Synchronize the local hsic-spib working tree, Four-Well and Double-Well
traj_gen files, double-well_CTC/*.npy trajectories and labels, prepared
Müller and Trp-cage npy arrays, and plotting scripts to:
  /userhome/cs3/lidepeng/TS/State-Predictive-Information-Bottleneck

Old versioned checkouts under TS are not overwritten.
EOF
}

DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ $# -le 1 ]] || { usage >&2; exit 2; }

LOCAL_ROOT="$(git rev-parse --show-toplevel)"
source "${LOCAL_ROOT}/hku/target_config.sh"
hku_load_target
SSH_HOST="${HKU_SSH_HOST}"
REMOTE_ROOT="${HKU_RUNTIME_CODE_ROOT}"
EXPECTED_BRANCH="${HKU_GIT_BRANCH:-hsic-spib}"
REMOTE_ROOT="${REMOTE_ROOT%/}"

case "${REMOTE_ROOT}" in
    /userhome/cs3/lidepeng/TS/State-Predictive-Information-Bottleneck) ;;
    *)
        [[ "${HKU_ALLOW_CUSTOM_RUNTIME_ROOT:-0}" == 1 ]] || {
            echo "Refusing to sync outside the dedicated HKU runtime path:" >&2
            echo "  ${REMOTE_ROOT}" >&2
            exit 2
        }
        ;;
esac
[[ "${REMOTE_ROOT}" != /userhome/cs3/lidepeng/TS ]] || {
    echo "Refusing to sync onto the TS workspace root." >&2
    exit 2
}

cd "${LOCAL_ROOT}"
CURRENT_BRANCH="$(git symbolic-ref --quiet --short HEAD || true)"
[[ "${CURRENT_BRANCH}" == "${EXPECTED_BRANCH}" ]] || {
    echo "Local checkout must be on ${EXPECTED_BRANCH}; current branch is ${CURRENT_BRANCH:-detached HEAD}." >&2
    exit 1
}

SYNC_TMP="$(mktemp -d "${TMPDIR:-/tmp}/spib-hku-sync.XXXXXX")"
cleanup() {
    local status=$?
    trap - EXIT
    rm -rf -- "${SYNC_TMP}"
    exit "${status}"
}
trap cleanup EXIT

STAGING_ROOT="${SYNC_TMP}/tree"
FILE_LIST="${SYNC_TMP}/files.list"
mkdir -p "${STAGING_ROOT}"
: >"${FILE_LIST}"

exclude_generated_path() {
    case "$1" in
        .DS_Store|*/.DS_Store|\
        *.h5|*.hdf5|*.csv|*.tsv|*.dat|\
        *.parquet|*.feather|*.pkl|*.pickle|*.nc|\
        *.xtc|*.dcd|*.trr|*.pdb|*.gro|*.top|*.tpr|\
        *.pt|*.pth|*.ckpt|*.log|*.out|\
        *.png|*.jpg|*.jpeg|*.gif|*.tif|*.tiff|*.mp4|*.mov|\
        *.zip|*.tar|*.tgz|*.gz|*.bz2|*.xz|*.sif|*.sqsh|*.pyc|\
        SPIB/*|fig/*|data/*|results/*|logs/*|wandb/*|.agents/*|\
        __pycache__/*|*/__pycache__/*|\
        HSIC-bottleneck/*|spib_msm/*|ts-dar/*|\
        muller/*.npy|trpcage/*.npy)
            return 0
            ;;
    esac
    return 1
}

exclude_untracked_path() {
    case "$1" in
        CHANGELOG.md|examples/Four_Well_hsic_hku_config.ini|examples/Double_Well_hsic_hku_config.ini)
            return 1
            ;;
        traj_gen/*.npy|examples/Four_Well_*.npy|double-well_CTC/*.npy)
            return 1
            ;;
        *.md|*.pdf|*.sif|*.sqsh|*.zip|*.tar|*.tgz|*.gz|*.bz2|*.xz)
            return 0
            ;;
    esac
    exclude_generated_path "$1"
}

while IFS= read -r -d '' path; do
    if [[ -e "${path}" || -L "${path}" ]] && ! exclude_generated_path "${path}"; then
        printf '%s\0' "${path}" >>"${FILE_LIST}"
    fi
done < <(git ls-files -z)

while IFS= read -r -d '' path; do
    exclude_untracked_path "${path}" || printf '%s\0' "${path}" >>"${FILE_LIST}"
done < <(git ls-files --others --exclude-standard -z)

# gitignore hides traj_gen/*.npy; include them explicitly for HKU runs.
if [[ -d traj_gen ]]; then
    while IFS= read -r -d '' path; do
        printf '%s\0' "${path}" >>"${FILE_LIST}"
    done < <(find traj_gen -maxdepth 1 -type f \( -name 'Four_Well_*.npy' -o -name 'Double_Well_*.npy' \) -print0)
fi
if [[ -d examples ]]; then
    while IFS= read -r -d '' path; do
        printf '%s\0' "${path}" >>"${FILE_LIST}"
    done < <(find examples -maxdepth 1 -type f -name 'Four_Well_*.npy' -print0)
fi
if [[ -d double-well_CTC ]]; then
    while IFS= read -r -d '' path; do
        printf '%s\0' "${path}" >>"${FILE_LIST}"
    done < <(find double-well_CTC -maxdepth 1 -type f -name '*.npy' -print0)
fi
# gitignore hides muller/*.npy; include the canonical xy_kmeans training arrays.
if [[ -d muller ]]; then
    while IFS= read -r -d '' path; do
        printf '%s\0' "${path}" >>"${FILE_LIST}"
    done < <(find muller -maxdepth 1 -type f \( \
        -name 'traj_data.npy' -o -name 'init_label_kmeans20.npy' \) -print0)
fi
# gitignore hides trpcage/*.npy; include the prepared 2024 SPIB arrays.
if [[ -d trpcage ]]; then
    while IFS= read -r -d '' path; do
        printf '%s\0' "${path}" >>"${FILE_LIST}"
    done < <(find trpcage -maxdepth 1 -type f \( \
        -name 'traj_data.npy' -o -name 'data_mean.npy' -o -name 'data_std.npy' \
        -o -name 'init_label_tica_kmeans200_lag200_index.npy' \) -print0)
fi

rsync -a -r --from0 --files-from="${FILE_LIST}" \
    "${LOCAL_ROOT}/" "${STAGING_ROOT}/"

BASE_COMMIT="$(git rev-parse HEAD^{commit})"
TRACKED_STATE=clean
if ! git diff --quiet || ! git diff --cached --quiet; then TRACKED_STATE=modified; fi
CODE_VERSION="$(tr -d '[:space:]' <VERSION 2>/dev/null || true)"
[[ -n "${CODE_VERSION}" ]] || CODE_VERSION=unversioned

{
    echo "source=local-working-tree"
    echo "branch=${CURRENT_BRANCH}"
    echo "base_commit=${BASE_COMMIT}"
    echo "code_version=${CODE_VERSION}"
    echo "tracked_state=${TRACKED_STATE}"
    echo "target=hku"
    echo "synced_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${STAGING_ROOT}/.hku-source-info"

git status --short --untracked-files=no >"${STAGING_ROOT}/.hku-source-status"
git diff --binary HEAD >"${STAGING_ROOT}/.hku-source.patch"

(
    cd "${STAGING_ROOT}"
    find . -type f ! -name '.hku-source-manifest.sha256' \
        -exec shasum -a 256 {} + | LC_ALL=C sort
) >"${STAGING_ROOT}/.hku-source-manifest.sha256"

quote_arg() { printf '%q' "$1"; }
REMOTE_ROOT_Q="$(quote_arg "${REMOTE_ROOT}")"
PARENT_Q="$(quote_arg "$(dirname "${REMOTE_ROOT}")")"

set +e
ssh -o BatchMode=yes -o ConnectTimeout=20 "${SSH_HOST}" "test -d ${PARENT_Q}"
REMOTE_PARENT_STATUS=$?
set -e
if [[ "${REMOTE_PARENT_STATUS}" -eq 255 ]]; then
    echo "Cannot authenticate to ${SSH_HOST}; HKU runtime was not inspected or changed." >&2
    exit 1
fi
if [[ "${REMOTE_PARENT_STATUS}" -ne 0 ]]; then
    echo "Missing HKU workspace parent: $(dirname "${REMOTE_ROOT}")" >&2
    exit 1
fi

if [[ "${DRY_RUN}" == 0 ]]; then
    ssh "${SSH_HOST}" "mkdir -p ${REMOTE_ROOT_Q}"
fi

SYNC_EXCLUDES=(
    --exclude=.git/
    --exclude=fig/
    --exclude=SPIB/
    --exclude=data/
    --exclude=results/
    --exclude=logs/
    --exclude=wandb/
    --exclude=envs/
)
# Do not compress: Trp-cage float32 npy files are large and already dense.
RSYNC_OPTIONS=(-a --omit-dir-times --itemize-changes "${SYNC_EXCLUDES[@]}")
[[ "${DRY_RUN}" == 0 ]] || RSYNC_OPTIONS+=(--dry-run)
rsync "${RSYNC_OPTIONS[@]}" "${STAGING_ROOT}/" "${SSH_HOST}:${REMOTE_ROOT}/"

if [[ "${DRY_RUN}" == 1 ]]; then
    echo "dry_run=complete"
    echo "remote=${SSH_HOST}:${REMOTE_ROOT}"
    exit 0
fi

VERIFY_OUTPUT="$(rsync -anci --omit-dir-times "${SYNC_EXCLUDES[@]}" \
    "${STAGING_ROOT}/" "${SSH_HOST}:${REMOTE_ROOT}/")"
if [[ -n "${VERIFY_OUTPUT}" ]]; then
    echo "HKU runtime verification failed; remaining differences:" >&2
    echo "${VERIFY_OUTPUT}" >&2
    exit 1
fi

echo "sync=verified"
echo "target=hku"
echo "remote=${SSH_HOST}:${REMOTE_ROOT}"
echo "branch=${CURRENT_BRANCH}"
echo "base_commit=${BASE_COMMIT}"
echo "code_version=${CODE_VERSION}"
echo "tracked_state=${TRACKED_STATE}"
