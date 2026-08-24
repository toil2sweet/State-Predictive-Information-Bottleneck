#!/usr/bin/env bash

# Mirror the local working tree to the single disposable NSCC runtime checkout.
# Git history remains local/GitHub; the server checkout is never a version store.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: nscc/sync_code.sh [a100|h100] [--dry-run]

Synchronize the local hsic-spib working tree to the single NSCC runtime mirror.
Tracked files are authoritative. New source/config/workflow files are included;
known datasets, papers, generated outputs, and unrelated projects are omitted.

Environment overrides:
  Target defaults to a100; append h100 to select ASPIRE2A+.
  NSCC_RUNTIME_CODE_ROOT Single remote runtime checkout
  NSCC_GIT_URL           Personal repository used to initialize the checkout
  NSCC_GIT_BRANCH        Expected branch (default: hsic-spib)
EOF
}

TARGET="${NSCC_TARGET:-a100}"
DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        a100|h100|aspire2a|aspire2a+|aspire2p) TARGET="${arg}" ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ $# -le 2 ]] || { usage >&2; exit 2; }
LOCAL_ROOT="$(git rev-parse --show-toplevel)"
source "${LOCAL_ROOT}/nscc/target_config.sh"
nscc_load_target "${TARGET}"
SSH_HOST="${NSCC_SSH_HOST}"
REMOTE_ROOT="${NSCC_RUNTIME_CODE_ROOT}"
REMOTE_REPOSITORY="${NSCC_GIT_URL:-https://github.com/toil2sweet/State-Predictive-Information-Bottleneck.git}"
EXPECTED_BRANCH="${NSCC_GIT_BRANCH:-hsic-spib}"
REMOTE_ROOT="${REMOTE_ROOT%/}"
case "${REMOTE_ROOT}" in
    /home/users/nus/depeng/State-Predictive-Information-Bottleneck) ;;
    *)
        [[ "${NSCC_ALLOW_CUSTOM_RUNTIME_ROOT:-0}" == 1 ]] || {
            echo "Refusing --delete outside the canonical NSCC runtime path:" >&2
            echo "  ${REMOTE_ROOT}" >&2
            exit 2
        }
        ;;
esac

cd "${LOCAL_ROOT}"
CURRENT_BRANCH="$(git symbolic-ref --quiet --short HEAD || true)"
[[ "${CURRENT_BRANCH}" == "${EXPECTED_BRANCH}" ]] || {
    echo "Local checkout must be on ${EXPECTED_BRANCH}; current branch is ${CURRENT_BRANCH:-detached HEAD}." >&2
    exit 1
}

SYNC_TMP="$(mktemp -d "${TMPDIR:-/tmp}/spib-nscc-sync.XXXXXX")"
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
        .DS_Store|*/.DS_Store|*.npy|*.npz|*.h5|*.hdf5|*.csv|*.tsv|*.dat|\
        *.parquet|*.feather|*.pkl|*.pickle|*.nc|\
        *.xtc|*.dcd|*.trr|*.pdb|*.gro|*.top|*.tpr|\
        *.pt|*.pth|*.ckpt|*.log|*.out|\
        *.png|*.jpg|*.jpeg|*.gif|*.tif|*.tiff|*.mp4|*.mov|\
        *.zip|*.tar|*.tgz|*.gz|*.bz2|*.xz|*.sif|*.sqsh|*.pyc|\
        SPIB/*|fig/*|data/*|results/*|logs/*|wandb/*|.agents/*|\
        __pycache__/*|*/__pycache__/*|\
        HSIC-bottleneck/*|spib_msm/*|ts-dar/*)
            return 0
            ;;
    esac
    return 1
}

exclude_untracked_path() {
    case "$1" in
        CHANGELOG.md)
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
    echo "synced_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${STAGING_ROOT}/.nscc-source-info"

# Generate working-tree provenance against the actual local base commit. The
# runtime checkout deliberately keeps its own .git directory, whose HEAD may be
# older and must not be used to construct a recovery patch.
git status --short --untracked-files=no >"${STAGING_ROOT}/.nscc-source-status"
git diff --binary HEAD >"${STAGING_ROOT}/.nscc-source.patch"

(
    cd "${STAGING_ROOT}"
    find . -type f ! -name '.nscc-source-manifest.sha256' \
        -exec shasum -a 256 {} + | LC_ALL=C sort
) >"${STAGING_ROOT}/.nscc-source-manifest.sha256"

quote_arg() { printf '%q' "$1"; }
REMOTE_ROOT_Q="$(quote_arg "${REMOTE_ROOT}")"
REMOTE_GIT_Q="$(quote_arg "${REMOTE_ROOT}/.git")"
REMOTE_REPOSITORY_Q="$(quote_arg "${REMOTE_REPOSITORY}")"
EXPECTED_BRANCH_Q="$(quote_arg "${EXPECTED_BRANCH}")"

set +e
ssh "${SSH_HOST}" "test -d ${REMOTE_GIT_Q}"
REMOTE_GIT_STATUS=$?
set -e
if [[ "${REMOTE_GIT_STATUS}" -eq 255 ]]; then
    echo "Cannot authenticate to ${SSH_HOST}; NSCC runtime was not inspected or changed." >&2
    exit 1
fi
if [[ "${REMOTE_GIT_STATUS}" -ne 0 ]]; then
    [[ "${DRY_RUN}" == 0 ]] || {
        echo "Remote runtime checkout is missing; run without --dry-run once to initialize it." >&2
        exit 1
    }
    if ssh "${SSH_HOST}" "test -e ${REMOTE_ROOT_Q}"; then
        echo "Runtime path exists but is not a Git checkout; refusing to overwrite it:" >&2
        echo "  ${REMOTE_ROOT}" >&2
        exit 1
    fi
    ssh "${SSH_HOST}" \
        "git clone --origin personal --branch ${EXPECTED_BRANCH_Q} --single-branch ${REMOTE_REPOSITORY_Q} ${REMOTE_ROOT_Q}"
fi

REMOTE_BRANCH="$(ssh "${SSH_HOST}" "git -C ${REMOTE_ROOT_Q} symbolic-ref --quiet --short HEAD || true" | tail -1)"
[[ "${REMOTE_BRANCH}" == "${EXPECTED_BRANCH}" ]] || {
    echo "Remote runtime checkout must be on ${EXPECTED_BRANCH}; current branch is ${REMOTE_BRANCH:-detached HEAD}." >&2
    exit 1
}

SYNC_EXCLUDES=(
    --exclude=.git/
    --exclude=fig/
    --exclude=SPIB/
    --exclude=data/
    --exclude=results/
    --exclude=logs/
    --exclude=wandb/
)
RSYNC_OPTIONS=(-az --delete-delay --omit-dir-times --itemize-changes "${SYNC_EXCLUDES[@]}")
[[ "${DRY_RUN}" == 0 ]] || RSYNC_OPTIONS+=(--dry-run)
rsync "${RSYNC_OPTIONS[@]}" "${STAGING_ROOT}/" "${SSH_HOST}:${REMOTE_ROOT}/"

if [[ "${DRY_RUN}" == 1 ]]; then
    echo "dry_run=complete"
    echo "remote=${SSH_HOST}:${REMOTE_ROOT}"
    exit 0
fi

VERIFY_OUTPUT="$(rsync -aznci --delete-delay --omit-dir-times "${SYNC_EXCLUDES[@]}" \
    "${STAGING_ROOT}/" "${SSH_HOST}:${REMOTE_ROOT}/")"
if [[ -n "${VERIFY_OUTPUT}" ]]; then
    echo "NSCC runtime verification failed; remaining differences:" >&2
    echo "${VERIFY_OUTPUT}" >&2
    exit 1
fi

echo "sync=verified"
echo "target=${NSCC_TARGET}"
echo "remote=${SSH_HOST}:${REMOTE_ROOT}"
echo "branch=${CURRENT_BRANCH}"
echo "base_commit=${BASE_COMMIT}"
echo "code_version=${CODE_VERSION}"
echo "tracked_state=${TRACKED_STATE}"
