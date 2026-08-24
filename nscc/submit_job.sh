#!/usr/bin/env bash

# Freeze either the synced working tree or an exact pushed Git commit before
# qsub. The single runtime checkout is a control/mirror location, not a version.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  nscc/submit_job.sh --working [config.ini]
  nscc/submit_job.sh --commit COMMIT [config.ini]

Compatibility alias:
  --stable  Archive the fetched personal/hsic-spib tip

Environment overrides:
  NSCC_SOURCE_CODE_ROOT  Single runtime checkout (default: current Git root)
  NSCC_PROJECT_ROOT      Project data/results root
  NSCC_ENV_PREFIX        SPIB Conda environment
  NSCC_GIT_REMOTE        Version remote (default: personal)
  NSCC_GIT_BRANCH        Version branch (default: hsic-spib)
EOF
}

MODE="${1:-}"
REMOTE="${NSCC_GIT_REMOTE:-personal}"
BRANCH_NAME="${NSCC_GIT_BRANCH:-hsic-spib}"
REQUESTED_COMMIT=""

case "${MODE}" in
    --working)
        SOURCE_MODE=working
        CONFIG="${2:-examples/Four_Well_hsic_config.ini}"
        [[ $# -le 2 ]] || { usage >&2; exit 2; }
        ;;
    --commit)
        SOURCE_MODE=commit
        REQUESTED_COMMIT="${2:-}"
        CONFIG="${3:-examples/Four_Well_hsic_config.ini}"
        [[ $# -ge 2 && $# -le 3 ]] || { usage >&2; exit 2; }
        ;;
    --stable)
        SOURCE_MODE=commit
        CONFIG="${2:-examples/Four_Well_hsic_config.ini}"
        [[ $# -le 2 ]] || { usage >&2; exit 2; }
        ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

[[ "${CONFIG}" != /* && "${CONFIG}" != *,* && "${CONFIG}" != *$'\n'* && "${CONFIG}" != *[[:space:]]* ]] || {
    echo "CONFIG must be a repository-relative path without commas or whitespace." >&2
    exit 2
}
[[ "${CONFIG}" != .. && "${CONFIG}" != ../* && "${CONFIG}" != */../* && "${CONFIG}" != */.. ]] || {
    echo "CONFIG must not escape the source checkout." >&2
    exit 2
}

SOURCE_ROOT="${NSCC_SOURCE_CODE_ROOT:-$(git rev-parse --show-toplevel)}"
PROJECT_ROOT="${NSCC_PROJECT_ROOT:-/data/projects/11014454/depeng}"
ENV_PREFIX="${NSCC_ENV_PREFIX:-${PROJECT_ROOT}/envs/spib}"
PROJECT_ID="${NSCC_PROJECT_ID:-11014454}"
QUEUE="${NSCC_QUEUE:-normal}"
BATCH_SELECT="${NSCC_BATCH_SELECT:-1:ncpus=8:mem=32gb:ngpus=1}"
BATCH_WALLTIME="${NSCC_BATCH_WALLTIME:-04:00:00}"
MINIFORGE_MODULE="${NSCC_MINIFORGE_MODULE:-miniforge3/24.3.0}"
TARGET="${NSCC_TARGET:-h100}"
SOURCE_ROOT="${SOURCE_ROOT%/}"

QSUB_BIN="${NSCC_QSUB_BIN:-$(command -v qsub || true)}"
if [[ -z "${QSUB_BIN}" && -x /opt/pbs/bin/qsub ]]; then QSUB_BIN=/opt/pbs/bin/qsub; fi
[[ -n "${QSUB_BIN}" && -x "${QSUB_BIN}" ]] || { echo "Cannot find an executable qsub." >&2; exit 1; }

cd "${SOURCE_ROOT}"
[[ -d .git ]] || { echo "Runtime mirror is not a Git checkout: ${SOURCE_ROOT}" >&2; exit 1; }
[[ -x "${ENV_PREFIX}/bin/python" ]] || { echo "Missing SPIB environment: ${ENV_PREFIX}" >&2; exit 1; }
BRANCH="$(git symbolic-ref --quiet --short HEAD || true)"
[[ "${BRANCH}" == hsic-spib ]] || { echo "Runtime mirror must remain on hsic-spib; current=${BRANCH:-detached}." >&2; exit 1; }

CHECKOUT_HEAD="$(git rev-parse HEAD^{commit})"
BASE_COMMIT="${CHECKOUT_HEAD}"
CODE_VERSION=unversioned

if [[ "${SOURCE_MODE}" == working ]]; then
    [[ -f "${CONFIG}" ]] || { echo "Missing config: ${SOURCE_ROOT}/${CONFIG}" >&2; exit 1; }
    if [[ -f .nscc-source-info ]]; then
        SYNCED_BASE="$(sed -n 's/^base_commit=//p' .nscc-source-info | head -1)"
        SYNCED_VERSION="$(sed -n 's/^code_version=//p' .nscc-source-info | head -1)"
        [[ "${SYNCED_BASE}" =~ ^[0-9a-fA-F]{40}$ ]] && BASE_COMMIT="${SYNCED_BASE}"
        [[ -n "${SYNCED_VERSION}" ]] && CODE_VERSION="${SYNCED_VERSION}"
    fi
    if [[ "${CODE_VERSION}" == unversioned && -f VERSION ]]; then
        CODE_VERSION="$(tr -d '[:space:]' <VERSION)"
    fi
else
    git remote get-url "${REMOTE}" >/dev/null
    git fetch --tags "${REMOTE}" "+refs/heads/${BRANCH_NAME}:refs/remotes/${REMOTE}/${BRANCH_NAME}"
    if [[ "${MODE}" == --stable ]]; then
        REQUESTED_COMMIT="$(git rev-parse "refs/remotes/${REMOTE}/${BRANCH_NAME}^{commit}")"
    fi
    [[ "${REQUESTED_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]] || { echo "--commit requires a full 40-character SHA." >&2; exit 2; }
    BASE_COMMIT="$(git rev-parse "${REQUESTED_COMMIT}^{commit}")"
    git merge-base --is-ancestor "${BASE_COMMIT}" "refs/remotes/${REMOTE}/${BRANCH_NAME}" || {
        echo "Commit is not reachable from fetched ${REMOTE}/${BRANCH_NAME}: ${BASE_COMMIT}" >&2
        exit 1
    }
    git cat-file -e "${BASE_COMMIT}:${CONFIG}" 2>/dev/null || { echo "Config absent from commit: ${CONFIG}" >&2; exit 1; }
    CODE_VERSION="$(git show "${BASE_COMMIT}:VERSION" 2>/dev/null | tr -d '[:space:]' || true)"
    [[ -n "${CODE_VERSION}" ]] || CODE_VERSION=unversioned
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SNAPSHOT_ID="${STAMP}-${SOURCE_MODE}-${CODE_VERSION//[^0-9A-Za-z._-]/_}-${BASE_COMMIT:0:12}-$$"
SNAPSHOT_PARENT="${PROJECT_ROOT}/run_sources/spib"
SNAPSHOT_ROOT="${SNAPSHOT_PARENT}/${SNAPSHOT_ID}"

mkdir -p "${SNAPSHOT_PARENT}" "${PROJECT_ROOT}/logs/spib" "${PROJECT_ROOT}/results/spib"
[[ ! -e "${SNAPSHOT_ROOT}" ]] || { echo "Snapshot already exists: ${SNAPSHOT_ROOT}" >&2; exit 1; }
mkdir "${SNAPSHOT_ROOT}"

if [[ "${SOURCE_MODE}" == commit ]]; then
    git archive "${BASE_COMMIT}" | tar -x -C "${SNAPSHOT_ROOT}"
    : >"${SNAPSHOT_ROOT}/.nscc-source-status"
    : >"${SNAPSHOT_ROOT}/.nscc-source.patch"
else
    rsync -a --exclude=.git/ --exclude=SPIB/ --exclude=fig/ --exclude=__pycache__/ --exclude='*.pyc' \
        "${SOURCE_ROOT}/" "${SNAPSHOT_ROOT}/"
    [[ -f "${SNAPSHOT_ROOT}/.nscc-source-status" ]] || \
        git status --short --untracked-files=no >"${SNAPSHOT_ROOT}/.nscc-source-status"
    [[ -f "${SNAPSHOT_ROOT}/.nscc-source.patch" ]] || \
        git diff --binary HEAD >"${SNAPSHOT_ROOT}/.nscc-source.patch"
fi

# Use the current execution harness even for an older source commit. It is kept
# separate from the archived scientific source and recorded in the manifest.
mkdir -p "${SNAPSHOT_ROOT}/.nscc-runner"
cp -- "${SOURCE_ROOT}/nscc/run_spib_advanced.pbs" "${SNAPSHOT_ROOT}/.nscc-runner/run_spib_advanced.pbs"

{
    echo "source_mode=${SOURCE_MODE}"
    echo "source_checkout=${SOURCE_ROOT}"
    echo "branch=${BRANCH}"
    echo "runtime_git_head=${CHECKOUT_HEAD}"
    echo "base_commit=${BASE_COMMIT}"
    echo "code_version=${CODE_VERSION}"
    echo "snapshot=${SNAPSHOT_ROOT}"
    echo "created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${SNAPSHOT_ROOT}/.nscc-run-source-info"

(
    cd "${SNAPSHOT_ROOT}"
    find . -type f ! -name '.nscc-source-manifest.sha256' -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum
) >"${SNAPSHOT_ROOT}/.nscc-source-manifest.sha256"

# PBS copies the submission script at qsub time, so submitting from the short
# runtime path is immutable for this job. Keep the recorded harness copy in the
# source snapshot, but avoid asking the scheduler to stage a deeply nested
# project-filesystem script path.
QSUB_SCRIPT="${SOURCE_ROOT}/nscc/run_spib_advanced.pbs"
[[ -f "${QSUB_SCRIPT}" ]] || { echo "Missing PBS harness: ${QSUB_SCRIPT}" >&2; exit 1; }

JOB_ID="$("${QSUB_BIN}" \
    -P "${PROJECT_ID}" -q "${QUEUE}" -l "select=${BATCH_SELECT}" -l "walltime=${BATCH_WALLTIME}" \
    -v "CONFIG=${CONFIG},NSCC_TARGET=${TARGET},NSCC_CODE_ROOT=${SNAPSHOT_ROOT},NSCC_PROJECT_ROOT=${PROJECT_ROOT},NSCC_ENV_PREFIX=${ENV_PREFIX},NSCC_MINIFORGE_MODULE=${MINIFORGE_MODULE},SPIB_GIT_COMMIT=${BASE_COMMIT},SPIB_CODE_VERSION=${CODE_VERSION},SPIB_SOURCE_MODE=${SOURCE_MODE},SPIB_PBS_SCRIPT_SOURCE=${SNAPSHOT_ROOT}/.nscc-runner/run_spib_advanced.pbs" \
    "${QSUB_SCRIPT}")"

echo "job=${JOB_ID}"
echo "target=${TARGET}"
echo "mode=${SOURCE_MODE}"
echo "version=${CODE_VERSION}"
echo "commit=${BASE_COMMIT}"
echo "config=${CONFIG}"
echo "snapshot=${SNAPSHOT_ROOT}"
echo "log=${PROJECT_ROOT}/logs/spib/${JOB_ID}.log"
echo "output=${PROJECT_ROOT}/results/spib/${JOB_ID}"
