#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${NSCC_CODE_ROOT:-/home/users/nus/depeng/State-Predictive-Information-Bottleneck}"
REMOTE="${NSCC_GIT_REMOTE:-personal}"
BRANCH="${NSCC_GIT_BRANCH:-main}"

cd "${CODE_ROOT}"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "NSCC checkout has local changes; refusing to pull over them." >&2
    git status --short
    exit 1
fi

git fetch "${REMOTE}" "${BRANCH}"
git merge --ff-only "${REMOTE}/${BRANCH}"
git log -1 --oneline --decorate
