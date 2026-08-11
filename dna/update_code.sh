#!/usr/bin/env bash

set -euo pipefail

CODE_ROOT="${DNA_CODE_ROOT:-/mnt/rna01/lidp/State-Predictive-Information-Bottleneck}"
REMOTE="${DNA_GIT_REMOTE:-personal}"
BRANCH="${DNA_GIT_BRANCH:-hsic-spib}"
REMOTE_REF="refs/remotes/${REMOTE}/${BRANCH}"

cd "${CODE_ROOT}"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "DNA checkout has tracked local changes; refusing to update it." >&2
    git status --short --untracked-files=no
    exit 1
fi

git remote get-url "${REMOTE}" >/dev/null
git fetch "${REMOTE}" \
    "+refs/heads/${BRANCH}:${REMOTE_REF}"

current_branch="$(git symbolic-ref --quiet --short HEAD || true)"
if [[ "${current_branch}" != "${BRANCH}" ]]; then
    echo "Switching DNA checkout from ${current_branch:-detached HEAD} to ${BRANCH}."
    if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
        git checkout "${BRANCH}"
    else
        git checkout --track -b "${BRANCH}" "${REMOTE}/${BRANCH}"
    fi
fi

git merge --ff-only "${REMOTE}/${BRANCH}"
git rev-parse HEAD
