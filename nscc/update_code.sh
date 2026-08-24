#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${NSCC_RUNTIME_CODE_ROOT:-${NSCC_CODE_ROOT:-/home/users/nus/depeng/State-Predictive-Information-Bottleneck}}"
REMOTE="${NSCC_GIT_REMOTE:-personal}"
BRANCH="${NSCC_GIT_BRANCH:-hsic-spib}"
REMOTE_REF="refs/remotes/${REMOTE}/${BRANCH}"

cd "${CODE_ROOT}"
git remote get-url "${REMOTE}" >/dev/null
git fetch --tags "${REMOTE}" \
    "+refs/heads/${BRANCH}:${REMOTE_REF}"

# The runtime worktree may contain synchronized uncommitted files. Fetch exact
# version objects without checking out or merging over that working tree.
echo "fetched_ref=${REMOTE}/${BRANCH}"
echo "fetched_commit=$(git rev-parse "${REMOTE_REF}^{commit}")"
git log -1 --oneline --decorate "${REMOTE_REF}"
