#!/usr/bin/env bash

# Prepare VERSION and CHANGELOG.md for a reviewed checkpoint. This script does
# not stage, commit, tag, or push; those actions remain explicit user choices.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/prepare_version.sh VERSION SUMMARY

Examples:
  scripts/prepare_version.sh 1.0.1 "Fix adaptive HSIC bandwidth selection"
  scripts/prepare_version.sh 1.1.0 "Add transition-state diagnostics"

VERSION must be SemVer without a leading v. Patch numbers do not roll over:
1.0.9 is followed by 1.0.10 unless the change itself warrants 1.1.0.

This command updates VERSION and CHANGELOG.md only. It never creates a Git
commit or tag.
EOF
}

case "${1:-}" in -h|--help) usage; exit 0 ;; esac
VERSION_VALUE="${1:-}"
SUMMARY="${2:-}"
[[ $# -eq 2 ]] || { usage >&2; exit 2; }

SEMVER_RE='^(0|[1-9][0-9]*)[.](0|[1-9][0-9]*)[.](0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?([+][0-9A-Za-z.-]+)?$'
[[ "${VERSION_VALUE}" =~ ${SEMVER_RE} ]] || {
    echo "Invalid semantic version: ${VERSION_VALUE}" >&2
    exit 2
}
[[ -n "${SUMMARY//[[:space:]]/}" && "${SUMMARY}" != *$'\n'* ]] || {
    echo "SUMMARY must be one non-empty line." >&2
    exit 2
}

ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}"
[[ "$(git branch --show-current)" == hsic-spib ]] || {
    echo "Version checkpoints must be prepared on hsic-spib." >&2
    exit 1
}
[[ -f CHANGELOG.md && -f VERSION ]] || {
    echo "Missing VERSION or CHANGELOG.md." >&2
    exit 1
}
if git rev-parse -q --verify "refs/tags/HSIC-SPIB_V${VERSION_VALUE}" >/dev/null; then
    echo "Milestone tag already exists: HSIC-SPIB_V${VERSION_VALUE}" >&2
    exit 1
fi

TODAY="$(date +%Y-%m-%d)"
TMP_FILE="$(mktemp "${TMPDIR:-/tmp}/spib-changelog.XXXXXX")"
cleanup() { rm -f -- "${TMP_FILE}"; }
trap cleanup EXIT

awk -v version="${VERSION_VALUE}" -v today="${TODAY}" -v summary="${SUMMARY}" '
    { print }
    /^## Unreleased$/ && !inserted {
        print ""
        print "## " version " - " today
        print ""
        print "- " summary "."
        inserted=1
    }
' CHANGELOG.md >"${TMP_FILE}"

printf '%s\n' "${VERSION_VALUE}" >VERSION
cp -- "${TMP_FILE}" CHANGELOG.md

echo "version=${VERSION_VALUE}"
echo "summary=${SUMMARY}"
echo "prepared=VERSION,CHANGELOG.md"
echo "git_actions=none"
