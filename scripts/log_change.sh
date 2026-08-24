#!/usr/bin/env bash

# Add one logical development change to CHANGELOG.md without creating a Git
# checkpoint. Failed experiments and tiny intermediate edits should not be
# logged separately.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/log_change.sh SUMMARY

Add one concise bullet under CHANGELOG.md "Unreleased". This command does not
change VERSION and never stages, commits, tags, or pushes.
EOF
}

case "${1:-}" in -h|--help) usage; exit 0 ;; esac
SUMMARY="${1:-}"
[[ $# -eq 1 ]] || { usage >&2; exit 2; }
[[ -n "${SUMMARY//[[:space:]]/}" && "${SUMMARY}" != *$'\n'* ]] || {
    echo "SUMMARY must be one non-empty line." >&2
    exit 2
}

ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}"
[[ "$(git branch --show-current)" == hsic-spib ]] || {
    echo "Changes must be logged on hsic-spib." >&2
    exit 1
}
[[ -f CHANGELOG.md ]] || { echo "Missing CHANGELOG.md." >&2; exit 1; }

TMP_FILE="$(mktemp "${TMPDIR:-/tmp}/spib-changelog.XXXXXX")"
cleanup() { rm -f -- "${TMP_FILE}"; }
trap cleanup EXIT

awk -v summary="${SUMMARY}" '
    /^## Unreleased$/ && !inserted {
        print
        print ""
        print "- " summary
        inserted=1
        skip_blank=1
        next
    }
    skip_blank && /^$/ {
        skip_blank=0
        next
    }
    { print }
    END {
        if (!inserted) exit 3
    }
' CHANGELOG.md >"${TMP_FILE}" || {
    echo "Could not find the Unreleased section in CHANGELOG.md." >&2
    exit 1
}

cp -- "${TMP_FILE}" CHANGELOG.md
echo "logged=${SUMMARY}"
echo "version=$(tr -d '[:space:]' <VERSION 2>/dev/null || echo unversioned)"
echo "git_actions=none"
