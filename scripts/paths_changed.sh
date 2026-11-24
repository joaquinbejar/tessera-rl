#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Whether any file under the given path prefixes changed between two commits.
#
#   scripts/paths_changed.sh <base sha> <head sha> <prefix>...
#
# Prints `changed=true` or `changed=false` (for $GITHUB_OUTPUT) and exits 0.
# A missing or all-zero base (a new branch, a first push), or a base not in
# the clone, counts as changed: the guarded steps then run, never skip
# silently. Used by ci.yml's burn-check job (#367 follow-up).
set -euo pipefail
base="${1-}"
head="${2:?head sha}"
shift 2 || true
[ "$#" -gt 0 ] || { echo "paths_changed: no prefixes" >&2; exit 2; }
if [ -z "$base" ] || [ -z "${base//0/}" ] || ! git cat-file -e "$base^{commit}" 2>/dev/null; then
    echo "changed=true"
    exit 0
fi
while IFS= read -r f; do
    for p in "$@"; do
        case "$f" in "$p"*) echo "changed=true"; exit 0 ;; esac
    done
done < <(git diff --name-only "$base" "$head")
echo "changed=false"
