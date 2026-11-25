#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# The lock-subset check of cargo tessera's end-to-end test (#351, #633):
#
#   scripts/lock_subset.sh REPO_LOCK SCAFFOLD_LOCK CRATE
#
# Fails (exit 1) and lists the packages when SCAFFOLD_LOCK has a package and
# version that REPO_LOCK lacks, the scaffold crate CRATE itself excepted.
#
# `sort` and `comm` run under one collation, LC_ALL=C: under a UTF-8 locale
# uutils `sort` orders bytewise while uutils `comm` checks order with the
# locale's collation, so `comm` warned that its sorted input was unsorted
# and could pair lines wrongly, a false negative (#633). Each input is also
# checked with `sort -c` before `comm`, so unsorted input fails instead of
# warning. `sort -c` rather than `comm --check-order`, which BSD `comm`
# (macOS) does not have.

set -euo pipefail
export LC_ALL=C

[ "$#" -eq 3 ] || { echo "usage: lock_subset.sh REPO_LOCK SCAFFOLD_LOCK CRATE" >&2; exit 2; }
repo_lock="$1"
scaffold_lock="$2"
crate="$3"

pairs() { awk '/^name = /{n=$3} /^version = /{print n, $3}' "$1" | sort -u; }
repo_pairs="$(pairs "$repo_lock")"
scaffold_pairs="$(pairs "$scaffold_lock")"
for list in "$repo_pairs" "$scaffold_pairs"; do
    printf '%s\n' "$list" | sort -c -u || { echo "lock_subset: a package list is not sorted under LC_ALL=C" >&2; exit 1; }
done
listed="$(comm -13 <(printf '%s\n' "$repo_pairs") <(printf '%s\n' "$scaffold_pairs"))"
extra="$(printf '%s\n' "$listed" | grep -v -e "^\"$crate\" " -e '^$' || true)"
if [ -n "$extra" ]; then
    echo "the scaffold's lock has packages the repository's lock does not:" >&2
    echo "$extra" >&2
    exit 1
fi
