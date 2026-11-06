#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# A changed TESSERA_BUILD_ID must rebuild the engine and reach it, and a repeated
# one must not rebuild (ADR-0011): a stale id is a wrong identity with no error.
set -euo pipefail

build() {
    # Plain text whatever CARGO_TERM_COLOR says: colour codes would hide the
    # "Compiling" line, failing the first check and passing the second vacuously.
    # A build that printed no "Finished" proves nothing either way.
    local out
    out=$(TESSERA_BUILD_ID="$1" cargo build -p tessera-rl-engine --locked --color never 2>&1)
    grep -q 'Finished' <<<"$out" || {
        echo "the build printed no Finished line:"
        echo "$out"
        exit 1
    }
    echo "$out"
}

check() {
    TESSERA_BUILD_ID="$1" EXPECT_BUILD_ID="$1" cargo test -p tessera-rl-engine --locked \
        --test persist_build_id -- --ignored >/dev/null 2>&1 || {
        echo "the engine does not hold id $1"
        exit 1
    }
}

a="check-a-$$"
b="check-b-$$"
build "$a" >/dev/null
first=$(build "$b")
if ! grep -q 'Compiling tessera-rl-engine' <<<"$first"; then
    echo "a changed id did not rebuild the engine"
    exit 1
fi
second=$(build "$b")
if grep -q 'Compiling tessera-rl-engine' <<<"$second"; then
    echo "a repeated id rebuilt the engine"
    exit 1
fi
check "$a"
check "$b"
echo "build id tracking holds"
