#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# A changed TESSERA_BUILD_ID must rebuild the engine and reach it, and a repeated
# one must not rebuild (ADR-0011): a stale id is a wrong identity with no error.
set -euo pipefail

build() {
    TESSERA_BUILD_ID="$1" cargo build -p tessera-rl-engine --locked 2>&1
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
if ! build "$b" | grep -q 'Compiling tessera-rl-engine'; then
    echo "a changed id did not rebuild the engine"
    exit 1
fi
if build "$b" | grep -q 'Compiling tessera-rl-engine'; then
    echo "a repeated id rebuilt the engine"
    exit 1
fi
check "$a"
check "$b"
echo "build id tracking holds"
