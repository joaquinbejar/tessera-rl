#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# The ADR-0037 floating-point boundary tests under a wheel's build settings
# (docs/reviews/pgo-wheels-design.md r5, section 7 item 6). They only detect
# compiler motion in optimized builds.
#
#   scripts/wheel_num_tests.sh                     release (W0)
#   WHEEL_CARGO_PROFILE=wheel scripts/...          the wheel profile (W1)
#   WHEEL_CARGO_PROFILE=wheel WHEEL_PROFILE_USE=/abs/merged.profdata scripts/...  (W2F)
#
# With a profile this is weaker evidence than the Python suite on the wheel: the
# profile was trained on the cdylib and matches these binaries poorly.
set -euo pipefail

args=(--release)
if [ -n "${WHEEL_CARGO_PROFILE:-}" ]; then
    args=(--profile "$WHEEL_CARGO_PROFILE")
fi
if [ -n "${WHEEL_PROFILE_USE:-}" ]; then
    case "$WHEEL_PROFILE_USE" in
        /*) ;;
        *) echo "wheel_num_tests: WHEEL_PROFILE_USE must be absolute" >&2; exit 2 ;;
    esac
    [ -n "${WHEEL_CARGO_PROFILE:-}" ] || { echo "wheel_num_tests: a profile needs WHEEL_CARGO_PROFILE" >&2; exit 2; }
    export RUSTFLAGS="-Cprofile-use=$WHEEL_PROFILE_USE"
fi
exec nice -n 10 cargo test -p tessera-rl-num "${args[@]}" --all-features --locked
