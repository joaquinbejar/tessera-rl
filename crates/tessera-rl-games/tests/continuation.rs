// SPDX-License-Identifier: MIT OR Apache-2.0

//! Continuation tests (SPEC 9.1, normative; ADR-0010) for Connect Four,
//! with the harness of the classic crate. Connect Four draws no
//! randomness, so `fork_branched` replays the same game: the harness
//! checks that it does not diverge.

#[path = "../../tessera-rl-classic/tests/support/mod.rs"]
mod support;

use support::continuation::Case;
use tessera_rl_engine::{AutoReset, EnvPool, Masked};
use tessera_rl_games::{Connect4, Connect4Config};

#[expect(clippy::unwrap_used, reason = "test setup: a failure is the assertion")]
fn connect4(auto_reset: AutoReset) -> Case<Connect4, Masked> {
    Case {
        cfg: Connect4Config {},
        build: |cfg, options| EnvPool::with_masks(cfg, options).unwrap(),
        auto_reset,
        limit: 64,
        deaths: false,
        branches: false,
    }
}

#[test]
fn test_continuation_connect4_next_step_restores_identically_everywhere() {
    connect4(AutoReset::NextStep).check_all(50);
}

#[test]
fn test_continuation_connect4_same_step_restores_identically_everywhere() {
    connect4(AutoReset::SameStep).check_all(50);
}
