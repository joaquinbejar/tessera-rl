// SPDX-License-Identifier: MIT OR Apache-2.0

//! Golden suites (ADR-0009, ADR-0024 layer 2) and cross-thread runs (SPEC
//! 8.5) for Connect Four and Leduc hold'em: every scenario matches its committed fixture in
//! Inline mode and with 1, 2, 7 and 64 workers. The harness is the one of
//! the classic crate.

#![expect(clippy::panic, reason = "test setup: a failure is the assertion")]

#[path = "../../tessera-rl-classic/tests/support/mod.rs"]
mod support;

use support::{Scenario, check, run};
use tessera_rl_engine::{AutoReset, EnvPool, Masked};
use tessera_rl_games::{Connect4, Connect4Config, Leduc, LeducConfig};

impl support::GoldenEnv for Connect4 {
    const ENV_ID: &'static str = tessera_rl_games::connect4::ENV_ID;
    const ENV_VERSION: u32 = tessera_rl_games::connect4::ENV_VERSION;
}

const WORKERS: [Option<usize>; 5] = [None, Some(1), Some(2), Some(7), Some(64)];

fn connect4(name: &'static str, auto_reset: AutoReset) -> Scenario<Connect4Config> {
    Scenario {
        name,
        cfg: Connect4Config {},
        seed: 1_112_026,
        num_envs: 16,
        chunk: 4,
        auto_reset,
        // Above the 42 plies of a game: the limit never truncates.
        max_episode_steps: 64,
        steps: 300,
        reseed_at: 150,
        reseed_ids: &[1, 12],
        reseed_seed: 4_444,
    }
}

fn check_connect4(s: &Scenario<Connect4Config>) {
    let build = |cfg, options| {
        EnvPool::<Connect4, Masked>::with_masks(cfg, options).unwrap_or_else(|e| panic!("{e}"))
    };
    for workers in WORKERS {
        check(s.name, &run::<Connect4, _>(s, build, workers));
    }
}

#[test]
fn test_golden_connect4_next_step_matches_fixture_on_every_mode() {
    check_connect4(&connect4("connect4-v1-next-step", AutoReset::NextStep));
}

#[test]
fn test_golden_connect4_same_step_matches_fixture_on_every_mode() {
    check_connect4(&connect4("connect4-v1-same-step", AutoReset::SameStep));
}

impl support::GoldenEnv for Leduc {
    const ENV_ID: &'static str = tessera_rl_games::leduc::ENV_ID;
    const ENV_VERSION: u32 = tessera_rl_games::leduc::ENV_VERSION;
}

fn leduc(name: &'static str, auto_reset: AutoReset) -> Scenario<LeducConfig> {
    Scenario {
        name,
        cfg: LeducConfig {},
        seed: 2_112_026,
        num_envs: 16,
        chunk: 4,
        auto_reset,
        // Above the 8 actions of a hand: the limit never truncates.
        max_episode_steps: 16,
        steps: 300,
        reseed_at: 150,
        reseed_ids: &[1, 12],
        reseed_seed: 5_555,
    }
}

fn check_leduc(s: &Scenario<LeducConfig>) {
    let build = |cfg, options| {
        EnvPool::<Leduc, Masked>::with_masks(cfg, options).unwrap_or_else(|e| panic!("{e}"))
    };
    for workers in WORKERS {
        check(s.name, &run::<Leduc, _>(s, build, workers));
    }
}

#[test]
fn test_golden_leduc_next_step_matches_fixture_on_every_mode() {
    check_leduc(&leduc("leduc-v1-next-step", AutoReset::NextStep));
}

#[test]
fn test_golden_leduc_same_step_matches_fixture_on_every_mode() {
    check_leduc(&leduc("leduc-v1-same-step", AutoReset::SameStep));
}
