// SPDX-License-Identifier: MIT OR Apache-2.0

//! Golden suites (ADR-0009, ADR-0024 layer 2) and cross-thread runs (SPEC
//! 8.5): every scenario matches its committed fixture in Inline mode and
//! with 1, 2, 7 and 64 workers.

#![expect(clippy::panic, reason = "test setup: a failure is the assertion")]

mod support;

use support::{Scenario, check, compare, run};
use tessera_rl_classic::cartpole::MAX_EPISODE_STEPS;
use tessera_rl_classic::{CartPole, CartPoleConfig, Tag, TagConfig};
use tessera_rl_engine::{AutoReset, EnvPool, Masked};

const WORKERS: [Option<usize>; 5] = [None, Some(1), Some(2), Some(7), Some(64)];

fn cartpole(name: &'static str, auto_reset: AutoReset) -> Scenario<CartPoleConfig> {
    Scenario {
        name,
        cfg: CartPoleConfig::default(),
        seed: 20_261_031,
        num_envs: 16,
        chunk: 4,
        auto_reset,
        max_episode_steps: MAX_EPISODE_STEPS,
        steps: 600,
        reseed_at: 300,
        reseed_ids: &[3, 7],
        reseed_seed: 999,
    }
}

fn tag(name: &'static str, cfg: TagConfig, auto_reset: AutoReset) -> Scenario<TagConfig> {
    Scenario {
        name,
        cfg,
        seed: 31_102_026,
        num_envs: 16,
        chunk: 4,
        auto_reset,
        max_episode_steps: 200,
        steps: 300,
        reseed_at: 150,
        reseed_ids: &[0, 9],
        reseed_seed: 4_242,
    }
}

fn crowded() -> TagConfig {
    TagConfig {
        pursuers: 4,
        evaders: 4,
        slip: 300,
        ..TagConfig::default()
    }
}

fn check_cartpole(s: &Scenario<CartPoleConfig>) {
    let build =
        |cfg, options| EnvPool::<CartPole>::new(cfg, options).unwrap_or_else(|e| panic!("{e}"));
    for workers in WORKERS {
        check(s.name, &run::<CartPole, _>(s, build, workers));
    }
}

fn check_tag(s: &Scenario<TagConfig>) {
    let build = |cfg, options| {
        EnvPool::<Tag, Masked>::with_masks(cfg, options).unwrap_or_else(|e| panic!("{e}"))
    };
    for workers in WORKERS {
        check(s.name, &run::<Tag, _>(s, build, workers));
    }
}

#[test]
fn test_golden_cartpole_next_step_matches_fixture_on_every_mode() {
    check_cartpole(&cartpole("cartpole-v1-next-step", AutoReset::NextStep));
}

#[test]
fn test_golden_cartpole_same_step_matches_fixture_on_every_mode() {
    check_cartpole(&cartpole("cartpole-v1-same-step", AutoReset::SameStep));
}

#[test]
fn test_golden_tag_default_next_step_matches_fixture_on_every_mode() {
    check_tag(&tag(
        "tag-v1-default-next-step",
        TagConfig::default(),
        AutoReset::NextStep,
    ));
}

#[test]
fn test_golden_tag_default_same_step_matches_fixture_on_every_mode() {
    check_tag(&tag(
        "tag-v1-default-same-step",
        TagConfig::default(),
        AutoReset::SameStep,
    ));
}

#[test]
fn test_golden_tag_eight_agents_slip_next_step_matches_fixture_on_every_mode() {
    check_tag(&tag(
        "tag-v1-crowded-next-step",
        crowded(),
        AutoReset::NextStep,
    ));
}

#[test]
fn test_golden_tag_eight_agents_slip_same_step_matches_fixture_on_every_mode() {
    check_tag(&tag(
        "tag-v1-crowded-same-step",
        crowded(),
        AutoReset::SameStep,
    ));
}

/// The comparison detects a one-field change and a version change.
#[test]
fn test_golden_check_rejects_a_changed_digest_and_version() {
    let s = cartpole("cartpole-v1-next-step", AutoReset::NextStep);
    let build =
        |cfg, options| EnvPool::<CartPole>::new(cfg, options).unwrap_or_else(|e| panic!("{e}"));
    let lines = run::<CartPole, _>(&s, build, None);
    let mut changed = lines.clone();
    let k = changed.len() / 2;
    changed[k] = changed[k].replacen(' ', " 0", 1);
    assert!(std::panic::catch_unwind(|| compare(s.name, &changed)).is_err());
    let bumped: Vec<String> = lines
        .iter()
        .map(|l| l.replace("env_version=1", "env_version=2"))
        .collect();
    assert!(std::panic::catch_unwind(|| compare(s.name, &bumped)).is_err());
}
