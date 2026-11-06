// SPDX-License-Identifier: MIT OR Apache-2.0

//! Golden suites (ADR-0009, ADR-0024 layer 2) and cross-thread runs (SPEC
//! 8.5): every scenario matches its committed fixture in Inline mode and
//! with 1, 2, 7 and 64 workers.

#![expect(clippy::panic, reason = "test setup: a failure is the assertion")]

mod support;

use support::{
    Bless, BlessReason, Scenario, bless_decision, bless_into, check, committed, compare_text, run,
};
use tessera_rl_classic::cartpole::MAX_EPISODE_STEPS;
use tessera_rl_classic::{CartPole, CartPoleConfig, Grid, GridConfig, Tag, TagConfig};
use tessera_rl_engine::{AutoReset, EnvPool, Masked, NoMasks};

impl support::GoldenEnv for CartPole {
    const ENV_ID: &'static str = tessera_rl_classic::cartpole::ENV_ID;
    const ENV_VERSION: u32 = tessera_rl_classic::cartpole::ENV_VERSION;
}

impl support::GoldenEnv for Tag {
    const ENV_ID: &'static str = tessera_rl_classic::tag::ENV_ID;
    const ENV_VERSION: u32 = tessera_rl_classic::tag::ENV_VERSION;
}

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

fn cartpole_lines() -> Vec<String> {
    let s = cartpole("cartpole-v1-next-step", AutoReset::NextStep);
    let build =
        |cfg, options| EnvPool::<CartPole>::new(cfg, options).unwrap_or_else(|e| panic!("{e}"));
    run::<CartPole, _>(&s, build, None)
}

fn text(lines: &[String]) -> String {
    lines.join("\n") + "\n"
}

fn bump(lines: &[String], key: &str, to: &str) -> Vec<String> {
    lines
        .iter()
        .map(
            |l| match l.strip_prefix(key).and_then(|r| r.strip_prefix('=')) {
                Some(_) => format!("{key}={to}"),
                None => l.clone(),
            },
        )
        .collect()
}

/// A changed digest: line `k`, one field altered.
fn with_changed_digest(lines: &[String]) -> Vec<String> {
    let mut changed = lines.to_vec();
    let k = changed.len() / 2;
    if let Some(line) = changed.get_mut(k) {
        *line = line.replacen(' ', " 0", 1);
    }
    changed
}

/// The comparison (pure, in memory) detects a one-field change and a
/// version change; it never touches `tests/golden/`.
#[test]
fn test_golden_compare_rejects_a_changed_digest_and_version() {
    let lines = cartpole_lines();
    let old = text(&lines);
    assert_eq!(compare_text("t", &old, &lines), Ok(()));
    let err = compare_text("t", &old, &with_changed_digest(&lines)).unwrap_err();
    assert!(err.contains("first divergence"), "{err}");
    assert!(compare_text("t", &old, &bump(&lines, "env_version", "2")).is_err());
}

/// The bless policy (ADR-0008, ADR-0023).
#[test]
fn test_golden_bless_policy_needs_an_env_version_bump() {
    let lines = cartpole_lines();
    let old = text(&lines);
    let changed = with_changed_digest(&lines);
    // Unchanged output: nothing is written.
    assert_eq!(bless_decision(Some(&old), &old, None), Ok(Bless::Unchanged));
    // A new fixture is written.
    assert_eq!(bless_decision(None, &old, None), Ok(Bless::Write));
    // A changed digest alone, or with only MATH_VERSION bumped: refused.
    assert!(bless_decision(Some(&old), &text(&changed), None).is_err());
    let math = bump(&changed, "math_version", "2");
    assert!(bless_decision(Some(&old), &text(&math), None).is_err());
    assert!(bless_decision(Some(&old), &text(&math), Some(BlessReason::Format)).is_err());
    // An env_version bump authorizes it.
    let env = bump(&changed, "env_version", "2");
    assert_eq!(
        bless_decision(Some(&old), &text(&env), None),
        Ok(Bless::Write)
    );
    // Harness migrations need the matching reason.
    let format = bump(&changed, "format", "99");
    assert!(bless_decision(Some(&old), &text(&format), None).is_err());
    assert!(bless_decision(Some(&old), &text(&format), Some(BlessReason::Generator)).is_err());
    assert_eq!(
        bless_decision(Some(&old), &text(&format), Some(BlessReason::Format)),
        Ok(Bless::Write)
    );
    let generator = bump(&changed, "actions", "other-v2");
    assert_eq!(
        bless_decision(Some(&old), &text(&generator), Some(BlessReason::Generator)),
        Ok(Bless::Write)
    );
}

/// Blessing a temporary fixture: an unchanged output leaves the file
/// untouched, and a refused change leaves it intact.
#[test]
fn test_golden_bless_leaves_unchanged_and_refused_fixtures_intact() {
    let lines = cartpole_lines();
    let dir = std::env::temp_dir().join(format!("tessera-golden-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
    let file = dir.join("fixture.golden");
    assert_eq!(bless_into(&file, &lines, None), Ok(Bless::Write));
    let written = std::fs::metadata(&file).and_then(|m| m.modified()).ok();
    assert_eq!(bless_into(&file, &lines, None), Ok(Bless::Unchanged));
    assert_eq!(
        std::fs::metadata(&file).and_then(|m| m.modified()).ok(),
        written
    );
    let math = bump(&with_changed_digest(&lines), "math_version", "2");
    assert!(bless_into(&file, &math, None).is_err());
    assert_eq!(std::fs::read_to_string(&file).ok(), Some(text(&lines)));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The committed fixture is exactly what the code produces, so a bless
/// run would leave it untouched.
#[test]
fn test_golden_bless_of_current_output_is_unchanged() {
    let lines = cartpole_lines();
    assert_eq!(
        bless_decision(
            Some(&committed("cartpole-v1-next-step")),
            &text(&lines),
            None
        ),
        Ok(Bless::Unchanged)
    );
}

impl support::GoldenEnv for Grid {
    const ENV_ID: &'static str = tessera_rl_classic::grid::ENV_ID;
    const ENV_VERSION: u32 = tessera_rl_classic::grid::ENV_VERSION;
}

fn grid(name: &'static str, auto_reset: AutoReset) -> Scenario<GridConfig> {
    Scenario {
        name,
        cfg: GridConfig {
            size: 6,
            max_steps: 40,
        },
        seed: 5_112_026,
        num_envs: 16,
        chunk: 4,
        auto_reset,
        // Above `max_steps`: the world truncates by itself.
        max_episode_steps: 64,
        steps: 300,
        reseed_at: 150,
        reseed_ids: &[1, 12],
        reseed_seed: 8_888,
    }
}

fn check_grid(s: &Scenario<GridConfig>) {
    let build = |cfg, options| {
        EnvPool::<Grid, NoMasks>::new(cfg, options).unwrap_or_else(|e| panic!("{e}"))
    };
    for workers in WORKERS {
        check(s.name, &run::<Grid, _>(s, build, workers));
    }
}

#[test]
fn test_golden_grid_next_step_matches_fixture_on_every_mode() {
    check_grid(&grid("grid-v1-next-step", AutoReset::NextStep));
}

#[test]
fn test_golden_grid_same_step_matches_fixture_on_every_mode() {
    check_grid(&grid("grid-v1-same-step", AutoReset::SameStep));
}
