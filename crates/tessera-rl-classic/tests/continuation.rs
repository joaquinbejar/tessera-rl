// SPDX-License-Identifier: MIT OR Apache-2.0

//! Continuation tests (SPEC 9.1, normative; ADR-0010): snapshot one
//! instance at step t, run K steps, then restore it into the same slot, a
//! different slot and a different pool with a different seed (by `restore`
//! and by `import`), replay the same actions, and require identical
//! observations, rewards, masks, agent and episode flags, generations,
//! episode statistics, headers and state hashes. `fork` continues the same
//! way; `fork_branched` diverges. The harness is `support/continuation.rs`.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers: a failure is the assertion"
)]

mod support;

use support::continuation::Case;
use tessera_rl_classic::cartpole::MAX_EPISODE_STEPS;
use tessera_rl_classic::{
    CartPole, CartPoleConfig, Grid, GridConfig, KeyDoor, KeyDoorConfig, Snake, SnakeConfig, Tag,
    TagConfig,
};
use tessera_rl_engine::{AutoReset, EnvPool, Masked, NoMasks};

fn cartpole(auto_reset: AutoReset) -> Case<CartPole, NoMasks> {
    Case {
        cfg: CartPoleConfig::default(),
        build: |cfg, options| EnvPool::new(cfg, options).unwrap(),
        auto_reset,
        limit: MAX_EPISODE_STEPS,
        deaths: false,
        branches: true,
    }
}

fn tag(auto_reset: AutoReset) -> Case<Tag, Masked> {
    Case {
        cfg: TagConfig {
            pursuers: 3,
            evaders: 3,
            slip: 200,
            ..TagConfig::default()
        },
        build: |cfg, options| EnvPool::with_masks(cfg, options).unwrap(),
        auto_reset,
        limit: 200,
        deaths: true,
        branches: true,
    }
}

#[test]
fn test_continuation_cartpole_next_step_restores_identically_everywhere() {
    // CartPole draws only at reset: a branch diverges after the next reset.
    cartpole(AutoReset::NextStep).check_all(600);
}

#[test]
fn test_continuation_cartpole_same_step_restores_identically_everywhere() {
    cartpole(AutoReset::SameStep).check_all(600);
}

#[test]
fn test_continuation_tag_next_step_restores_identically_everywhere() {
    tag(AutoReset::NextStep).check_all(50);
}

#[test]
fn test_continuation_tag_same_step_restores_identically_everywhere() {
    tag(AutoReset::SameStep).check_all(50);
}

#[expect(clippy::unwrap_used, reason = "test setup: a failure is the assertion")]
fn grid(auto_reset: AutoReset) -> Case<Grid, NoMasks> {
    Case {
        cfg: GridConfig {
            size: 6,
            max_steps: 40,
        },
        build: |cfg, options| EnvPool::new(cfg, options).unwrap(),
        auto_reset,
        limit: 64,
        deaths: false,
        // The goal is drawn at each reset, and walks here end within the
        // harness's horizon, so a branch diverges at its next auto-reset.
        branches: true,
    }
}

#[test]
fn test_continuation_grid_next_step_restores_identically_everywhere() {
    grid(AutoReset::NextStep).check_all(50);
}

#[test]
fn test_continuation_grid_same_step_restores_identically_everywhere() {
    grid(AutoReset::SameStep).check_all(50);
}

#[expect(clippy::unwrap_used, reason = "test setup: a failure is the assertion")]
fn keydoor(auto_reset: AutoReset) -> Case<KeyDoor, Masked> {
    Case {
        cfg: KeyDoorConfig { size: 6, agents: 3 },
        build: |cfg, options| EnvPool::with_masks(cfg, options).unwrap(),
        auto_reset,
        limit: 40,
        deaths: false,
        // A new world is drawn at each reset, and the step limit forces resets
        // within the harness's horizon, so a branch diverges there.
        branches: true,
    }
}

#[test]
fn test_continuation_keydoor_next_step_restores_identically_everywhere() {
    keydoor(AutoReset::NextStep).check_all(50);
}

#[test]
fn test_continuation_keydoor_same_step_restores_identically_everywhere() {
    keydoor(AutoReset::SameStep).check_all(50);
}

#[expect(clippy::unwrap_used, reason = "test setup: a failure is the assertion")]
fn snake(auto_reset: AutoReset) -> Case<Snake, Masked> {
    Case {
        cfg: SnakeConfig {
            size: 6,
            agents: 2,
            obs: tessera_rl_classic::snake::OBS_STATE,
        },
        build: |cfg, options| EnvPool::with_masks(cfg, options).unwrap(),
        auto_reset,
        limit: 40,
        deaths: true,
        // Food is drawn after each meal, so a branch diverges there.
        branches: true,
    }
}

#[test]
fn test_continuation_snake_next_step_restores_identically_everywhere() {
    snake(AutoReset::NextStep).check_all(50);
}

#[test]
fn test_continuation_snake_same_step_restores_identically_everywhere() {
    snake(AutoReset::SameStep).check_all(50);
}
