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
    BrickWall, BrickWallConfig, CartPole, CartPoleConfig, Grid, GridConfig, KeyDoor, KeyDoorConfig,
    MountainCar, MountainCarConfig, PaddleDuel, PaddleDuelConfig, Snake, SnakeConfig, Tag,
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

#[expect(clippy::unwrap_used, reason = "test setup: a failure is the assertion")]
fn brickwall(auto_reset: AutoReset) -> Case<BrickWall, NoMasks> {
    Case {
        cfg: BrickWallConfig {
            frameskip: 4,
            lives: 1,
            obs: tessera_rl_classic::brickwall::OBS_STATE,
        },
        build: |cfg, options| EnvPool::new(cfg, options).unwrap(),
        auto_reset,
        limit: 40,
        deaths: false,
        // A serve draws its speed, so a branch diverges there.
        branches: true,
    }
}

#[test]
fn test_continuation_brickwall_next_step_restores_identically_everywhere() {
    brickwall(AutoReset::NextStep).check_all(50);
}

#[test]
fn test_continuation_brickwall_same_step_restores_identically_everywhere() {
    brickwall(AutoReset::SameStep).check_all(50);
}

#[expect(clippy::unwrap_used, reason = "test setup: a failure is the assertion")]
fn paddleduel(auto_reset: AutoReset) -> Case<PaddleDuel, NoMasks> {
    Case {
        cfg: PaddleDuelConfig {
            agents: 2,
            frameskip: 4,
            max_score: 1,
            bounce: 1,
            obs: tessera_rl_classic::paddleduel::OBS_STATE,
        },
        build: |cfg, options| EnvPool::new(cfg, options).unwrap(),
        auto_reset,
        limit: 40,
        deaths: false,
        // A serve draws its speeds, so a branch diverges there.
        branches: true,
    }
}

#[test]
fn test_continuation_paddleduel_next_step_restores_identically_everywhere() {
    paddleduel(AutoReset::NextStep).check_all(50);
}

#[test]
fn test_continuation_paddleduel_same_step_restores_identically_everywhere() {
    paddleduel(AutoReset::SameStep).check_all(50);
}

fn mountaincar(auto_reset: AutoReset) -> Case<MountainCar, NoMasks> {
    Case {
        cfg: MountainCarConfig::default(),
        build: |cfg, options| EnvPool::new(cfg, options).unwrap(),
        auto_reset,
        // The dynamics draw nothing, so branches diverge only at a reset:
        // a short limit puts resets inside the 50-step horizon.
        limit: 40,
        deaths: false,
        branches: true,
    }
}

#[test]
fn test_continuation_mountaincar_next_step_restores_identically_everywhere() {
    mountaincar(AutoReset::NextStep).check_all(50);
}

#[test]
fn test_continuation_mountaincar_same_step_restores_identically_everywhere() {
    mountaincar(AutoReset::SameStep).check_all(50);
}
