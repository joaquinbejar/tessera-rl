// SPDX-License-Identifier: MIT OR Apache-2.0

//! Curriculum targets allocate nothing per step (ADR-0081 item 5.3, SPEC
//! 5.5): a Sync pool with a curriculum whose targets have changed steps,
//! auto-resets (copying the targets into headers) and resets without
//! allocating, on any thread, in both auto-reset modes. The barrier runs
//! outside the measured window: it allocates its report's list of changed
//! slots, once per epoch. One test per binary: the counting allocator
//! counts every thread while enabled.

// `allow`, not `expect`, and crate-level as in `alloc_rings.rs`: whether
// rustc reports the shared modules' unused items as dead differs between
// toolchains (on 1.95.0 the expectations were unfulfilled, #659), and
// `clippy::allow_attributes` covers outer attributes only.
#![allow(
    dead_code,
    reason = "the shared test modules: this binary uses their counting allocator only"
)]

mod common;

#[path = "common/alloc_diag.rs"]
mod alloc_diag;

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    Adaptive, AgentSlot, Environment, EpisodeEnd, ObsLeaf, Observation, RecordHeader, SpaceError,
    StepOut, Tier,
};
use tessera_rl_engine::{
    ActionInput, AdaptiveTargets, AutoReset, Curriculum, EnvPool, EpochReport, NoMasks,
    PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

#[derive(EnvConfig, Debug, Clone)]
struct LevelConfig {
    #[config(default = 1, adaptive)]
    level: u32,
    #[config(default = 3)]
    length: u32,
}

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct LevelState {
    level: u32,
    steps: u32,
}

#[derive(Observation)]
struct LevelObs {
    #[space(box(low = 0, high = 1_000_000, shape = [1]))]
    level: ObsLeaf<u32>,
}

/// The reward of every step is the episode's level, read at reset.
struct Level;

impl Environment for Level {
    type State = LevelState;
    type Config = LevelConfig;
    type Obs = LevelObs;
    type Action = ();
    type Header = RecordHeader<1, LevelConfigAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;
    const READS_ADAPTIVE: bool = true;

    fn observation_space(cfg: &LevelConfig) -> Result<Space, SpaceError> {
        LevelObs::space(cfg)
    }

    fn action_space(_: &LevelConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(cfg: &LevelConfig, _: &mut StepRng, state: &mut LevelState) {
        *state = LevelState {
            level: cfg.level,
            steps: 0,
        };
    }

    fn reset_with(
        _: &LevelConfig,
        adaptive: &Adaptive<Self>,
        _: &mut StepRng,
        state: &mut LevelState,
    ) {
        *state = LevelState {
            level: adaptive.level,
            steps: 0,
        };
    }

    fn step(
        cfg: &LevelConfig,
        state: &mut LevelState,
        _: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        state.steps += 1;
        #[expect(clippy::cast_precision_loss, reason = "small test values")]
        out.set_reward(AgentSlot::FIRST, state.level as f32);
        if state.steps >= cfg.length {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &LevelConfig,
        state: &LevelState,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, LevelObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(LevelObs::LEVEL) {
            leaf.set(0, state.level);
        }
    }
}

/// Raises every slot's level at each barrier.
struct Raise;

impl Curriculum<Level> for Raise {
    const ID: &'static str = "tests/alloc-raise-v1";
    type State = u32;

    fn update(
        &mut self,
        _: &EpochReport,
        state: &mut u32,
        targets: &mut AdaptiveTargets<'_, Level>,
    ) {
        *state += 1;
        if let Some(&first) = targets.get(0) {
            let mut next = first;
            next.level += 1;
            targets.set_all(next);
        }
    }
}

#[test]
fn test_curriculum_step_and_reset_do_not_allocate_on_any_thread() {
    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
        let options = PoolOptions {
            auto_reset,
            chunk_size: NonZeroU32::new(8).unwrap(),
            ..PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(1))
        };
        let mut pool = EnvPool::<Level, NoMasks>::new(LevelConfig::default(), options)
            .unwrap()
            .with_curriculum(Raise)
            .unwrap()
            .with_workers(SyncOptions::new(NonZeroUsize::new(4).unwrap()))
            .unwrap();
        let actions = vec![0i32; 64];
        let input = || ActionInput {
            discrete: &actions,
            continuous: &[],
        };
        pool.reset_all(Reseed::Keep).unwrap();
        pool.step(input()).unwrap();
        // Targets changed: every reset in the window copies a new level.
        let report = pool.epoch_barrier().unwrap();
        assert_eq!(report.curriculum.map(|c| c.ids.len()), Some(64));
        alloc_diag::start();
        for _ in 0..20 {
            let _ = std::hint::black_box(pool.step(input()));
        }
        let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
        let n = alloc_diag::stop();
        assert_eq!(
            n,
            0,
            "{auto_reset:?}: allocations in the measured window:\n{}",
            alloc_diag::report()
        );
        // The window's resets took the raised target.
        let obs = pool.outputs().unwrap().observation(0).unwrap().to_vec();
        assert!(
            obs.as_chunks::<4>()
                .0
                .iter()
                .all(|b| u32::from_le_bytes(*b) == 2),
            "{auto_reset:?}"
        );
    }
}
