// SPDX-License-Identifier: MIT OR Apache-2.0

//! Curricula through targets (ADR-0081 item 5) read through `reset_with`
//! (ADR-0091): a target applies only at the slot's next reset; fork and
//! restore keep the record's values; the pool context carries the
//! curriculum; environments that do not read adaptive values are refused.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    Adaptive, AgentSlot, Environment, EpisodeEnd, ObsLeaf, Observation, RecordHeader, SpaceError,
    StepOut, Tier,
};
use tessera_rl_engine::{
    ActionInput, AdaptiveTargets, AutoReset, Curriculum, EngineError, EnvPool, EpochReport,
    NoMasks, PoolOptions, Reseed,
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

/// The same environment without the override: refused.
struct Ignores;

impl Environment for Ignores {
    type State = LevelState;
    type Config = LevelConfig;
    type Obs = LevelObs;
    type Action = ();
    type Header = RecordHeader<1, LevelConfigAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &LevelConfig) -> Result<Space, SpaceError> {
        Level::observation_space(cfg)
    }

    fn action_space(cfg: &LevelConfig) -> Result<Space, SpaceError> {
        Level::action_space(cfg)
    }

    fn reset(cfg: &LevelConfig, rng: &mut StepRng, state: &mut LevelState) {
        Level::reset(cfg, rng, state);
    }

    fn step(
        cfg: &LevelConfig,
        state: &mut LevelState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        Level::step(cfg, state, actions, rng, out);
    }

    fn observe(
        cfg: &LevelConfig,
        state: &LevelState,
        agent: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, LevelObs>,
    ) {
        Level::observe(cfg, state, agent, rng, out);
    }
}

/// Raises every slot's level by one at each barrier with ended episodes;
/// counts its barriers in its state.
struct Raise;

impl Curriculum<Level> for Raise {
    const ID: &'static str = "tests/raise-v1";
    type State = u32;

    fn update(
        &mut self,
        report: &EpochReport,
        state: &mut u32,
        targets: &mut AdaptiveTargets<'_, Level>,
    ) {
        *state += 1;
        if report.episodes.count == 0 {
            return;
        }
        let mut next = *targets.get(0).unwrap();
        next.level += 1;
        targets.set_all(next);
    }
}

/// Raises only slot 1, through `set`: the report names exactly the
/// changed slot.
struct One;

impl Curriculum<Level> for One {
    const ID: &'static str = "tests/one-v1";
    type State = u32;

    fn update(&mut self, _: &EpochReport, _: &mut u32, targets: &mut AdaptiveTargets<'_, Level>) {
        let mut next = *targets.get(1).unwrap();
        next.level += 1;
        assert!(targets.set(1, next));
        assert!(!targets.set(99, next), "out of range");
    }
}

const N: u32 = 4;

fn options() -> PoolOptions {
    PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(3))
}

type Pool = EnvPool<Level, NoMasks>;

fn step(p: &mut Pool) {
    p.step(ActionInput {
        discrete: &[0; N as usize],
        continuous: &[],
    })
    .unwrap();
}

/// Each instance's episode level, as its observation reports it.
fn levels(p: &Pool) -> Vec<u32> {
    let obs = p.outputs().unwrap().observation(0).unwrap();
    obs.as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

fn with_level(level: u32) -> Adaptive<Level> {
    let mut a = *Pool::new(LevelConfig::default(), options())
        .unwrap()
        .with_targets()
        .unwrap()
        .targets()
        .unwrap()
        .first()
        .unwrap();
    a.level = level;
    a
}

#[test]
fn test_curriculum_target_applies_only_at_the_next_reset() {
    let mut p = Pool::new(LevelConfig::default(), options())
        .unwrap()
        .with_curriculum(Raise)
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(levels(&p), [1; N as usize]);
    step(&mut p);
    let report = p.epoch_barrier().unwrap();
    assert_eq!(
        report.curriculum.unwrap().ids,
        Vec::<u32>::new(),
        "no episode ended yet"
    );
    step(&mut p);
    step(&mut p); // the length-3 episodes end here
    let report = p.epoch_barrier().unwrap();
    assert_eq!(report.curriculum.unwrap().ids, [0, 1, 2, 3]);
    assert_eq!(p.targets().unwrap()[0].level, 2);
    // Mid-episode nothing changes; the auto-reset (next step) applies it.
    assert_eq!(levels(&p), [1; N as usize]);
    step(&mut p);
    assert_eq!(levels(&p), [2; N as usize]);
    assert_eq!(
        p.outputs().unwrap().rewards(),
        &[0.0; N as usize][..],
        "reset step"
    );
    step(&mut p);
    assert_eq!(p.outputs().unwrap().rewards(), &[2.0; N as usize][..]);
}

#[test]
fn test_curriculum_fork_and_restore_keep_the_record_and_the_slot_target_applies_next() {
    let mut p = Pool::new(LevelConfig::default(), options())
        .unwrap()
        .with_targets()
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p.set_targets(&[(1, with_level(7))]).unwrap();
    // Slot 0's record (level 1) forked into slot 1 keeps level 1 until slot
    // 1's next reset, which takes slot 1's target (7).
    p.fork(0, &[1]).unwrap();
    assert_eq!(levels(&p)[1], 1);
    p.reset(&[1], Reseed::Keep).unwrap();
    assert_eq!(levels(&p)[1], 7);
    let record = p.snapshot(1).unwrap();
    p.restore(2, &record).unwrap();
    assert_eq!(levels(&p)[2], 7, "the record's current values");
    p.reset(&[2], Reseed::Keep).unwrap();
    assert_eq!(levels(&p)[2], 1, "slot 2's target");
}

#[test]
fn test_curriculum_pool_context_restores_state_and_targets_and_refuses_mismatches() {
    let mut p = Pool::new(LevelConfig::default(), options())
        .unwrap()
        .with_curriculum(Raise)
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    for _ in 0..3 {
        step(&mut p);
    }
    p.epoch_barrier().unwrap();
    let context = p.pool_context().unwrap();
    assert!(context.adaptive.curriculum.identity != 0);
    for _ in 0..3 {
        step(&mut p);
    }
    p.epoch_barrier().unwrap();
    assert_ne!(p.pool_context().unwrap(), context);
    p.restore_pool_context(context.clone(), None).unwrap();
    assert_eq!(p.pool_context().unwrap(), context);

    // A targets-only pool has another identity: refused, nothing changed.
    let mut q = Pool::new(LevelConfig::default(), options())
        .unwrap()
        .with_targets()
        .unwrap();
    q.reset_all(Reseed::Keep).unwrap();
    let before = q.pool_context().unwrap();
    assert!(matches!(
        q.restore_pool_context(context.clone(), None),
        Err(EngineError::AdaptiveContext { .. })
    ));
    assert_eq!(q.pool_context().unwrap(), before);
    // A pool without targets refuses a context that has them.
    let mut bare = Pool::new(LevelConfig::default(), options()).unwrap();
    bare.reset_all(Reseed::Keep).unwrap();
    assert!(matches!(
        bare.restore_pool_context(context, None),
        Err(EngineError::AdaptiveContext { .. })
    ));
}

#[test]
fn test_curriculum_refused_for_an_env_that_ignores_adaptive_values_and_after_reset() {
    let ignores = EnvPool::<Ignores, NoMasks>::new(LevelConfig::default(), options()).unwrap();
    assert!(matches!(
        ignores.with_targets(),
        Err(EngineError::AdaptiveNotRead { .. })
    ));
    let mut late = Pool::new(LevelConfig::default(), options()).unwrap();
    late.reset_all(Reseed::Keep).unwrap();
    assert!(matches!(
        late.with_curriculum(Raise),
        Err(EngineError::Options { .. })
    ));
    let mut p = Pool::new(LevelConfig::default(), options()).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    assert!(matches!(
        p.set_targets(&[(0, with_level(2))]),
        Err(EngineError::AdaptiveNotRead { .. })
    ));
}

#[test]
fn test_curriculum_frozen_changes_nothing_and_without_override_is_bit_identical() {
    let mut p = Pool::new(LevelConfig::default(), options())
        .unwrap()
        .with_curriculum(Raise)
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p.freeze_curriculum(true);
    for _ in 0..3 {
        step(&mut p);
    }
    assert!(p.epoch_barrier().unwrap().curriculum.is_none());
    assert_eq!(p.targets().unwrap()[0].level, 1);
    // An environment that does not override `reset_with` runs as before:
    // the default forwards to `reset`, which reads the static config.
    let mut ig = EnvPool::<Ignores, NoMasks>::new(LevelConfig::default(), options()).unwrap();
    let mut lv = Pool::new(LevelConfig::default(), options()).unwrap();
    ig.reset_all(Reseed::Keep).unwrap();
    lv.reset_all(Reseed::Keep).unwrap();
    for _ in 0..5 {
        step(&mut lv);
        ig.step(ActionInput {
            discrete: &[0; N as usize],
            continuous: &[],
        })
        .unwrap();
        assert_eq!(
            ig.outputs().unwrap().rewards(),
            lv.outputs().unwrap().rewards()
        );
    }
}

#[test]
fn test_curriculum_report_names_exactly_the_slots_set() {
    let mut p = Pool::new(LevelConfig::default(), options())
        .unwrap()
        .with_curriculum(One)
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(p.epoch_barrier().unwrap().curriculum.unwrap().ids, [1]);
    assert_eq!(p.targets().unwrap()[1].level, 2);
    assert_eq!(p.targets().unwrap()[0].level, 1);
}

#[test]
fn test_curriculum_context_of_another_curriculum_with_the_same_state_is_refused() {
    // `One` and `Raise` share `State = u32`: only the identity tells them
    // apart.
    let mut raise = Pool::new(LevelConfig::default(), options())
        .unwrap()
        .with_curriculum(Raise)
        .unwrap();
    raise.reset_all(Reseed::Keep).unwrap();
    let context = raise.pool_context().unwrap();
    let mut one = Pool::new(LevelConfig::default(), options())
        .unwrap()
        .with_curriculum(One)
        .unwrap();
    one.reset_all(Reseed::Keep).unwrap();
    let before = one.pool_context().unwrap();
    assert!(matches!(
        one.restore_pool_context(context, None),
        Err(EngineError::AdaptiveContext { .. })
    ));
    assert_eq!(one.pool_context().unwrap(), before);
}

/// `Level` with a lane-wise state (ADR-0090, `#[state(lanes)]`): the
/// column store gathers and scatters each instance's state around the same
/// `reset_with` and `step_with` calls.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
#[state(lanes)]
struct LaneLevelState {
    level: u32,
    steps: u32,
}

struct LaneLevel;

impl Environment for LaneLevel {
    type State = LaneLevelState;
    type Config = LevelConfig;
    type Obs = LevelObs;
    type Action = ();
    type Header = RecordHeader<1, LevelConfigAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;
    const READS_ADAPTIVE: bool = true;

    fn observation_space(cfg: &LevelConfig) -> Result<Space, SpaceError> {
        Level::observation_space(cfg)
    }

    fn action_space(cfg: &LevelConfig) -> Result<Space, SpaceError> {
        Level::action_space(cfg)
    }

    fn reset(cfg: &LevelConfig, _: &mut StepRng, state: &mut LaneLevelState) {
        *state = LaneLevelState {
            level: cfg.level,
            steps: 0,
        };
    }

    fn reset_with(
        _: &LevelConfig,
        adaptive: &Adaptive<Self>,
        _: &mut StepRng,
        state: &mut LaneLevelState,
    ) {
        *state = LaneLevelState {
            level: adaptive.level,
            steps: 0,
        };
    }

    fn step(
        cfg: &LevelConfig,
        state: &mut LaneLevelState,
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
        state: &LaneLevelState,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, LevelObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(LevelObs::LEVEL) {
            leaf.set(0, state.level);
        }
    }
}

/// `Raise` for the lane-wise environment: the same rule.
struct LaneRaise;

impl Curriculum<LaneLevel> for LaneRaise {
    const ID: &'static str = "tests/raise-v1";
    type State = u32;

    fn update(
        &mut self,
        report: &EpochReport,
        state: &mut u32,
        targets: &mut AdaptiveTargets<'_, LaneLevel>,
    ) {
        *state += 1;
        if report.episodes.count == 0 {
            return;
        }
        let mut next = *targets.get(0).unwrap();
        next.level += 1;
        targets.set_all(next);
    }
}

#[test]
fn test_curriculum_lane_wise_state_reads_targets_as_the_row_store_does() {
    const { assert!(<LaneLevelState as tessera_rl_core::EnvState>::LANES) };
    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
        let opts = PoolOptions {
            auto_reset,
            ..options()
        };
        let mut rows = Pool::new(LevelConfig::default(), opts.clone())
            .unwrap()
            .with_curriculum(Raise)
            .unwrap();
        let mut lanes = EnvPool::<LaneLevel, NoMasks>::new(LevelConfig::default(), opts)
            .unwrap()
            .with_curriculum(LaneRaise)
            .unwrap();
        rows.reset_all(Reseed::Keep).unwrap();
        lanes.reset_all(Reseed::Keep).unwrap();
        let actions = [0; N as usize];
        for k in 0..30 {
            let input = || ActionInput {
                discrete: &actions,
                continuous: &[],
            };
            rows.step(input()).unwrap();
            lanes.step(input()).unwrap();
            let (r, l) = (rows.outputs().unwrap(), lanes.outputs().unwrap());
            assert_eq!(r.rewards(), l.rewards(), "{auto_reset:?} step {k}");
            assert_eq!(
                r.observation(0).unwrap(),
                l.observation(0).unwrap(),
                "{auto_reset:?} step {k}"
            );
            if k % 4 == 3 {
                let (a, b) = (
                    rows.epoch_barrier().unwrap(),
                    lanes.epoch_barrier().unwrap(),
                );
                assert_eq!(a, b, "{auto_reset:?} barrier at {k}");
            }
        }
        // The curriculum moved the levels, and both stores saw it.
        assert!(rows.targets().unwrap()[0].level > 2, "{auto_reset:?}");
        assert!(
            rows.outputs().unwrap().rewards().iter().any(|&r| r > 2.0),
            "{auto_reset:?}"
        );
    }
}
