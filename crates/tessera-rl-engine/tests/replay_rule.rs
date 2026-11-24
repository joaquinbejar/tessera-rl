// SPDX-License-Identifier: MIT OR Apache-2.0

//! A changed environment rule under the same id, version and config
//! (ADR-0072 section 10): replay reports `Diverged` at the first failing
//! checkpoint and names the last one that matched. The rule is a process
//! switch, so this file holds a single test.

#![cfg(feature = "replay")]
#![expect(clippy::unwrap_used, reason = "tests: a failure is the assertion")]

use core::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, Ordering};

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, InvariantViolation, NoAdaptive, ObsLeaf, Observation, Persist,
    RecordHeader, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_engine::replay::{
    DivergenceKind, Recorder, Replay, ReplayError, ReplayWriterOptions,
};
use tessera_rl_engine::{ActionInput, DecodeLimits, EnvPool, NoMasks, PoolOptions, Reseed};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

/// From the fourth step of an episode on, the increment is 2, not 1.
static CHANGED_RULE: AtomicBool = AtomicBool::new(false);

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Count {
    value: i32,
    steps: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct CountCfg {
    #[config(default = 1)]
    size: u32,
}

#[derive(Observation)]
struct CountView {
    #[space(box(low = -1_000_000, high = 1_000_000, shape = [1]))]
    value: ObsLeaf<i32>,
}

struct Flip;

impl Environment for Flip {
    type State = Count;
    type Config = CountCfg;
    type Obs = CountView;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;
    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &CountCfg) -> Result<Space, SpaceError> {
        CountView::space(cfg)
    }

    fn action_space(_: &CountCfg) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 3 })
    }

    fn reset(_: &CountCfg, _: &mut StepRng, state: &mut Count) {
        *state = Count::default();
    }

    fn step(
        _: &CountCfg,
        state: &mut Count,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let bonus = if CHANGED_RULE.load(Ordering::Relaxed) && state.steps >= 3 {
            2
        } else {
            1
        };
        let inc = actions.discrete(AgentSlot::FIRST, 0).unwrap_or(0) + bonus;
        state.value += inc;
        state.steps += 1;
        out.set_reward(AgentSlot::FIRST, f32::from(i16::try_from(inc).unwrap_or(0)));
    }

    fn observe(
        _: &CountCfg,
        state: &Count,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, CountView>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(CountView::VALUE) {
            leaf.set(0, state.value);
        }
    }
}

impl Persist for Flip {
    const ENV_ID: &'static str = "test/Flip-v1";
    const ENV_VERSION: u32 = 1;
}

impl Validate for Flip {
    fn validate(_: &CountCfg, _: &Count) -> Result<(), InvariantViolation> {
        Ok(())
    }
}

fn pool() -> EnvPool<Flip, NoMasks> {
    let options = PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(4));
    EnvPool::new(CountCfg::default(), options).unwrap()
}

#[test]
fn test_replay_changed_rule_diverges_at_the_first_failing_checkpoint() {
    let mut p = pool();
    let options = ReplayWriterOptions {
        keyframe_every: 0,
        checkpoint_every: 2,
    };
    let mut rec = Recorder::new(&mut p, Vec::new(), options).unwrap();
    rec.reset_all(Reseed::Keep).unwrap();
    for t in 0..6 {
        rec.step(ActionInput {
            discrete: &[t % 3, 1],
            continuous: &[],
        })
        .unwrap();
    }
    let bytes = rec.finish().unwrap();
    let replay = Replay::open(&bytes, DecodeLimits::default()).unwrap();
    replay.run(&mut pool()).unwrap();

    // Events: 0 keyframe, 1 reset, 2 step, 3 checkpoint, 4-5 steps,
    // 6 checkpoint, 7-8 steps, 9 checkpoint, 10 step, 11 final checkpoint.
    // The changed rule first acts in the fourth step (event 7), so the
    // checkpoint at event 9 fails and the one at event 6 is the last match.
    CHANGED_RULE.store(true, Ordering::Relaxed);
    let err = replay.run(&mut pool()).unwrap_err();
    CHANGED_RULE.store(false, Ordering::Relaxed);
    assert!(
        matches!(
            err,
            ReplayError::Diverged {
                event_index: 9,
                kind: DivergenceKind::Checkpoint,
                instance: Some(0),
                last_matching_checkpoint: Some(6),
            }
        ),
        "{err:?}"
    );
}
