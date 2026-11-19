// SPDX-License-Identifier: MIT OR Apache-2.0

//! The validation fast path's facts (env-step plan r3 section 3.3), one
//! directed test per fact: what clears `fast_clean` (a reset, a restore, an
//! import, a split-phase round trip) and what keeps `counter_bound` an upper
//! bound (steps, resets, counters written outside a step). Each test runs a
//! fast pool and a pool forced onto the slow path before every operation;
//! results, outputs and every record must be equal after each one.
//!
//! The environment folds every slot's actions into its state, also a slot
//! that is dead or does not act, which the slow path zeroes. A fast path
//! armed on a stale `fast_clean` copies those values instead, so the states
//! differ; one armed on a stale `counter_bound` skips the counter checks, so
//! a step the slow path refuses with `CounterExhausted` goes through.

use core::num::NonZeroU32;
#[cfg(not(loom))]
use core::num::NonZeroUsize;

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Bool8, Environment, EpisodeEnd, InvariantViolation, NoAdaptive, ObsLeaf,
    Observation, RecordHeader, RecordLayout, SpaceError, StepOut, Tier, TurnStructure, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::{RngIdentity, Seed};
use tessera_rl_num::rng::{ObsRng, StepRng};

#[cfg(not(loom))]
use crate::AsyncOptions;
use crate::record::{DecodedRecord, Record};
use crate::{ActionInput, AutoReset, EnvPool, PoolOptions, Reseed};

const N: usize = 4;
const N_U32: u32 = 4;
const SECOND: AgentSlot = match AgentSlot::new(1, 2) {
    Some(s) => s,
    None => AgentSlot::FIRST,
};

/// The first agent's second head: what the step does besides folding.
const QUIET: i32 = 0;
const KILL_SECOND: i32 = 1;
const END_EPISODE: i32 = 3;
const TOGGLE_IDLE: i32 = 4;

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Fold {
    acc: u32,
    /// 1: the second slot does not act.
    idle: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct FoldConfig {
    /// 1: after a reset the second slot does not act until toggled.
    #[config(default = 0, range = 0..=1)]
    idle_at_reset: u32,
}

#[derive(Observation)]
struct FoldObs {
    #[space(box(low = 0, high = 1, shape = [1]))]
    zero: ObsLeaf<i32>,
}

struct FoldEnv;

impl Environment for FoldEnv {
    type State = Fold;
    type Config = FoldConfig;
    type Obs = FoldObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = 2;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Mixed;

    fn observation_space(cfg: &FoldConfig) -> Result<Space, SpaceError> {
        FoldObs::space(cfg)
    }

    fn action_space(_: &FoldConfig) -> Result<Space, SpaceError> {
        Ok(Space::MultiDiscrete { nvec: vec![3, 5] })
    }

    fn reset(cfg: &FoldConfig, _: &mut StepRng, state: &mut Fold) {
        *state = Fold {
            acc: 0,
            idle: cfg.idle_at_reset,
        };
    }

    fn acting(_: &FoldConfig, state: &Fold, agent: AgentSlot) -> bool {
        agent.index() == 0 || state.idle == 0
    }

    fn step(
        _: &FoldConfig,
        state: &mut Fold,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        for agent in [AgentSlot::FIRST, SECOND] {
            for head in 0..2 {
                let v = actions.discrete(agent, head).unwrap_or(-9);
                state.acc = state
                    .acc
                    .wrapping_mul(131)
                    .wrapping_add(u32::from_ne_bytes(v.to_ne_bytes()));
            }
        }
        match actions.discrete(AgentSlot::FIRST, 1) {
            Some(KILL_SECOND) => out.agent_died(SECOND),
            Some(END_EPISODE) => out.end_episode(EpisodeEnd::Terminated),
            Some(TOGGLE_IDLE) => state.idle ^= 1,
            _ => {}
        }
    }

    fn observe(
        _: &FoldConfig,
        _: &Fold,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, FoldObs>,
    ) {
    }
}

impl Validate for FoldEnv {
    fn validate(_: &FoldConfig, _: &Fold) -> Result<(), InvariantViolation> {
        Ok(())
    }
}

fn pool(idle_at_reset: u32) -> EnvPool<FoldEnv> {
    let options = PoolOptions {
        auto_reset: AutoReset::SameStep,
        chunk_size: NonZeroU32::new(2).unwrap(),
        max_episode_steps: None,
        ..PoolOptions::new(NonZeroU32::new(N_U32).unwrap(), Seed::new(11))
    };
    EnvPool::<FoldEnv>::new(FoldConfig { idle_at_reset }, options).unwrap()
}

/// Outputs and every record, as text.
fn state(p: &mut EnvPool<FoldEnv>) -> Vec<String> {
    let v = p.outputs.view_all();
    let b = |s: &[Bool8]| s.iter().map(|f| f.get()).collect::<Vec<_>>();
    let mut out = vec![
        format!(
            "{:?}",
            v.rewards.iter().map(|f| f.to_bits()).collect::<Vec<_>>()
        ),
        format!("{:?}", b(v.alive)),
        format!("{:?}", b(v.acting)),
        format!("{:?}", b(v.died)),
        format!("{:?}", b(v.episode_terminated)),
        format!("{:?}", v.episode_length),
    ];
    for i in 0..N_U32 {
        out.push(format!("{:?}", p.snapshot(i).map(|r| r.to_bytes())));
    }
    out
}

/// A fast pool and a pool forced onto the slow path, run in lockstep.
struct Pair {
    fast: EnvPool<FoldEnv>,
    slow: EnvPool<FoldEnv>,
}

impl Pair {
    fn new(idle_at_reset: u32) -> Self {
        let mut p = Self {
            fast: pool(idle_at_reset),
            slow: pool(idle_at_reset),
        };
        p.op("reset_all", |q| {
            format!("{:?}", q.reset_all(Reseed::Keep).map(|_| ()))
        });
        p
    }

    /// Runs `f` on both pools; results and state must be equal.
    fn op(&mut self, what: &str, f: impl Fn(&mut EnvPool<FoldEnv>) -> String) -> String {
        self.slow.forget_fast_facts();
        let armed = self.fast.fast_path_armed();
        let (rf, rs) = (f(&mut self.fast), f(&mut self.slow));
        assert_eq!(rf, rs, "{what} (fast path armed: {armed}): results differ");
        assert_eq!(
            state(&mut self.fast),
            state(&mut self.slow),
            "{what} (fast path armed: {armed}): state differs"
        );
        rf
    }

    /// One step: `event` on instance `on` (every instance if `None`); the
    /// second slot always sends nonzero values, which the slow path zeroes
    /// where that slot is dead or does not act.
    fn step(&mut self, what: &str, event: i32, on: Option<usize>) -> String {
        let a = actions(event, on);
        self.op(what, |q| {
            format!(
                "{:?}",
                q.step(ActionInput {
                    discrete: &a,
                    continuous: &[],
                })
                .map(|_| ())
            )
        })
    }

    /// A step that must arm the fast path for the next one.
    fn clean_step(&mut self) {
        self.step("clean step", QUIET, None);
        assert!(
            self.fast.fast_path_armed(),
            "a clean step must arm the fast path"
        );
    }
}

fn actions(event: i32, on: Option<usize>) -> Vec<i32> {
    (0..N)
        .flat_map(|i| {
            let ev = if on.is_none_or(|k| k == i) {
                event
            } else {
                QUIET
            };
            [1, ev, 2, 3]
        })
        .collect()
}

/// Instance 0's record with its step and episode counters replaced.
fn forged(p: &EnvPool<FoldEnv>, episode_index: Option<u32>, steps: Option<u32>) -> Record<FoldEnv> {
    let mut r: Record<FoldEnv> = p.snapshot(0).unwrap();
    let e = r.header.engine_mut();
    let id = e.identity;
    e.identity = RngIdentity::from_parts(
        id.seed(),
        id.branch_salt(),
        id.logical_id(),
        episode_index.unwrap_or(id.episode_index()),
        steps.unwrap_or(id.step_index()),
    );
    if let Some(s) = steps {
        e.episode_steps = s;
    }
    r
}

#[test]
fn test_fast_facts_partial_reset_clears_clean_flag() {
    // After a clean step, a reset of instance 0 leaves its second slot not
    // acting: the next step must take the slow path.
    let mut p = Pair::new(1);
    p.step("toggle the second slot on", TOGGLE_IDLE, None);
    p.clean_step();
    p.op("reset [0]", |q| {
        format!("{:?}", q.reset(&[0], Reseed::Keep).map(|_| ()))
    });
    p.step("step after the reset", QUIET, None);
}

#[test]
fn test_fast_facts_step_leaving_a_slot_not_acting_clears_clean_flag() {
    // A step of a Mixed environment (which can arm) leaves the second slot
    // alive but not acting: that step must not set fast_clean.
    let mut p = Pair::new(0);
    p.clean_step();
    p.step("toggle the second slot off", TOGGLE_IDLE, None);
    p.step("step with the second slot not acting", QUIET, None);
}

#[test]
fn test_fast_facts_stale_restore_forgets_facts() {
    // A record taken while the second slot did not act, restored into a
    // pool whose facts hold.
    let mut p = Pair::new(1);
    let stale: Record<FoldEnv> = p.fast.snapshot(0).unwrap();
    p.step("toggle the second slot on", TOGGLE_IDLE, None);
    p.clean_step();
    p.op("restore 1 from the stale record", |q| {
        format!("{:?}", q.restore(1, &stale))
    });
    p.step("step after the restore", QUIET, None);
}

#[test]
fn test_fast_facts_stale_import_forgets_facts() {
    let mut p = Pair::new(1);
    let stale = DecodedRecord::<FoldEnv>::from(p.fast.snapshot(0).unwrap());
    p.step("toggle the second slot on", TOGGLE_IDLE, None);
    p.clean_step();
    p.op("import 2 from the stale record", |q| {
        format!("{:?}", q.import(&[2], &[stale]))
    });
    p.step("step after the import", QUIET, None);
}

#[test]
fn test_fast_facts_stale_restore_all_forgets_facts() {
    let mut p = Pair::new(1);
    let stale: Vec<Record<FoldEnv>> = (0..N_U32).map(|i| p.fast.snapshot(i).unwrap()).collect();
    p.step("toggle the second slot on", TOGGLE_IDLE, None);
    p.clean_step();
    p.op("restore_all from the stale records", |q| {
        format!("{:?}", q.restore_all(&stale))
    });
    p.step("step after restore_all", QUIET, None);
}

#[test]
fn test_fast_facts_step_counter_near_max_is_refused_alike() {
    // episode_steps restored a few steps below u32::MAX: the slow path
    // refuses the step that would exhaust it, so the fast path must not be
    // armed by then.
    let mut p = Pair::new(0);
    p.clean_step();
    let r = forged(&p.fast, None, Some(u32::MAX - 5));
    p.op("restore 0 with episode_steps MAX-5", |q| {
        format!("{:?}", q.restore(0, &r))
    });
    let mut refused = false;
    for k in 0..8 {
        if p.step(&format!("step {k} near the step limit"), QUIET, None)
            .contains("CounterExhausted")
        {
            refused = true;
            break;
        }
    }
    assert!(refused, "the step counter was never exhausted");
}

#[test]
fn test_fast_facts_episode_counter_near_max_is_refused_alike() {
    // episode_index restored a few episodes below u32::MAX and raised by
    // resets and by a same-step reset: the bound must follow both.
    let mut p = Pair::new(0);
    p.clean_step();
    let r = forged(&p.fast, Some(u32::MAX - 6), None);
    p.op("restore 0 with episode_index MAX-6", |q| {
        format!("{:?}", q.restore(0, &r))
    });
    p.step("step after the restore", QUIET, None);
    for k in 0..5 {
        p.op(&format!("reset [0] #{k}"), |q| {
            format!("{:?}", q.reset(&[0], Reseed::Keep).map(|_| ()))
        });
    }
    p.step("step after the resets", QUIET, None);
    p.step("end instance 0's episode", END_EPISODE, Some(0));
    let last = p.step("step after the episode end", QUIET, None);
    assert!(
        last.contains("CounterExhausted"),
        "the episode counter was never exhausted: {last}"
    );
}

/// Converts `pool` to Async mode (not built under loom), steps every instance once with `event` on
/// instance 0, and converts it back.
#[cfg(not(loom))]
fn async_round_trip(pool: EnvPool<FoldEnv>, event: i32) -> EnvPool<FoldEnv> {
    let mut a = pool
        .async_mode(AsyncOptions::new(
            NonZeroUsize::MIN,
            NonZeroUsize::new(N).unwrap(),
        ))
        .map_err(|e| e.error)
        .unwrap();
    let ids: Vec<u32> = (0..N_U32).collect();
    let act = actions(event, Some(0));
    a.send(
        &ids,
        ActionInput {
            discrete: &act,
            continuous: &[],
        },
    )
    .unwrap();
    drop(a.recv().unwrap());
    a.into_pool().map_err(|_| ()).unwrap()
}

#[cfg(not(loom))]
#[test]
fn test_fast_facts_split_phase_round_trip_forgets_facts() {
    // A clean pool goes through Async mode, where instance 0's second slot
    // dies; back in Inline mode, the next step must take the slow path.
    let mut p = Pair::new(0);
    p.clean_step();
    let Pair { fast, slow } = p;
    let mut p = Pair {
        fast: async_round_trip(fast, KILL_SECOND),
        slow: async_round_trip(slow, KILL_SECOND),
    };
    p.op("after the round trip", |_| String::new());
    p.step("step after the round trip", QUIET, None);
}
