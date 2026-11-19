// SPDX-License-Identifier: MIT OR Apache-2.0

//! The validation fast path against the slow path (env-step plan r3 section
//! 3.3) for an environment that exercises every check the fast path
//! replaces: two discrete heads and one continuous head per agent, three
//! agent slots, `Mixed` turns where some slots do not act, and a slot that
//! dies and respawns. As `fast_path_tests`, two identical pools run the same
//! random operations, one forced onto the slow path before each; results,
//! errors, outputs and every record must be equal, and the fast path must be
//! taken.

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, Bool8, Environment, EpisodeEnd, InvariantViolation, NoAdaptive, ObsLeaf,
    Observation, RecordHeader, SpaceError, StepOut, Tier, TurnStructure, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

use crate::record::{DecodedRecord, Record};
use crate::{ActionInput, AutoReset, EnvPool, PoolOptions, Reseed, SyncOptions};

const AGENTS: usize = 3;
const DH: usize = 2;
const NVEC: [u32; DH] = [3, 5];
const N: usize = 6;
const N_U64: u64 = 6;
const OPS: usize = 1200;
/// Steps on the fast path each seed must reach, so that a run poisoned
/// and rebuilt often (debug-checks) still covers it.
const MIN_FAST: usize = 40;
const THIRD: AgentSlot = match AgentSlot::new(2, 3) {
    Some(s) => s,
    None => AgentSlot::FIRST,
};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Turns {
    steps: u32,
    /// 1 while every alive slot acts; 0 while slot 1 sits out.
    all_act: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct TurnsConfig {}

#[derive(Observation)]
struct TurnsObs {
    #[space(box(low = 0, high = 1, shape = [1]))]
    zero: ObsLeaf<i32>,
}

/// Agent 0's head 0: 2 toggles "every slot acts"; head 1: 1 ends the
/// episode, 3 spawns slot 2, 4 kills slot 2. Every head and the continuous
/// action of every acting slot is validated by the engine.
struct Mixed;

impl Environment for Mixed {
    type State = Turns;
    type Config = TurnsConfig;
    type Obs = TurnsObs;
    type Action = ();
    type Header = RecordHeader<3, NoAdaptive>;

    const MAX_AGENTS: usize = AGENTS;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Mixed;

    fn observation_space(cfg: &TurnsConfig) -> Result<Space, SpaceError> {
        TurnsObs::space(cfg)
    }

    fn action_space(_: &TurnsConfig) -> Result<Space, SpaceError> {
        Ok(Space::Tuple(vec![
            Space::MultiDiscrete {
                nvec: NVEC.to_vec(),
            },
            Space::Box {
                shape: vec![1],
                dtype: DType::F32,
                bounds: BoxBounds::Float {
                    low: vec![-1.0],
                    high: vec![1.0],
                },
            },
        ]))
    }

    fn acting(_: &TurnsConfig, state: &Turns, agent: AgentSlot) -> bool {
        state.all_act == 1 || agent.index() != 1
    }

    fn reset(_: &TurnsConfig, _: &mut StepRng, state: &mut Turns) {
        *state = Turns {
            steps: 0,
            all_act: 1,
        };
    }

    fn step(
        _: &TurnsConfig,
        state: &mut Turns,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        state.steps += 1;
        if actions.discrete(AgentSlot::FIRST, 0) == Some(2) {
            state.all_act ^= 1;
        }
        let pull = actions.continuous(AgentSlot::FIRST, 0).unwrap_or(0.0);
        out.set_reward(AgentSlot::FIRST, pull);
        match actions.discrete(AgentSlot::FIRST, 1) {
            Some(1) => out.end_episode(EpisodeEnd::Terminated),
            Some(3) => out.agent_spawned(THIRD),
            Some(4) => out.agent_died(THIRD),
            _ => {}
        }
    }

    fn observe(
        _: &TurnsConfig,
        _: &Turns,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, TurnsObs>,
    ) {
    }
}

impl Validate for Mixed {
    fn validate(_: &TurnsConfig, _: &Turns) -> Result<(), InvariantViolation> {
        Ok(())
    }
}

/// xorshift64: a fixed pseudo-random operation stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> usize {
        usize::try_from(self.next() % n).unwrap_or(0)
    }

    fn below_i32(&mut self, n: u32) -> i32 {
        i32::try_from(self.next() % u64::from(n)).unwrap_or(0)
    }
}

fn pool(auto_reset: AutoReset, workers: usize) -> EnvPool<Mixed> {
    let options = PoolOptions {
        auto_reset,
        chunk_size: NonZeroU32::new(2).unwrap(),
        max_episode_steps: NonZeroU32::new(9),
        ..PoolOptions::new(
            NonZeroU32::new(u32::try_from(N).unwrap()).unwrap(),
            Seed::new(5),
        )
    };
    let p = EnvPool::<Mixed>::new(TurnsConfig {}, options).unwrap();
    match NonZeroUsize::new(workers) {
        Some(w) => p.with_workers(SyncOptions::new(w)).unwrap(),
        None => p,
    }
}

fn state(p: &mut EnvPool<Mixed>) -> (Vec<String>, Vec<String>) {
    let v = p.outputs.view_all();
    let b = |s: &[Bool8]| s.iter().map(|f| f.get()).collect::<Vec<_>>();
    let outputs = vec![
        format!("{:?}", v.rewards),
        format!("{:?}", b(v.agent_terminated)),
        format!("{:?}", b(v.agent_truncated)),
        format!("{:?}", b(v.died)),
        format!("{:?}", b(v.spawned)),
        format!("{:?}", b(v.invalid_agent_event)),
        format!("{:?}", b(v.alive)),
        format!("{:?}", b(v.acting)),
        format!("{:?}", v.generations),
        format!("{:?}", b(v.episode_terminated)),
        format!("{:?}", b(v.episode_truncated)),
        format!("{:?}", v.episode_return),
        format!("{:?}", v.episode_length),
    ];
    // A poisoned pool refuses snapshots (debug-checks poisons on invalid
    // agent events): equal refusals are an equal state.
    let records = (0..N)
        .map(|i| {
            format!(
                "{:?}",
                p.snapshot(u32::try_from(i).unwrap()).map(|r| r.to_bytes())
            )
        })
        .collect();
    (outputs, records)
}

/// One random action input: every head of every slot in range (also those
/// of slots that do not act, which the engine zeroes), and with probability
/// 1/6 one bad value: a discrete value out of its head's range, or a
/// continuous value out of bounds, infinite or NaN, in any slot.
fn actions(rng: &mut Rng) -> (Vec<i32>, Vec<f32>) {
    let mut d = Vec::with_capacity(N * AGENTS * DH);
    let mut c = Vec::with_capacity(N * AGENTS);
    for _ in 0..N * AGENTS {
        let toggle = rng.below(40) == 0;
        d.push(if toggle { 2 } else { rng.below_i32(2) });
        d.push(match rng.below(40) {
            0 => 1,
            1 => 3,
            2 => 4,
            _ => 0,
        });
        c.push(match rng.below(4) {
            0 => -1.0,
            1 => 1.0,
            _ => 0.25,
        });
    }
    match rng.below(18) {
        0 => {
            let k = rng.below(u64::try_from(d.len()).unwrap());
            d[k] = [3, 5, -1, 9][rng.below(4)];
        }
        1 => {
            let k = rng.below(u64::try_from(d.len()).unwrap()) / DH * DH + 1;
            d[k] = 5;
        }
        2 => {
            let k = rng.below(u64::try_from(c.len()).unwrap());
            c[k] = [1.5, -1.5, f32::INFINITY, f32::NAN][rng.below(4)];
        }
        _ => {}
    }
    (d, c)
}

fn ids(rng: &mut Rng) -> Vec<u32> {
    let mut v: Vec<u32> = (0..u32::try_from(N).unwrap())
        .filter(|_| rng.below(2) == 0)
        .collect();
    if v.is_empty() {
        v.push(0);
    }
    v
}

/// Runs operation `kind` on both pools; returns their results as text.
fn apply(
    kind: usize,
    fast: &mut EnvPool<Mixed>,
    slow: &mut EnvPool<Mixed>,
    rng: &mut Rng,
    fast_steps: &mut usize,
) -> (String, String) {
    match kind {
        0 => (
            format!("{:?}", fast.reset_all(Reseed::Keep).map(|_| ())),
            format!("{:?}", slow.reset_all(Reseed::Keep).map(|_| ())),
        ),
        1 => {
            let ids = ids(rng);
            (
                format!("{:?}", fast.reset(&ids, Reseed::Keep).map(|_| ())),
                format!("{:?}", slow.reset(&ids, Reseed::Keep).map(|_| ())),
            )
        }
        2 => {
            let src = u32::try_from(rng.below(N_U64)).unwrap();
            let dst = u32::try_from(rng.below(N_U64)).unwrap();
            let r: Record<Mixed> = fast.snapshot(src).unwrap();
            (
                format!("{:?}", fast.restore(dst, &r)),
                format!("{:?}", slow.restore(dst, &r)),
            )
        }
        3 => {
            let src = u32::try_from(rng.below(N_U64)).unwrap();
            let dst: Vec<u32> = ids(rng).into_iter().filter(|&d| d != src).collect();
            (
                format!("{:?}", fast.fork(src, &dst)),
                format!("{:?}", slow.fork(src, &dst)),
            )
        }
        4 => {
            let id = u32::try_from(rng.below(N_U64)).unwrap();
            let from = u32::try_from(rng.below(N_U64)).unwrap();
            let rec = DecodedRecord::<Mixed>::from(fast.snapshot(from).unwrap());
            (
                format!("{:?}", fast.import(&[id], &[rec])),
                format!("{:?}", slow.import(&[id], &[rec])),
            )
        }
        _ => {
            let (d, c) = actions(rng);
            if fast.fast_path_armed() {
                *fast_steps += 1;
            }
            let input = ActionInput {
                discrete: &d,
                continuous: &c,
            };
            (
                format!("{:?}", fast.step(input).map(|_| ())),
                format!("{:?}", slow.step(input).map(|_| ())),
            )
        }
    }
}

fn differential(auto_reset: AutoReset, workers: usize, seed: u64) -> usize {
    let mut fast = pool(auto_reset, workers);
    let mut slow = pool(auto_reset, 0);
    let mut rng = Rng(seed);
    let mut fast_steps = 0;
    fast.reset_all(Reseed::Keep).unwrap();
    slow.reset_all(Reseed::Keep).unwrap();
    for op in 0..OPS {
        slow.forget_fast_facts();
        let kind = rng.below(48);
        let (rf, rs) = apply(kind, &mut fast, &mut slow, &mut rng, &mut fast_steps);
        assert_eq!(rf, rs, "op {op} (kind {kind}): results differ");
        assert_eq!(
            state(&mut fast),
            state(&mut slow),
            "op {op} (kind {kind}): state differs"
        );
        // Under debug-checks an invalid agent event poisons the pool: both
        // must poison together, and the run goes on with fresh pools.
        assert_eq!(
            fast.poisoned, slow.poisoned,
            "op {op} (kind {kind}): poisoning differs"
        );
        if fast.poisoned {
            fast = pool(auto_reset, workers);
            slow = pool(auto_reset, 0);
            fast.reset_all(Reseed::Keep).unwrap();
            slow.reset_all(Reseed::Keep).unwrap();
        }
    }
    fast_steps
}

#[test]
fn test_fast_path_mixed_heads_inline_same_step_matches_slow_path() {
    for seed in 21..=24 {
        assert!(
            differential(AutoReset::SameStep, 0, seed) >= MIN_FAST,
            "seed {seed}: fast path taken too rarely"
        );
    }
}

#[test]
fn test_fast_path_mixed_heads_inline_next_step_matches_slow_path() {
    for seed in 25..=28 {
        assert!(
            differential(AutoReset::NextStep, 0, seed) >= MIN_FAST,
            "seed {seed}: fast path taken too rarely"
        );
    }
}

#[test]
fn test_fast_path_mixed_heads_sync_same_step_matches_slow_path() {
    for seed in 29..=31 {
        assert!(
            differential(AutoReset::SameStep, 2, seed) >= MIN_FAST,
            "seed {seed}: fast path taken too rarely"
        );
    }
}
