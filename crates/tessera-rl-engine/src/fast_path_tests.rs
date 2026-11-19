// SPDX-License-Identifier: MIT OR Apache-2.0

//! The validation fast path (env-step plan r3 section 3.3) against the slow
//! path, differentially: two identical pools run the same random sequence of
//! steps (valid and invalid actions), resets, snapshots, restores, forks and
//! imports; one may take the fast path, the other is forced onto the slow
//! path before every operation. Results, errors, outputs and every record
//! must be identical, in Inline and Sync mode and under both auto-reset modes
//! that step after an episode end.

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Bool8, Environment, EpisodeEnd, InvariantViolation, NoAdaptive, ObsLeaf,
    Observation, RecordHeader, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

use crate::record::{DecodedRecord, Record};
use crate::{ActionInput, AutoReset, EnvPool, PoolOptions, Reseed, SyncOptions};

const KILL: i32 = 1;
const SPAWN: i32 = 2;
const END: i32 = 4;
const AGENTS: usize = 2;
const N: usize = 6;
const OPS: usize = 600;
const N_U64: u64 = N as u64;
const SECOND: AgentSlot = match AgentSlot::new(1, 2) {
    Some(s) => s,
    None => AgentSlot::FIRST,
};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Count {
    steps: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct CountConfig {}

#[derive(Observation)]
struct CountObs {
    #[space(box(low = 0, high = 1_000_000, shape = [1]))]
    steps: ObsLeaf<i32>,
}

/// Agent 0's action carries event bits; the second slot dies and respawns.
struct Events;

impl Environment for Events {
    type State = Count;
    type Config = CountConfig;
    type Obs = CountObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = AGENTS;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        CountObs::space(cfg)
    }

    fn action_space(_: &CountConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 8 })
    }

    fn reset(_: &CountConfig, _: &mut StepRng, state: &mut Count) {
        *state = Count::default();
    }

    fn step(
        _: &CountConfig,
        state: &mut Count,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        state.steps += 1;
        let bits = actions.discrete(AgentSlot::FIRST, 0).unwrap_or(0);
        out.set_reward(AgentSlot::FIRST, f32::from(u8::try_from(bits).unwrap_or(0)));
        if bits & KILL != 0 {
            out.agent_died(SECOND);
        }
        if bits & SPAWN != 0 {
            out.agent_spawned(SECOND);
        }
        if bits & END != 0 {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &CountConfig,
        _: &Count,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, CountObs>,
    ) {
    }
}

impl Validate for Events {
    fn validate(_: &CountConfig, _: &Count) -> Result<(), InvariantViolation> {
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
}

fn pool(auto_reset: AutoReset, workers: usize) -> EnvPool<Events> {
    let options = PoolOptions {
        auto_reset,
        chunk_size: NonZeroU32::new(2).unwrap(),
        max_episode_steps: NonZeroU32::new(9),
        ..PoolOptions::new(
            NonZeroU32::new(u32::try_from(N).unwrap()).unwrap(),
            Seed::new(7),
        )
    };
    let p = EnvPool::<Events>::new(CountConfig {}, options).unwrap();
    match NonZeroUsize::new(workers) {
        Some(w) => p.with_workers(SyncOptions::new(w)).unwrap(),
        None => p,
    }
}

/// Every output the step writes, and every record, as comparable values.
fn state(p: &mut EnvPool<Events>) -> (Vec<String>, Vec<Vec<u8>>) {
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
        format!("{:?}", v.observations[0]),
    ];
    let records = (0..N)
        .map(|i| p.snapshot(u32::try_from(i).unwrap()).unwrap().to_bytes())
        .collect();
    (outputs, records)
}

/// One random action buffer: mostly quiet, some events, and with
/// probability 1/12 one out-of-range value (8 or -1) at a random row.
fn actions(rng: &mut Rng) -> Vec<i32> {
    let mut a: Vec<i32> = (0..N)
        .flat_map(|_| {
            let bits = match rng.below(10) {
                0 => KILL,
                1 => SPAWN,
                2 => END,
                3 => KILL | SPAWN,
                _ => 0,
            };
            [bits, 0]
        })
        .collect();
    if rng.below(12) == 0 {
        let k = rng.below(u64::try_from(a.len()).unwrap());
        a[k] = if rng.below(2) == 0 { 8 } else { -1 };
    }
    a
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

/// Runs `OPS` random operations on a fast-path pool and a slow-path
/// reference; returns how many steps took the fast path.
fn differential(auto_reset: AutoReset, workers: usize, seed: u64) -> usize {
    let mut fast = pool(auto_reset, workers);
    let mut slow = pool(auto_reset, 0);
    let mut rng = Rng(seed);
    let mut fast_steps = 0;
    fast.reset_all(Reseed::Keep).unwrap();
    slow.reset_all(Reseed::Keep).unwrap();
    for op in 0..OPS {
        slow.forget_fast_facts();
        let kind = rng.below(20);
        let (rf, rs): (String, String) = match kind {
            0 => (
                format!("{:?}", fast.reset_all(Reseed::Keep).map(|_| ())),
                format!("{:?}", slow.reset_all(Reseed::Keep).map(|_| ())),
            ),
            1 => {
                let ids = ids(&mut rng);
                (
                    format!("{:?}", fast.reset(&ids, Reseed::Keep).map(|_| ())),
                    format!("{:?}", slow.reset(&ids, Reseed::Keep).map(|_| ())),
                )
            }
            2 => {
                let (src, dst) = (rng.below(N_U64), rng.below(N_U64));
                let r: Record<Events> = fast.snapshot(u32::try_from(src).unwrap()).unwrap();
                let d = u32::try_from(dst).unwrap();
                (
                    format!("{:?}", fast.restore(d, &r)),
                    format!("{:?}", slow.restore(d, &r)),
                )
            }
            3 => {
                let mut all: Vec<Record<Events>> = (0..N)
                    .map(|i| fast.snapshot(u32::try_from(i).unwrap()).unwrap())
                    .collect();
                all.rotate_left(1);
                (
                    format!("{:?}", fast.restore_all(&all)),
                    format!("{:?}", slow.restore_all(&all)),
                )
            }
            4 => {
                let src = u32::try_from(rng.below(N_U64)).unwrap();
                let dst: Vec<u32> = ids(&mut rng).into_iter().filter(|&d| d != src).collect();
                (
                    format!("{:?}", fast.fork(src, &dst)),
                    format!("{:?}", slow.fork(src, &dst)),
                )
            }
            5 => {
                let src = u32::try_from(rng.below(N_U64)).unwrap();
                let dst: Vec<u32> = ids(&mut rng).into_iter().filter(|&d| d != src).collect();
                (
                    format!("{:?}", fast.fork_branched(src, &dst, None)),
                    format!("{:?}", slow.fork_branched(src, &dst, None)),
                )
            }
            6 => {
                let id = u32::try_from(rng.below(N_U64)).unwrap();
                let from = u32::try_from(rng.below(N_U64)).unwrap();
                let rec = DecodedRecord::<Events>::from(fast.snapshot(from).unwrap());
                (
                    format!("{:?}", fast.import(&[id], &[rec])),
                    format!("{:?}", slow.import(&[id], &[rec])),
                )
            }
            7 => {
                let recs: Vec<DecodedRecord<Events>> = (0..N)
                    .rev()
                    .map(|i| DecodedRecord::from(fast.snapshot(u32::try_from(i).unwrap()).unwrap()))
                    .collect();
                (
                    format!("{:?}", fast.import_all(&recs)),
                    format!("{:?}", slow.import_all(&recs)),
                )
            }
            _ => {
                let a = actions(&mut rng);
                if fast.fast_path_armed() {
                    fast_steps += 1;
                }
                let input = ActionInput {
                    discrete: &a,
                    continuous: &[],
                };
                (
                    format!("{:?}", fast.step(input).map(|_| ())),
                    format!("{:?}", slow.step(input).map(|_| ())),
                )
            }
        };
        assert_eq!(rf, rs, "op {op} (kind {kind}): results differ");
        assert_eq!(
            state(&mut fast),
            state(&mut slow),
            "op {op} (kind {kind}): state differs"
        );
    }
    fast_steps
}

#[test]
fn test_fast_path_inline_same_step_matches_slow_path() {
    for seed in 1..=4 {
        assert!(
            differential(AutoReset::SameStep, 0, seed) > 0,
            "seed {seed}: fast path never taken"
        );
    }
}

#[test]
fn test_fast_path_inline_next_step_matches_slow_path() {
    for seed in 5..=8 {
        assert!(
            differential(AutoReset::NextStep, 0, seed) > 0,
            "seed {seed}: fast path never taken"
        );
    }
}

#[test]
fn test_fast_path_sync_same_step_matches_slow_path() {
    for seed in 9..=11 {
        assert!(
            differential(AutoReset::SameStep, 2, seed) > 0,
            "seed {seed}: fast path never taken"
        );
    }
}

#[test]
fn test_fast_path_sync_next_step_matches_slow_path() {
    for seed in 12..=14 {
        assert!(
            differential(AutoReset::NextStep, 2, seed) > 0,
            "seed {seed}: fast path never taken"
        );
    }
}

/// A refused fast-path call mutates nothing: outputs and every record are
/// unchanged after an out-of-range value.
#[test]
fn test_fast_path_refused_call_mutates_nothing() {
    let mut p = pool(AutoReset::SameStep, 0);
    p.reset_all(Reseed::Keep).unwrap();
    let quiet = vec![0; N * AGENTS];
    let input = ActionInput {
        discrete: &quiet,
        continuous: &[],
    };
    p.step(input).unwrap();
    p.step(input).unwrap();
    assert!(p.fast_path_armed(), "the fast path is armed");
    let records = state(&mut p);
    let mut bad = quiet.clone();
    bad[3] = 8;
    let err = p.step(ActionInput {
        discrete: &bad,
        continuous: &[],
    });
    assert!(err.is_err());
    assert_eq!(state(&mut p), records);
}
