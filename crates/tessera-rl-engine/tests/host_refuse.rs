// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-agent refusal in host pools (ADR-0070): `refuse` runs per instance
//! before `step`, refused agents are reported in the `refused` and
//! `refuse_code` outputs while every other agent and instance steps, and a
//! refusal of a slot that is not acting poisons the pool.

#![cfg(feature = "host")]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers unwrap and slice: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    ActionReject, AgentSlot, Bool8, EnvConfig, HostEnvironment, HostError, HostOut, HostSeed,
    Refusals, RefuseCode, SpaceError, TurnStructure,
};
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, HostPool, Outputs, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;

const SLOTS: usize = 3;
const N: usize = 8;
const MAX_INSTANCES: usize = 16;

/// The agent scores a point (0 does nothing).
const SCORE: i32 = 1;
/// `refuse` refuses the agent with `REFUSE_CODE + slot`.
const REFUSE: i32 = 2;
/// `refuse` refuses slot 2, which is not alive with `initial == 2`.
const MISUSE: i32 = 3;
/// `validate` rejects the whole call.
const REJECT: i32 = 4;
/// `refuse` panics.
const PANIC: i32 = 5;
const CHOICES: u32 = 6;

const REFUSE_CODE: u32 = 20;
const REJECT_CODE: u32 = 7;

#[derive(Debug, Default)]
struct Shared {
    next_id: AtomicU64,
    steps: [AtomicU64; MAX_INSTANCES],
    refuses: [AtomicU64; MAX_INSTANCES],
}

fn get(counts: &[AtomicU64; MAX_INSTANCES], i: usize) -> u64 {
    counts[i].load(Ordering::Relaxed)
}

fn sum(counts: &[AtomicU64; MAX_INSTANCES]) -> u64 {
    counts.iter().map(|c| c.load(Ordering::Relaxed)).sum()
}

#[derive(Debug, Clone)]
struct Cfg {
    shared: Arc<Shared>,
    initial: usize,
}

impl EnvConfig for Cfg {}

/// `T`: 0 simultaneous, 1 sequential; `R`: whether it refuses agents.
#[derive(Debug)]
struct Ledger<const T: u8, const R: bool> {
    shared: Arc<Shared>,
    id: usize,
    initial: usize,
    steps: u32,
    turn: usize,
    score: [i32; SLOTS],
}

fn slot(s: usize) -> AgentSlot {
    AgentSlot::new(u16::try_from(s).unwrap(), SLOTS).unwrap()
}

impl<const T: u8, const R: bool> HostEnvironment for Ledger<T, R> {
    const ENV_ID: &'static str = "test/Ledger-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = SLOTS;
    const TURNS: TurnStructure = if T == 0 {
        TurnStructure::Simultaneous
    } else {
        TurnStructure::Sequential
    };
    const VALIDATES: bool = true;
    const REFUSES_AGENTS: bool = R;
    type Config = Cfg;

    fn new(cfg: &Cfg) -> Result<Self, HostError> {
        let id = cfg.shared.next_id.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            shared: Arc::clone(&cfg.shared),
            id: usize::try_from(id).unwrap(),
            initial: cfg.initial,
            steps: 0,
            turn: 0,
            score: [0; SLOTS],
        })
    }

    fn initial_agents(cfg: &Cfg) -> usize {
        cfg.initial
    }

    fn observation_space(_: &Cfg) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![2],
            dtype: DType::I32,
            bounds: BoxBounds::Int {
                low: vec![-1_000_000],
                high: vec![1_000_000],
            },
        })
    }

    fn action_space(_: &Cfg) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: CHOICES })
    }

    fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {
        self.steps = 0;
        self.turn = 0;
        self.score = [0; SLOTS];
    }

    fn validate(&self, actions: &Actions<'_, ()>) -> Result<(), ActionReject> {
        if (0..SLOTS).any(|s| actions.discrete(slot(s), 0) == Some(REJECT)) {
            return Err(ActionReject {
                code: REJECT_CODE,
                reason: "the ledger rejected the call",
            });
        }
        Ok(())
    }

    fn refuse(&self, actions: &Actions<'_, ()>, out: &mut Refusals<'_>) {
        self.shared.refuses[self.id].fetch_add(1, Ordering::Relaxed);
        for s in 0..SLOTS {
            assert_ne!(actions.discrete(slot(s), 0), Some(PANIC), "refuse panicked");
            match actions.discrete(slot(s), 0) {
                Some(REFUSE) => {
                    let code = u32::try_from(s).unwrap() + REFUSE_CODE;
                    out.refuse(slot(s), RefuseCode::new(code).unwrap());
                }
                Some(MISUSE) => out.refuse(slot(2), RefuseCode::new(9).unwrap()),
                _ => {}
            }
        }
    }

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut HostOut<'_>) {
        self.shared.steps[self.id].fetch_add(1, Ordering::Relaxed);
        self.steps += 1;
        let mut turn_passes = true;
        for s in 0..self.initial {
            if T == 1 && s != self.turn {
                continue;
            }
            let reward = if actions.is_refused(slot(s)) {
                // The environment's rule: a refused agent pays a point and,
                // in turns, keeps the turn.
                turn_passes = false;
                -1
            } else {
                i32::from(actions.discrete(slot(s), 0) == Some(SCORE))
            };
            self.score[s] += reward;
            out.set_reward(slot(s), f32::from(i16::try_from(reward).unwrap()));
        }
        if T == 1 && turn_passes {
            self.turn = (self.turn + 1) % self.initial;
        }
    }

    fn observe(&self, agent: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>) {
        let values = [
            self.score[agent.index()],
            i32::try_from(self.steps).unwrap(),
        ];
        if let Some(leaf) = obs.leaf(0) {
            for (bytes, v) in leaf.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                bytes.copy_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn acting(&self, agent: AgentSlot) -> bool {
        T == 0 || agent.index() == self.turn
    }
}

type Pool<const T: u8, const R: bool> = HostPool<Ledger<T, R>, SLOTS>;

fn options() -> PoolOptions {
    let mut o = PoolOptions::new(
        NonZeroU32::new(u32::try_from(N).unwrap()).unwrap(),
        Seed::new(5),
    );
    // Two instances per chunk, so Sync spreads them over the workers.
    o.chunk_size = NonZeroU32::new(2).unwrap();
    o
}

fn pool_with<const T: u8, const R: bool>(
    shared: &Arc<Shared>,
    workers: usize,
    initial: usize,
    o: PoolOptions,
) -> Pool<T, R> {
    let cfg = Cfg {
        shared: Arc::clone(shared),
        initial,
    };
    let p = Pool::<T, R>::new(cfg, o).unwrap();
    if workers > 0 {
        p.with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
            .unwrap()
    } else {
        p
    }
}

fn pool<const T: u8, const R: bool>(shared: &Arc<Shared>, workers: usize) -> Pool<T, R> {
    pool_with::<T, R>(shared, workers, SLOTS, options())
}

/// Steps with `fill` everywhere except `special` (instance, slot, action).
fn step<const T: u8, const R: bool>(
    p: &mut Pool<T, R>,
    fill: i32,
    special: &[(usize, usize, i32)],
) -> Result<(), EngineError> {
    let mut a = vec![fill; N * SLOTS];
    for &(i, s, v) in special {
        a[i * SLOTS + s] = v;
    }
    p.step(ActionInput {
        discrete: &a,
        continuous: &[],
    })
    .map(|_| ())
}

fn flags(f: &[Bool8]) -> Vec<bool> {
    f.iter().map(|x| x.get()).collect()
}

/// Every output a test compares across modes, as bytes.
fn outputs(out: &Outputs) -> Vec<Vec<u8>> {
    let b = |f: &[Bool8]| f.iter().map(|x| u8::from(x.get())).collect::<Vec<_>>();
    vec![
        out.observation(0).unwrap().to_vec(),
        out.rewards()
            .iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect(),
        b(out.acting()),
        b(out.refused()),
        out.refuse_code()
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
    ]
}

fn refused_at(rows: &[(usize, usize)]) -> (Vec<bool>, Vec<u32>) {
    let mut refused = vec![false; N * SLOTS];
    let mut codes = vec![0u32; N * SLOTS];
    for &(i, s) in rows {
        refused[i * SLOTS + s] = true;
        codes[i * SLOTS + s] = REFUSE_CODE + u32::try_from(s).unwrap();
    }
    (refused, codes)
}

#[test]
fn test_host_refuse_one_agent_others_and_instances_step() {
    let mut runs = Vec::new();
    for workers in [0, 2] {
        let shared = Arc::new(Shared::default());
        let mut p = pool::<0, true>(&shared, workers);
        p.reset_all(Reseed::Keep).unwrap();
        step(
            &mut p,
            SCORE,
            &[(3, 1, REFUSE), (5, 0, REFUSE), (5, 2, REFUSE)],
        )
        .unwrap();
        let out = p.outputs().unwrap();
        let (refused, codes) = refused_at(&[(3, 1), (5, 0), (5, 2)]);
        assert_eq!(flags(out.refused()), refused, "workers={workers}");
        assert_eq!(out.refuse_code(), codes.as_slice());
        // Every instance stepped, the refusing ones included.
        assert_eq!(sum(&shared.steps), u64::try_from(N).unwrap());
        assert_eq!(sum(&shared.refuses), u64::try_from(N).unwrap());
        // A refused agent pays a point; every other agent scored one.
        for i in 0..N {
            for s in 0..SLOTS {
                let want = if refused[i * SLOTS + s] { -1.0 } else { 1.0 };
                assert_eq!(out.rewards()[i * SLOTS + s], want, "instance {i} slot {s}");
            }
        }
        assert!(!p.is_poisoned());
        runs.push(outputs(p.outputs().unwrap()));
    }
    // Inline and Sync give the same rows.
    assert_eq!(runs[0], runs[1]);
}

#[test]
fn test_host_refuse_rows_clear_on_the_next_call_and_on_resets() {
    for workers in [0, 2] {
        let shared = Arc::new(Shared::default());
        let mut p = pool::<0, true>(&shared, workers);
        p.reset_all(Reseed::Keep).unwrap();
        let clear = (vec![false; N * SLOTS], vec![0u32; N * SLOTS]);
        // A step without refusals after one with them.
        step(&mut p, SCORE, &[(2, 2, REFUSE)]).unwrap();
        step(&mut p, SCORE, &[]).unwrap();
        let out = p.outputs().unwrap();
        assert_eq!((flags(out.refused()), out.refuse_code().to_vec()), clear);
        // An explicit reset of other instances clears every row.
        step(&mut p, SCORE, &[(2, 2, REFUSE)]).unwrap();
        p.reset(&[0], Reseed::Keep).unwrap();
        let out = p.outputs().unwrap();
        assert_eq!((flags(out.refused()), out.refuse_code().to_vec()), clear);
        // So does a full reset.
        step(&mut p, SCORE, &[(2, 2, REFUSE)]).unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        let out = p.outputs().unwrap();
        assert_eq!((flags(out.refused()), out.refuse_code().to_vec()), clear);
    }
}

#[test]
fn test_host_refuse_rejected_call_keeps_rows_and_never_refuses() {
    for workers in [0, 2] {
        let shared = Arc::new(Shared::default());
        let mut p = pool::<0, true>(&shared, workers);
        p.reset_all(Reseed::Keep).unwrap();
        step(&mut p, SCORE, &[(4, 1, REFUSE)]).unwrap();
        let before = outputs(p.outputs().unwrap());
        let (steps, refuses) = (sum(&shared.steps), sum(&shared.refuses));
        // Whole-call validation wins over per-agent refusal.
        let r = step(&mut p, SCORE, &[(1, 0, REFUSE), (6, 2, REJECT)]);
        assert!(
            matches!(
                r,
                Err(EngineError::ActionRejected {
                    instance: 6,
                    code: REJECT_CODE,
                    ..
                })
            ),
            "{r:?}"
        );
        assert_eq!((sum(&shared.steps), sum(&shared.refuses)), (steps, refuses));
        assert_eq!(outputs(p.outputs().unwrap()), before, "workers={workers}");
        assert!(!p.is_poisoned());
    }
}

#[test]
fn test_host_refuse_non_acting_slot_poisons_and_skips_the_step() {
    for workers in [0, 2] {
        let shared = Arc::new(Shared::default());
        // Slot 2 is never alive.
        let mut p = pool_with::<0, true>(&shared, workers, 2, options());
        p.reset_all(Reseed::Keep).unwrap();
        let r = step(&mut p, SCORE, &[(5, 0, MISUSE), (3, 1, MISUSE)]);
        assert_eq!(
            r,
            Err(EngineError::RefusedNotActing {
                instance: 3,
                agent: 2
            }),
            "workers={workers}"
        );
        assert!(p.is_poisoned());
        assert_eq!(get(&shared.steps, 3), 0);
        assert_eq!(get(&shared.steps, 5), 0);
        assert_eq!(get(&shared.steps, 0), 1);
        assert_eq!(p.outputs().err(), Some(EngineError::Poisoned));
    }
}

#[test]
fn test_host_refuse_panic_poisons() {
    for workers in [0, 2] {
        let shared = Arc::new(Shared::default());
        let mut p = pool::<0, true>(&shared, workers);
        p.reset_all(Reseed::Keep).unwrap();
        let r = step(&mut p, SCORE, &[(1, 1, PANIC)]);
        assert_eq!(r, Err(EngineError::Panicked), "workers={workers}");
        assert!(p.is_poisoned());
    }
}

#[test]
fn test_host_refuse_sequential_refused_agent_keeps_the_turn() {
    let shared = Arc::new(Shared::default());
    let mut p = pool::<1, true>(&shared, 0);
    p.reset_all(Reseed::Keep).unwrap();
    // Slot 0 acts and is refused in instance 0: it acts again.
    step(&mut p, SCORE, &[(0, 0, REFUSE)]).unwrap();
    let out = p.outputs().unwrap();
    assert_eq!(flags(&out.acting()[0..SLOTS]), [true, false, false]);
    assert_eq!(flags(&out.acting()[SLOTS..2 * SLOTS]), [false, true, false]);
    assert!(out.refused()[0].get());
    // The refused agent's next action counts.
    step(&mut p, SCORE, &[]).unwrap();
    let out = p.outputs().unwrap();
    assert_eq!(flags(&out.acting()[0..SLOTS]), [false, true, false]);
    assert!(!p.is_poisoned());
}

#[test]
fn test_host_refuse_refusing_instance_advances_its_step_counter() {
    let shared = Arc::new(Shared::default());
    let mut o = options();
    o.max_episode_steps = NonZeroU32::new(2);
    let mut p = pool_with::<0, true>(&shared, 0, SLOTS, o);
    p.reset_all(Reseed::Keep).unwrap();
    // A refused call is a step: the second one hits the limit of 2.
    step(&mut p, SCORE, &[(0, 0, REFUSE)]).unwrap();
    step(&mut p, SCORE, &[(0, 0, REFUSE)]).unwrap();
    let out = p.outputs().unwrap();
    assert!(out.episode_truncated()[0].get());
    assert_eq!(out.episode_length()[0], 2.0);
}

#[test]
fn test_host_refuse_pending_instance_reports_no_refusal() {
    let shared = Arc::new(Shared::default());
    let mut o = options();
    o.max_episode_steps = NonZeroU32::new(1);
    o.auto_reset = AutoReset::NextStep;
    let mut p = pool_with::<0, true>(&shared, 0, SLOTS, o);
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p, SCORE, &[]).unwrap();
    // Every instance is pending: this call only resets them.
    let refuses = sum(&shared.refuses);
    step(&mut p, SCORE, &[(0, 0, REFUSE)]).unwrap();
    assert_eq!(sum(&shared.refuses), refuses);
    assert!(p.outputs().unwrap().refused().iter().all(|f| !f.get()));
}

#[test]
fn test_host_refuse_non_refusing_pool_has_no_refusal_outputs() {
    let shared = Arc::new(Shared::default());
    let mut p = pool::<0, false>(&shared, 2);
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p, SCORE, &[(3, 1, REFUSE)]).unwrap();
    let out = p.outputs().unwrap();
    assert_eq!(out.refused(), &[]);
    assert_eq!(out.refuse_code(), &[]);
    // `refuse` is never called, and the agent's action is not refused.
    assert_eq!(sum(&shared.refuses), 0);
    assert_eq!(out.rewards()[3 * SLOTS + 1], 0.0);
    assert!(!p.layout().refuses_agents);
}
