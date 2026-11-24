// SPDX-License-Identifier: MIT OR Apache-2.0

//! The epoch barrier (ADR-0081): episode statistics reduced pairwise over
//! instance order at any worker count, schedule and chunk size; freeze;
//! the quiescence rules of the split-phase modes; partials are not
//! context.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers unwrap and index: a failure is the assertion"
)]

mod common;

use core::num::{NonZeroU32, NonZeroUsize};

use common::{CountConfig, Counter};
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, NoAdaptive, ObsLeaf, Observation, RecordHeader, SpaceError,
    StepOut, Tier,
};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, EngineError, EnvPool, EpisodeStats, Masked, PipelinedOptions,
    PoolOptions, Reseed, Schedule, SyncOptions,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::fp;
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

type Pool = EnvPool<Counter, Masked>;

const N: u32 = 13;

fn cfg() -> CountConfig {
    CountConfig {
        length: 3,
        ..CountConfig::default()
    }
}

fn options(chunk: u32, schedule: Schedule) -> PoolOptions {
    PoolOptions {
        chunk_size: NonZeroU32::new(chunk).unwrap(),
        schedule,
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(11))
    }
}

fn pool(workers: usize, chunk: u32, schedule: Schedule) -> Pool {
    let mut p = Pool::with_masks(cfg(), options(chunk, schedule)).unwrap();
    if workers > 0 {
        p = p
            .with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
            .unwrap();
    }
    p.reset_all(Reseed::Keep).unwrap();
    p
}

/// Actions that differ per instance and step, so returns differ.
fn actions(p: &Pool, k: usize) -> Vec<i32> {
    let n = p.outputs().unwrap().rewards().len();
    (0..n)
        .map(|r| i32::from((r + k).is_multiple_of(3)))
        .collect()
}

fn step(p: &mut Pool, k: usize) {
    let a = actions(p, k);
    p.step(ActionInput {
        discrete: &a,
        continuous: &[],
    })
    .unwrap();
}

/// The reference: per-instance partials rebuilt from the outputs of every
/// step, and the pairwise tree over instance order (split at the floor
/// half, left first), as ADR-0081 item 3 states it.
#[derive(Clone, Copy, Default)]
struct Partial {
    count: u64,
    ret: f64,
    len: f64,
    sq: f64,
}

fn tree(parts: &[Partial]) -> (f64, f64, f64) {
    match parts {
        [] => (0.0, 0.0, 0.0),
        [one] => (one.ret, one.len, one.sq),
        _ => {
            let (left, right) = parts.split_at(parts.len() / 2);
            let (lo, hi) = (tree(left), tree(right));
            (lo.0 + hi.0, lo.1 + hi.1, lo.2 + hi.2)
        }
    }
}

fn observe(p: &Pool, partials: &mut [Partial]) {
    let out = p.outputs().unwrap();
    for (i, part) in partials.iter_mut().enumerate() {
        if out.episode_terminated()[i].get() || out.episode_truncated()[i].get() {
            let r = f64::from(out.episode_return()[i]);
            part.count += 1;
            part.ret += r;
            part.len += f64::from(out.episode_length()[i]);
            part.sq += r * r;
        }
    }
}

type Bits = (u64, u64, u64, u64, u64);

fn run(workers: usize, chunk: u32, schedule: Schedule) -> (EpisodeStats, Bits) {
    let mut p = pool(workers, chunk, schedule);
    let mut partials = vec![Partial::default(); N as usize];
    for k in 0..20 {
        step(&mut p, k);
        observe(&p, &mut partials);
    }
    let report = p.epoch_barrier().unwrap();
    let (ret, len, sq) = tree(&partials);
    let count: u64 = partials.iter().map(|x| x.count).sum();
    let want = (count, ret.to_bits(), len.to_bits(), sq.to_bits(), 0);
    (report.episodes, want)
}

fn bits(s: &EpisodeStats) -> Bits {
    (
        s.count,
        s.sum_return.to_bits(),
        s.sum_length.to_bits(),
        s.sum_sq_return.to_bits(),
        s.aborted,
    )
}

#[test]
fn test_epoch_barrier_episode_stats_equal_the_instance_order_tree_at_any_worker_count() {
    let (inline, want) = run(0, 2, Schedule::Static);
    assert!(inline.count > N.into(), "episodes ended: {}", inline.count);
    assert_eq!(bits(&inline), want);
    for (w, chunk, s) in [
        (1, 2, Schedule::Static),
        (2, 2, Schedule::Static),
        (7, 2, Schedule::Static),
        (3, 2, Schedule::Dynamic),
        (7, 5, Schedule::Dynamic),
        (2, 3, Schedule::Static),
    ] {
        let (got, reference) = run(w, chunk, s);
        assert_eq!(bits(&got), reference, "{w} workers, chunk {chunk}, {s:?}");
        assert_eq!(
            bits(&got),
            bits(&inline),
            "{w} workers, chunk {chunk}, {s:?}"
        );
    }
    assert!(inline.mean_return().is_some());
    assert!(inline.var_return().unwrap() >= 0.0);
}

#[test]
fn test_epoch_barrier_clears_partials_and_freeze_counts_nothing() {
    let mut p = pool(2, 2, Schedule::Static);
    for k in 0..6 {
        step(&mut p, k);
    }
    assert!(p.epoch_barrier().unwrap().episodes.count > 0);
    assert_eq!(p.epoch_barrier().unwrap().episodes.count, 0, "cleared");
    p.freeze_episode_stats(true);
    for k in 0..6 {
        step(&mut p, k);
    }
    assert_eq!(p.epoch_barrier().unwrap().episodes.count, 0, "frozen");
    p.freeze_adaptive(false);
    // Nothing was added while frozen: a barrier right after unfreezing
    // reports no episode.
    assert_eq!(
        p.epoch_barrier().unwrap().episodes.count,
        0,
        "nothing added"
    );
    for k in 0..6 {
        step(&mut p, k);
    }
    assert!(p.epoch_barrier().unwrap().episodes.count > 0, "unfrozen");
}

#[test]
fn test_epoch_barrier_partials_restart_empty_after_a_context_restore() {
    let mut p = pool(0, 2, Schedule::Static);
    let context = p.pool_context().unwrap();
    for k in 0..6 {
        step(&mut p, k);
    }
    p.restore_pool_context(context, None).unwrap();
    assert_eq!(p.epoch_barrier().unwrap().episodes.count, 0);
}

#[test]
fn test_epoch_barrier_pipelined_needs_both_halves_joined() {
    let mut p = Pool::with_masks(cfg(), options(2, Schedule::Static))
        .unwrap()
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
        .map_err(|e| e.error)
        .unwrap();
    let halves = p.halves();
    let [ta, tb] = p.reset_all(Reseed::Keep).unwrap();
    let rows = usize::try_from(halves[0].end - halves[0].start).unwrap() * 2;
    p.send(
        ta,
        ActionInput {
            discrete: &vec![1; rows],
            continuous: &[],
        },
    )
    .map_err(|e| e.error)
    .unwrap();
    assert!(matches!(p.epoch_barrier(), Err(EngineError::NotQuiescent)));
    let ta = p.recv().unwrap();
    drop((ta, tb));
    let report = p.epoch_barrier().unwrap();
    assert_eq!(report.episodes.count, 0, "one step of length-3 episodes");
}

#[test]
fn test_epoch_barrier_async_refused_unless_every_subsystem_is_frozen() {
    let mut a = Pool::with_masks(cfg(), options(2, Schedule::Static))
        .unwrap()
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(4).unwrap(),
        ))
        .unwrap();
    drop(a.reset_all(Reseed::Keep).unwrap());
    assert!(matches!(
        a.epoch_barrier(),
        Err(EngineError::AdaptiveInAsync)
    ));
    a.freeze_episode_stats(true).unwrap();
    assert_eq!(a.epoch_barrier().unwrap().episodes.count, 0);
}

#[derive(EnvConfig, Debug, Clone)]
struct TenthConfig {}

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct TenthState {
    steps: u32,
}

#[derive(Observation)]
struct TenthObs {
    #[space(box(low = 0, high = 1000, shape = [1]))]
    steps: ObsLeaf<u32>,
}

/// Rewards of 0.1 and episodes of varied length: sums that round, so the
/// rounding mode would show in their bits.
struct Tenth;

impl Environment for Tenth {
    type State = TenthState;
    type Config = TenthConfig;
    type Obs = TenthObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &TenthConfig) -> Result<Space, SpaceError> {
        TenthObs::space(cfg)
    }

    fn action_space(_: &TenthConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(_: &TenthConfig, _: &mut StepRng, state: &mut TenthState) {
        state.steps = 0;
    }

    fn step(
        _: &TenthConfig,
        state: &mut TenthState,
        _: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        state.steps += 1;
        out.set_reward(AgentSlot::FIRST, 0.1);
        if state.steps >= 2 + rng.next_u32() % 5 {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &TenthConfig,
        state: &TenthState,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, TenthObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(TenthObs::STEPS) {
            leaf.set(0, state.steps);
        }
    }
}

fn tenth_report(hostile: bool) -> EpisodeStats {
    let options = PoolOptions::new(NonZeroU32::new(37).unwrap(), Seed::new(5));
    let mut p = EnvPool::<Tenth, tessera_rl_engine::NoMasks>::new(TenthConfig {}, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    for _ in 0..40 {
        p.step(ActionInput {
            discrete: &[0; 37],
            continuous: &[],
        })
        .unwrap();
    }
    if hostile {
        // Round toward zero and flush-to-zero on the caller (ADR-0037).
        fp::hostile::run(
            |c| fp::hostile::toward_zero(fp::hostile::flush_to_zero(c)),
            || p.epoch_barrier(),
        )
        .0
        .unwrap()
        .episodes
    } else {
        p.epoch_barrier().unwrap().episodes
    }
}

#[test]
fn test_epoch_barrier_report_is_bit_identical_under_hostile_caller_fp_controls() {
    let canonical = tenth_report(false);
    let hostile = tenth_report(true);
    assert!(canonical.count > 37, "episodes ended: {}", canonical.count);
    // The sums of tenths round: the test would see a rounding-mode change.
    assert_ne!(canonical.sum_return.fract(), 0.0);
    assert_eq!(bits(&hostile), bits(&canonical));
}
