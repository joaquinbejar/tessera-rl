// SPDX-License-Identifier: MIT OR Apache-2.0

//! A Sync pool with an ONNX league opponent never allocates per step in the
//! engine, on any thread (ADR-0088 section 11, erratum 2): the row gather,
//! the range check and the write into the action copy included. The policy
//! here is a Rust stand-in that does not allocate; ONNX Runtime's own
//! allocations during a run are outside the engine's guarantee (as for the
//! search evaluator). The counting allocator counts on every thread while
//! enabled, so this binary holds a single test.

use core::num::{NonZeroU32, NonZeroUsize};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use tessera_rl_classic::{PaddleDuel, PaddleDuelConfig};
use tessera_rl_engine::league::{OpponentPolicies, OpponentPolicyError, sealed};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, NoMasks, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_league::trueskill::Rating;
use tessera_rl_league::{
    League, LeagueConfig, Matchmaking, PolicyKind, PoolLeague, Population, elo,
};
use tessera_rl_num::identity::Seed;

static ON: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicU64 = AtomicU64::new(0);

struct Counting;

#[expect(
    unsafe_code,
    reason = "a counting global allocator for the zero-allocation test"
)]
// SAFETY: forwards to `System` unchanged; only counts while `ON`.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ON.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller's contract is `System::alloc`'s.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System` through `alloc`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if ON.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: as for `dealloc`.
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const HASH: [u8; 16] = [7; 16];

/// A non-allocating stand-in model: the first byte of the row, mod 3.
struct FirstByte;

impl sealed::Sealed for FirstByte {}

impl OpponentPolicies for FirstByte {
    fn serves(&self, content_hash: &[u8; 16]) -> bool {
        *content_hash == HASH
    }

    fn act(
        &mut self,
        _content_hash: &[u8; 16],
        rows: &[u8],
        n: usize,
        actions: &mut [i64],
    ) -> Result<(), OpponentPolicyError> {
        let row = rows.len() / n;
        for (j, a) in actions.iter_mut().enumerate() {
            *a = i64::from(rows.get(j * row).copied().unwrap_or(0) % 3);
        }
        Ok(())
    }
}

#[test]
fn test_onnx_league_step_does_not_allocate_in_the_engine_on_any_thread() {
    let mut population = Population::new();
    for k in [
        PolicyKind::Heuristic,
        PolicyKind::External { tag: 3 },
        PolicyKind::Onnx { content_hash: HASH },
    ] {
        population
            .add(k, elo::INITIAL, Rating::initial(), 0)
            .unwrap();
    }
    let config = LeagueConfig {
        matchmaking: Matchmaking::Uniform,
        ..LeagueConfig::default()
    };
    let league = PoolLeague::new(League::new(config, population).unwrap()).unwrap();
    let view = league.view().unwrap();
    let options = PoolOptions {
        auto_reset: AutoReset::SameStep,
        max_episode_steps: NonZeroU32::new(3),
        chunk_size: NonZeroU32::new(8).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(1))
    };
    let cfg = PaddleDuelConfig {
        agents: 2,
        ..PaddleDuelConfig::default()
    };
    let mut pool = EnvPool::<PaddleDuel, NoMasks>::new(cfg, options)
        .unwrap()
        .with_league(&[1], view, Box::new(league))
        .unwrap()
        .with_onnx_opponents(Box::new(FirstByte))
        .unwrap()
        .with_workers(SyncOptions::new(NonZeroUsize::new(4).unwrap()))
        .unwrap();
    let actions = vec![0i32; 128];
    // Warm up: the first epoch may initialize thread-local runtime state.
    pool.reset_all(Reseed::Keep).unwrap();
    pool.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap();
    ON.store(true, Ordering::Relaxed);
    // 30 steps: 10 episodes per instance, so draws and full rings
    // (capacity 8) both happen in the window.
    for _ in 0..30 {
        let _ = std::hint::black_box(pool.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        }));
    }
    let _ = std::hint::black_box(pool.assignment());
    ON.store(false, Ordering::Relaxed);
    assert_eq!(
        COUNT.load(Ordering::Relaxed),
        0,
        "engine allocations per step with ONNX opponents"
    );
}
