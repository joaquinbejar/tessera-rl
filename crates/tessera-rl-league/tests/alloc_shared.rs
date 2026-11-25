// SPDX-License-Identifier: MIT OR Apache-2.0

//! Two Sync pools of one shared league never allocate per step, on any
//! thread (ADR-0088 section 11, #384 head E1): the shared league is reached
//! only at barriers. The counting allocator counts on every thread while
//! enabled, so this binary holds a single test.

use core::num::{NonZeroU32, NonZeroUsize};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use tessera_rl_classic::{PaddleDuel, PaddleDuelConfig};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, NoMasks, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_league::trueskill::{Params, Rating};
use tessera_rl_league::{
    Matchmaking, PolicyKind, Population, RoleConfig, SharedConfig, SharedLeague, elo,
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

#[test]
fn test_shared_league_step_does_not_allocate_on_any_thread() {
    let mut population = Population::new();
    for k in [PolicyKind::Heuristic, PolicyKind::External { tag: 3 }] {
        population
            .add(k, elo::INITIAL, Rating::initial(), 0)
            .unwrap();
    }
    let config = SharedConfig {
        elo_k: 16.0,
        trueskill: Params::default(),
        rate_population: false,
        roles: vec![RoleConfig::new(Matchmaking::Uniform); 2],
    };
    let league = SharedLeague::new(config, population).unwrap();
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
    let mut pools: Vec<EnvPool<PaddleDuel, NoMasks>> = (0..2)
        .map(|role| {
            EnvPool::<PaddleDuel, NoMasks>::new(cfg.clone(), options.clone())
                .unwrap()
                .with_league(
                    &[1],
                    league.view(role).unwrap(),
                    Box::new(league.member(role).unwrap()),
                )
                .unwrap()
                .with_workers(SyncOptions::new(NonZeroUsize::new(2).unwrap()))
                .unwrap()
        })
        .collect();
    let actions = vec![0i32; 128];
    let input = || ActionInput {
        discrete: &actions,
        continuous: &[],
    };
    // Warm up: the first epoch may initialize thread-local runtime state.
    for p in &mut pools {
        p.reset_all(Reseed::Keep).unwrap();
        p.step(input()).unwrap();
    }
    ON.store(true, Ordering::Relaxed);
    for _ in 0..30 {
        for p in &mut pools {
            let _ = std::hint::black_box(p.step(input()));
        }
    }
    ON.store(false, Ordering::Relaxed);
    assert_eq!(
        COUNT.load(Ordering::Relaxed),
        0,
        "allocations per step with a shared league"
    );
}
