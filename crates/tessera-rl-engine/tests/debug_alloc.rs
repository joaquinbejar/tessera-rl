// SPDX-License-Identifier: MIT OR Apache-2.0

//! ADR-0055: with `debug-checks` and `TrackingAllocator` installed, an
//! allocation inside a chunk step poisons the pool and fails the call with
//! `AllocationInChunkStep`, in every mode. Counting is per thread and per
//! pool: a clean pool stepping beside an allocating one is never blamed.

#![cfg(all(not(loom), feature = "debug-checks"))]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

#[path = "debug_env/mod.rs"]
mod debug_env;

use core::num::{NonZeroU32, NonZeroUsize};

use debug_env::{ALLOC_BOX, ALLOC_GROW, ALLOC_ZEROED, ProbeConfig, ProbeEnv};
use tessera_rl_engine::debug::TrackingAllocator;
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AutoReset, EngineError, EnvPool, PipelinedOptions, PoolOptions,
    Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

const N: u32 = 8;

fn options() -> PoolOptions {
    PoolOptions {
        auto_reset: AutoReset::NextStep,
        chunk_size: NonZeroU32::new(2).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(5))
    }
}

fn inline(cfg: ProbeConfig) -> EnvPool<ProbeEnv> {
    EnvPool::new(cfg, options()).unwrap()
}

fn sync(cfg: ProbeConfig) -> EnvPool<ProbeEnv> {
    inline(cfg)
        .with_workers(SyncOptions::new(NonZeroUsize::new(3).unwrap()))
        .unwrap()
}

const ACTIONS: [i32; N as usize] = [0; N as usize];

fn input(discrete: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete,
        continuous: &[],
    }
}

fn at(step: u32, kind: u32) -> ProbeConfig {
    ProbeConfig {
        alloc_at: step,
        alloc_kind: kind,
        ..ProbeConfig::default()
    }
}

/// One allocation per instance at step 2: `alloc` and `alloc_zeroed` count
/// one each, a grown allocation two (`alloc` then `realloc`), and the
/// frees count nothing. The step before is clean.
#[test]
fn test_debug_alloc_inline_and_sync_count_alloc_zeroed_and_realloc_but_not_dealloc() {
    let n = u64::from(N);
    for (kind, count) in [(ALLOC_BOX, n), (ALLOC_ZEROED, n), (ALLOC_GROW, 2 * n)] {
        for mut p in [inline(at(2, kind)), sync(at(2, kind))] {
            p.reset_all(Reseed::Keep).unwrap();
            p.step(input(&ACTIONS)).unwrap();
            assert_eq!(
                p.step(input(&ACTIONS)).map(|_| ()),
                Err(EngineError::AllocationInChunkStep { count })
            );
            assert!(p.is_poisoned());
            assert_eq!(
                p.step(input(&ACTIONS)).map(|_| ()),
                Err(EngineError::Poisoned)
            );
        }
    }
}

/// Resets are not chunk steps: an allocation there is not counted, and a
/// clean pool steps without error.
#[test]
fn test_debug_alloc_reset_and_clean_steps_are_not_detected() {
    for cfg in [
        ProbeConfig {
            alloc_on_reset: true,
            ..ProbeConfig::default()
        },
        ProbeConfig::default(),
    ] {
        for mut p in [inline(cfg.clone()), sync(cfg.clone())] {
            p.reset_all(Reseed::Keep).unwrap();
            p.reset(&[1, 4], Reseed::Keep).unwrap();
            for _ in 0..5 {
                p.step(input(&ACTIONS)).unwrap();
            }
            assert!(!p.is_poisoned());
        }
    }
}

#[test]
fn test_debug_alloc_pipelined_recv_fails_and_poisons() {
    let mut p = inline(at(1, ALLOC_BOX))
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
        .unwrap();
    let [a, b] = p.reset_all(Reseed::Keep).unwrap();
    drop(b);
    let h = &p.halves()[0];
    let half = vec![0; usize::try_from(h.end - h.start).unwrap()];
    p.send(a, input(&half)).map_err(|e| e.error).unwrap();
    assert_eq!(
        p.recv().map(|_| ()),
        Err(EngineError::AllocationInChunkStep {
            count: u64::from(N / 2)
        })
    );
    assert!(p.is_poisoned());
}

#[test]
fn test_debug_alloc_async_recv_fails_and_poisons() {
    let mut p = inline(at(1, ALLOC_BOX))
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(3).unwrap(),
        ))
        .unwrap();
    drop(p.reset_all(Reseed::Keep).unwrap());
    p.send(&[0, 1, 2], input(&[0, 0, 0])).unwrap();
    assert_eq!(
        p.recv().map(|_| ()),
        Err(EngineError::AllocationInChunkStep { count: 3 })
    );
    assert!(p.is_poisoned());
}

/// The execution modes, for the two-pool attribution tests.
#[derive(Clone, Copy)]
enum Mode {
    Inline,
    Sync,
    Pipelined,
    Async,
}

const ROUNDS: u32 = 100;

fn alloc_cfg(dirty: bool) -> ProbeConfig {
    ProbeConfig {
        alloc_always: dirty,
        length: 1_000_000,
        ..ProbeConfig::default()
    }
}

/// Runs `ROUNDS` rounds of one step (a full reset first, which recovers a
/// poisoned pool) in `mode`; returns how many rounds failed with
/// `AllocationInChunkStep`. Any other error fails the test.
fn rounds(mode: Mode, dirty: bool) -> u32 {
    let mut failures = 0;
    let mut count = |r: Result<(), EngineError>| match r {
        Ok(()) => {}
        Err(EngineError::AllocationInChunkStep { .. }) => failures += 1,
        other => assert_eq!(other, Ok(()), "unexpected error"),
    };
    match mode {
        Mode::Inline | Mode::Sync => {
            let mut p = match mode {
                Mode::Inline => inline(alloc_cfg(dirty)),
                _ => sync(alloc_cfg(dirty)),
            };
            for _ in 0..ROUNDS {
                p.reset_all(Reseed::Keep).unwrap();
                count(p.step(input(&ACTIONS)).map(|_| ()));
            }
        }
        Mode::Pipelined => {
            let mut p = inline(alloc_cfg(dirty))
                .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
                .unwrap();
            let h = &p.halves()[0];
            let half = vec![0; usize::try_from(h.end - h.start).unwrap()];
            for _ in 0..ROUNDS {
                let [a, b] = p.reset_all(Reseed::Keep).unwrap();
                drop(b);
                p.send(a, input(&half)).map_err(|e| e.error).unwrap();
                count(p.recv().map(|_| ()));
            }
        }
        Mode::Async => {
            let mut p = inline(alloc_cfg(dirty))
                .async_mode(AsyncOptions::new(
                    NonZeroUsize::new(2).unwrap(),
                    NonZeroUsize::new(N as usize).unwrap(),
                ))
                .unwrap();
            for _ in 0..ROUNDS {
                let token = p.reset_all(Reseed::Keep).unwrap();
                p.send_batch(token, input(&ACTIONS))
                    .map_err(|e| e.error)
                    .unwrap();
                count(p.recv().map(|_| ()));
            }
        }
    }
    failures
}

/// Two pools of `mode` step at the same time in this binary, one
/// allocating at every step: every round of that pool fails, and the clean
/// one never does (no cross-pool blame).
fn two_pools(mode: Mode) {
    std::thread::scope(|s| {
        let dirty = s.spawn(move || rounds(mode, true));
        let clean = s.spawn(move || rounds(mode, false));
        assert_eq!(clean.join().unwrap(), 0, "the clean pool was blamed");
        assert_eq!(dirty.join().unwrap(), ROUNDS);
    });
}

#[test]
fn test_debug_alloc_two_inline_pools_only_the_allocating_one_fails() {
    two_pools(Mode::Inline);
}

#[test]
fn test_debug_alloc_two_sync_pools_only_the_allocating_one_fails() {
    two_pools(Mode::Sync);
}

#[test]
fn test_debug_alloc_two_pipelined_pools_only_the_allocating_one_fails() {
    two_pools(Mode::Pipelined);
}

#[test]
fn test_debug_alloc_two_async_pools_only_the_allocating_one_fails() {
    two_pools(Mode::Async);
}
