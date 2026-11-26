// SPDX-License-Identifier: MIT OR Apache-2.0

//! The live watch (ADR-0083 item 7) on a lane-wise pool allocates nothing
//! per step (SPEC 5.5): a served record is gathered from the columns into
//! the watch's own buffer, which `attach_watch` allocates once. Inline,
//! Sync and Pipelined; the counting allocator of `support/counting.rs` attributes
//! allocator calls by thread.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

#[path = "../benches/support/soa_cartpole.rs"]
mod soa_cartpole;
use soa_cartpole::SoaCartPole;
use tessera_rl_classic::cartpole::CartPoleConfig;
use tessera_rl_engine::{
    ActionInput, EnvPool, NoMasks, PipelinedOptions, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;

#[path = "support/counting.rs"]
mod counting;
use counting::window;

/// Allocation calls of the counted threads while `f` runs.
fn allocations(f: impl FnOnce()) -> u64 {
    window(f).1.calls
}

fn test_counter_detects_heap_use() {
    assert!(allocations(|| drop(std::hint::black_box(vec![1u8; 16]))) > 0);
}

fn run(workers: Option<usize>) {
    const N: u32 = 16;
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(8).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(5))
    };
    let pool = EnvPool::<SoaCartPole, NoMasks>::new(CartPoleConfig {}, options).unwrap();
    let mut pool = match workers {
        None => pool,
        Some(w) => pool
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap(),
    };
    pool.reset_all(Reseed::Keep).unwrap();
    let mut handle = pool.attach_watch().unwrap();
    let mut out = vec![0u8; handle.record_len()];
    let actions = [1i32; N as usize];
    let mut served = 0_u32;
    let n = allocations(|| {
        for t in 0..64_u32 {
            handle.request(t % N).unwrap();
            let input = ActionInput {
                discrete: &actions,
                continuous: &[],
            };
            pool.step(input).unwrap();
            if handle.take(&mut out).unwrap().is_some() {
                served += 1;
            }
        }
    });
    assert_eq!(n, 0, "workers {workers:?}");
    assert_eq!(served, 64, "every request is served at the next boundary");
}

/// Pipelined with 2 workers: each cycle sends both halves and receives
/// both; a request is served at the boundary of the half that holds it.
fn test_soa_watch_pipelined_does_not_allocate() {
    const N: u32 = 16;
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(8).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(5))
    };
    let mut pool = EnvPool::<SoaCartPole, NoMasks>::new(CartPoleConfig {}, options).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let mut pool = pool
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
        .unwrap();
    let mut handle = pool.attach_watch().unwrap();
    let mut out = vec![0u8; handle.record_len()];
    let halves = pool.halves();
    let actions: Vec<Vec<i32>> = halves
        .iter()
        .map(|r| vec![1; usize::try_from(r.end - r.start).unwrap()])
        .collect();
    let mut served = 0_u32;
    let n = allocations(|| {
        for t in 0..32_u32 {
            handle.request(t % N).unwrap();
            for (h, a) in (0_u32..).zip(&actions) {
                let token = pool.idle_token(h).unwrap();
                let input = ActionInput {
                    discrete: a,
                    continuous: &[],
                };
                pool.send(token, input).map_err(|e| e.error).unwrap();
            }
            drop(pool.recv().unwrap());
            drop(pool.recv().unwrap());
            if handle.take(&mut out).unwrap().is_some() {
                served += 1;
            }
        }
    });
    assert_eq!(n, 0, "Pipelined");
    assert_eq!(served, 32, "every request is served within its cycle");
}

fn test_soa_watch_inline_does_not_allocate() {
    run(None);
}

fn test_soa_watch_sync_does_not_allocate() {
    run(Some(2));
}

/// The scenarios this binary runs.
const SCENARIOS: usize = 4;

#[test]
fn test_soa_watch_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_counter_detects_heap_use",
                test_counter_detects_heap_use,
            ),
            (
                "test_soa_watch_inline_does_not_allocate",
                test_soa_watch_inline_does_not_allocate,
            ),
            (
                "test_soa_watch_sync_does_not_allocate",
                test_soa_watch_sync_does_not_allocate,
            ),
            (
                "test_soa_watch_pipelined_does_not_allocate",
                test_soa_watch_pipelined_does_not_allocate,
            ),
        ],
    );
}
