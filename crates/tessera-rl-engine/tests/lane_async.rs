// SPDX-License-Identifier: MIT OR Apache-2.0

//! Async mode never takes the lane path (ADR-0071 decision 3): a `Walk`
//! Async pool runs no lane step, and its records equal a scalar Sync
//! pool's after every round. Alone in its binary, so that the process-wide
//! lane-step counter sees no other test.

#![cfg(not(loom))]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod lanes;

use core::num::{NonZeroU32, NonZeroUsize};
use core::sync::atomic::Ordering;

use lanes::{LANE_CALLS, Plain, Walk, WalkConfig};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AsyncPool, AutoReset, EnvPool, NoMasks, PoolOptions, Record, Reseed,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 64;

fn options() -> PoolOptions {
    PoolOptions {
        chunk_size: NonZeroU32::new(16).unwrap(),
        max_episode_steps: NonZeroU32::new(9),
        auto_reset: AutoReset::SameStep,
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(11))
    }
}

fn action(t: u32, i: u32) -> i32 {
    i32::try_from((i * 7 + t * 5 + (t >> 2)) % 3).unwrap()
}

#[test]
fn test_lane_path_async_stays_scalar_and_matches_sync() {
    // Control: the Sync lane pool does run lane steps.
    let mut sync = EnvPool::<Walk>::new(WalkConfig {}, options()).unwrap();
    sync.reset_all(Reseed::Keep).unwrap();
    let all: Vec<u32> = (0..N).collect();
    let acts: Vec<i32> = all.iter().map(|&i| action(0, i)).collect();
    sync.step(ActionInput {
        discrete: &acts,
        continuous: &[],
    })
    .unwrap();
    assert!(
        LANE_CALLS.load(Ordering::Relaxed) > 0,
        "the control took no lane step"
    );

    let mut asy: AsyncPool<Walk, NoMasks> = EnvPool::<Walk>::new(WalkConfig {}, options())
        .unwrap()
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(N as usize).unwrap(),
        ))
        .unwrap();
    let mut plain = EnvPool::<Plain>::new(WalkConfig {}, options()).unwrap();
    drop(asy.reset_all(Reseed::Keep).unwrap());
    plain.reset_all(Reseed::Keep).unwrap();
    let before = LANE_CALLS.load(Ordering::Relaxed);
    for t in 0..300 {
        let acts: Vec<i32> = all.iter().map(|&i| action(t, i)).collect();
        asy.send(
            &all,
            ActionInput {
                discrete: &acts,
                continuous: &[],
            },
        )
        .unwrap();
        let mut got = 0;
        while got < all.len() {
            got += asy.recv().unwrap().ids().count();
        }
        plain
            .step(ActionInput {
                discrete: &acts,
                continuous: &[],
            })
            .unwrap();
        let mut recs = vec![sync.snapshot(0).unwrap(); N as usize];
        asy.snapshot_all(&mut recs).unwrap();
        let a: Vec<Vec<u8>> = recs.iter().map(Record::to_bytes).collect();
        let b: Vec<Vec<u8>> = (0..N)
            .map(|i| plain.snapshot(i).unwrap().to_bytes())
            .collect();
        assert_eq!(a, b, "async at {t}");
    }
    assert_eq!(
        LANE_CALLS.load(Ordering::Relaxed),
        before,
        "Async took the lane path"
    );
}
