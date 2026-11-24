// SPDX-License-Identifier: MIT OR Apache-2.0

//! ADR-0073 item 4: a steady `snapshot_all` and `restore_all` loop
//! allocates nothing once its records hold their buffers. The engine
//! reserves `max_snapshot_len` before each `save` and reuses the records;
//! the walk's `save` and `load` stay within their capacity, so the count of
//! the whole loop (engine and environment) must be zero. The counting
//! allocator of `support/counting.rs` measures it.

#![cfg(feature = "host")]
#![expect(clippy::unwrap_used, reason = "a failure is the assertion")]

#[path = "support/counting.rs"]
mod counting;
mod host_walk;

use core::num::NonZeroU32;

use host_walk::{Walk, WalkConfig};
use tessera_rl_engine::{ActionInput, HostPool, HostRecord, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

#[test]
fn test_host_snapshot_steady_loop_allocates_nothing() {
    let n = 4;
    let mut options = PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(3));
    options.max_episode_steps = NonZeroU32::new(32);
    let mut pool = HostPool::<Walk, 1>::new(WalkConfig { length: 32 }, options).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let actions = vec![2i32; n as usize];
    for _ in 0..5 {
        pool.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        })
        .unwrap();
    }
    let mut records: Vec<HostRecord<1>> = (0..n).map(|_| HostRecord::new()).collect();
    // Warm up: the records reserve their buffers once.
    pool.snapshot_all(&mut records).unwrap();
    pool.restore_all(&records).unwrap();
    let (ok, tally) = counting::window(|| {
        let mut ok = true;
        for _ in 0..3 {
            ok &= pool.snapshot_all(&mut records).is_ok();
            ok &= pool.restore_all(&records).is_ok();
        }
        ok
    });
    assert!(ok);
    assert_eq!(tally.calls, 0, "{tally:?}");
}
