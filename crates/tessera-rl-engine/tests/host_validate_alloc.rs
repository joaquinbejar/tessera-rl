// SPDX-License-Identifier: MIT OR Apache-2.0

//! A host `validate` that allocates is charged to the host pool's tally,
//! never as `AllocationInChunkStep` (ADR-0065 item 2, ADR-0062 item 4).

#![cfg(all(not(loom), feature = "host", feature = "debug-checks"))]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod host_script;

use core::num::{NonZeroU32, NonZeroUsize};
use std::sync::atomic::Ordering;

use host_script::{NOTHING, SLOTS, Script, ScriptConfig, Shared};
use tessera_rl_engine::debug::TrackingAllocator;
use tessera_rl_engine::{ActionInput, HostPool, PoolOptions, Reseed, SyncOptions};
use tessera_rl_num::identity::Seed;

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn test_host_validate_allocations_are_charged_to_the_host_tally() {
    for workers in [0, 2] {
        let shared = Shared::new();
        let options = PoolOptions::new(NonZeroU32::new(4).unwrap(), Seed::new(2));
        let mut p =
            HostPool::<Script<0, true>, SLOTS>::new(&ScriptConfig::new(&shared), options).unwrap();
        if workers > 0 {
            p = p
                .with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
                .unwrap();
        }
        p.reset_all(Reseed::Keep).unwrap();
        let actions = vec![NOTHING; 4 * SLOTS];
        let input = || ActionInput {
            discrete: &actions,
            continuous: &[],
        };
        p.step(input()).unwrap();
        let before = p.allocations_in_host_steps();
        shared.alloc_validate.store(true, Ordering::Relaxed);
        p.step(input()).unwrap();
        assert!(
            p.allocations_in_host_steps() >= before + 4,
            "workers={workers}"
        );
    }
}
