// SPDX-License-Identifier: MIT OR Apache-2.0

//! `DynHostPool` mirrors `HostPool` (ADR-0065 item 3): every method, called
//! through the trait object, returns what the concrete pool returns.

#![cfg(feature = "host")]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod host_support;

use core::num::{NonZeroU32, NonZeroUsize};

use host_support::{AGENTS, Book, BookConfig};
use tessera_rl_core::HostEnvironment;
use tessera_rl_core::envs::HostRegistration;
use tessera_rl_engine::{
    ActionInput, DynHostPool, HostPool, Outputs, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;

type Pool = HostPool<Book, { Book::MAX_AGENTS }>;

fn pool(workers: usize) -> Pool {
    let options = PoolOptions::new(NonZeroU32::new(4).unwrap(), Seed::new(11));
    let p = Pool::new(&BookConfig::default(), options).unwrap();
    if workers == 0 {
        p
    } else {
        p.with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
            .unwrap()
    }
}

fn bytes(out: &Outputs) -> Vec<u8> {
    let mut v = out.observation(0).unwrap().to_vec();
    v.extend(out.rewards().iter().flat_map(|r| r.to_bits().to_le_bytes()));
    v.extend(out.masks().iter().map(|m| u8::from(m.get())));
    v.extend(out.pending_reset().iter().map(|m| u8::from(m.get())));
    v
}

/// The accessors, by value, in `HostPool`'s order.
fn accessors(
    registration: &HostRegistration,
    options: &PoolOptions,
    rest: (usize, bool, (usize, usize), bool, u64, u64),
) -> String {
    format!("{registration:?} {options:?} {rest:?}")
}

#[test]
fn test_host_dyn_pool_mirrors_every_method() {
    for workers in [0, 2] {
        let mut concrete = pool(workers);
        let mut boxed: Box<dyn DynHostPool> = Box::new(pool(workers));
        assert_eq!(boxed.registration(), &HostRegistration::of::<Book>());
        assert_eq!(concrete.registration(), boxed.registration());
        assert_eq!(concrete.layout(), boxed.layout());
        let actions = vec![1i32; 4 * AGENTS * 2];
        for call in 0..7 {
            let (a, b) = match call {
                0 => (
                    bytes(concrete.reset_all(Reseed::Keep).unwrap()),
                    bytes(boxed.reset_all(Reseed::Keep).unwrap()),
                ),
                3 => (
                    bytes(concrete.reset(&[2, 0], Reseed::Base(Seed::new(3))).unwrap()),
                    bytes(boxed.reset(&[2, 0], Reseed::Base(Seed::new(3))).unwrap()),
                ),
                _ => {
                    let input = || ActionInput {
                        discrete: &actions,
                        continuous: &[],
                    };
                    (
                        bytes(concrete.step(input()).unwrap()),
                        bytes(boxed.step(input()).unwrap()),
                    )
                }
            };
            assert_eq!(a, b, "call {call} workers={workers}");
            assert_eq!(
                bytes(HostPool::outputs(&concrete).unwrap()),
                bytes(boxed.outputs().unwrap())
            );
            assert_eq!(
                accessors(
                    concrete.registration(),
                    concrete.options(),
                    (
                        concrete.workers(),
                        concrete.pinned_workers(),
                        concrete.action_heads(),
                        concrete.is_poisoned(),
                        concrete.aborts(),
                        concrete.host_fp_resets(),
                    ),
                ),
                accessors(
                    boxed.registration(),
                    boxed.options(),
                    (
                        boxed.workers(),
                        boxed.pinned_workers(),
                        boxed.action_heads(),
                        boxed.is_poisoned(),
                        boxed.aborts(),
                        boxed.host_fp_resets(),
                    ),
                )
            );
            #[cfg(feature = "debug-checks")]
            assert_eq!(
                concrete.allocations_in_host_steps(),
                boxed.allocations_in_host_steps()
            );
        }
        // An error is the same error.
        let bad = ActionInput {
            discrete: &actions[..2],
            continuous: &[],
        };
        assert_eq!(
            concrete.step(bad).map(|_| ()).unwrap_err(),
            boxed.step(bad).map(|_| ()).unwrap_err()
        );
    }
}
