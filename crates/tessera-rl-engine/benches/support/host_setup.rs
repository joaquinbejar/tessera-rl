// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `host_instructions` setup, shared with the native
//! `host_instructions_body` test so that the bench's own inputs run without
//! Valgrind too.

use core::num::NonZeroU32;

use tessera_rl_engine::{ActionInput, HostPool, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

use super::host_cartpole::{HostCartPole, HostCartPoleConfig};

/// The measured pool.
pub type Pool = HostPool<HostCartPole, 1>;

/// Settling steps before the measurement, as `cartpole_pool_steady`.
pub const SETTLE_STEPS: usize = 300;

/// Measured steps, as `cartpole_pool_steady`.
pub const MEASURED_STEPS: usize = 64;

fn options(n: u32) -> PoolOptions {
    let mut o = PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(1));
    o.max_episode_steps = NonZeroU32::new(500);
    o
}

/// xorshift64: a fixed pseudo-random action stream, independent of the engine.
fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// A host CartPole pool of `n` instances in steady state, and the next
/// `MEASURED_STEPS` batches of actions, all drawn here.
#[must_use]
pub fn steady(n: u32) -> (Pool, Vec<Vec<i32>>) {
    let mut pool = Pool::new(&HostCartPoleConfig, options(n)).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut row = || -> Vec<i32> {
        (0..n)
            .map(|_| i32::from(xorshift(&mut seed) & 1 == 1))
            .collect()
    };
    for _ in 0..SETTLE_STEPS {
        let actions = row();
        pool.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        })
        .unwrap();
    }
    let rows = (0..MEASURED_STEPS).map(|_| row()).collect();
    (pool, rows)
}
