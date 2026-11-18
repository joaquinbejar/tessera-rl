// SPDX-License-Identifier: MIT OR Apache-2.0

//! Runs the `host_instructions` bench's environment natively (the bench
//! itself needs Valgrind): the host CartPole port follows the first-party
//! physics, and a pool of it steps without poisoning.

#![cfg(feature = "host")]

use core::num::NonZeroU32;

use tessera_rl_classic::CartPoleConfig;
use tessera_rl_classic::cartpole::physics;
use tessera_rl_engine::{ActionInput, HostPool, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

#[path = "../benches/support/host_cartpole.rs"]
mod host_cartpole;

use host_cartpole::HostCartPole;

#[test]
fn test_host_cartpole_steps_follow_the_first_party_physics() {
    let mut pool = HostPool::<HostCartPole, 1>::new(
        &CartPoleConfig::default(),
        PoolOptions::new(NonZeroU32::new(4).unwrap(), Seed::new(1)),
    )
    .unwrap();
    let first = pool
        .reset_all(Reseed::Keep)
        .unwrap()
        .observation(0)
        .unwrap()
        .to_vec();
    let actions = [1, 0, 1, 0];
    let out = pool
        .step(ActionInput {
            discrete: &actions,
            continuous: &[],
        })
        .unwrap();
    assert!(
        out.rewards()
            .iter()
            .all(|r| r.to_bits() == 1.0_f32.to_bits())
    );
    let after = out.observation(0).unwrap();
    for (i, push) in actions.iter().enumerate() {
        let row = |bytes: &[u8]| -> [f64; 4] {
            core::array::from_fn(|k| {
                let at = (i * 4 + k) * 4;
                f64::from(f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()))
            })
        };
        let before = row(&first);
        let expected = physics::step(before, *push == 1);
        for (got, want) in row(after).iter().zip(expected) {
            // The state is f64; the observation rounds it to f32, and the
            // check starts from the rounded observation.
            assert!((got - want).abs() < 1e-6, "instance {i}: {got} vs {want}");
        }
    }
    assert!(!pool.is_poisoned());
}
