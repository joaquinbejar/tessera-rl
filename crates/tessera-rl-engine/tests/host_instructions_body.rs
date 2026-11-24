// SPDX-License-Identifier: MIT OR Apache-2.0

//! Runs the `host_instructions` bench's own setup and measured steps
//! natively (the bench itself needs Valgrind): the host CartPole port
//! follows its physics, and the measured steps end the episodes they are
//! pinned to end.

#![cfg(feature = "host")]
#![expect(clippy::unwrap_used, reason = "test code unwraps")]

use core::num::NonZeroU32;

use tessera_rl_engine::{ActionInput, HostPool, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

#[path = "../benches/support/mod.rs"]
mod support;

use support::host_cartpole::{HostCartPole, HostCartPoleConfig, physics_step};
use support::host_setup::steady;

/// Episodes terminated in the 64 measured steps of `steady(n)`.
fn measured_terminations(n: u32) -> usize {
    let (mut pool, rows) = steady(n);
    let mut ended = 0;
    for actions in &rows {
        let out = pool
            .step(ActionInput {
                discrete: actions,
                continuous: &[],
            })
            .unwrap();
        ended += out.episode_terminated().iter().filter(|f| f.get()).count();
    }
    assert!(!pool.is_poisoned());
    ended
}

#[test]
fn test_host_instructions_measured_steps_n1_end_pinned_episodes() {
    assert_eq!(measured_terminations(1), 2);
}

#[test]
fn test_host_instructions_measured_steps_n1024_end_pinned_episodes() {
    assert_eq!(measured_terminations(1024), 2848);
}

#[test]
fn test_host_cartpole_steps_follow_its_physics() {
    let mut pool = HostPool::<HostCartPole, 1>::new(
        HostCartPoleConfig,
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
        // The reset draw lies in the first-party reset range.
        assert!(
            before.iter().all(|v| v.abs() <= 0.05),
            "instance {i}: {before:?}"
        );
        let expected = physics_step(before, *push == 1);
        for (got, want) in row(after).iter().zip(expected) {
            // The state is f64; the observation rounds it to f32, and the
            // check starts from the rounded observation.
            assert!((got - want).abs() < 1e-6, "instance {i}: {got} vs {want}");
        }
    }
    assert!(!pool.is_poisoned());
}
