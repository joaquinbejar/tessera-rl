// SPDX-License-Identifier: MIT OR Apache-2.0

//! Action masks and the illegal-action policy (ADR-0014, ADR-0028).

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod common;

use core::num::NonZeroU32;

use common::{CountConfig, Counter};
use tessera_rl_core::Bool8;
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, EnvPool, IllegalActionPolicy, Masked, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;

fn options(policy: IllegalActionPolicy) -> PoolOptions {
    PoolOptions {
        illegal_action: policy,
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(8))
    }
}

fn masked(policy: IllegalActionPolicy) -> EnvPool<Counter, Masked> {
    let cfg = CountConfig {
        length: 2,
        ..CountConfig::default()
    };
    let mut p = EnvPool::with_masks(cfg, options(policy)).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn step<M: tessera_rl_engine::MaskProvider<Counter>>(
    p: &mut EnvPool<Counter, M>,
    act: &[i32],
) -> Vec<f32> {
    p.step(ActionInput {
        discrete: act,
        continuous: &[],
    })
    .unwrap()
    .rewards()
    .to_vec()
}

#[test]
fn test_masks_are_written_for_the_current_state() {
    let p = masked(IllegalActionPolicy::Noop);
    let t = Bool8::TRUE;
    assert_eq!(p.outputs().masks(), [t, t, Bool8::FALSE, t, t, t]);
}

#[test]
fn test_illegal_action_replaced_by_noop_and_counted() {
    let (mut legal, mut illegal) = (
        masked(IllegalActionPolicy::Noop),
        masked(IllegalActionPolicy::Noop),
    );
    // Agent 0's action 2 is illegal: it behaves like the no-op 0.
    let a = step(&mut legal, &[0, 1]);
    let b = step(&mut illegal, &[2, 1]);
    assert_eq!(a, b);
    assert_eq!(illegal.header(0).unwrap().agents[0].illegal_actions, 1);
    assert_eq!(illegal.header(0).unwrap().agents[1].illegal_actions, 0);
    // Counted in the episode stats when the episode ends.
    step(&mut illegal, &[2, 2]);
    assert_eq!(illegal.outputs().episode_illegal_actions(), [2.0]);
}

#[test]
fn test_penalize_adds_the_penalty_to_the_illegal_agent() {
    let mut noop = masked(IllegalActionPolicy::Noop);
    let mut penal = masked(IllegalActionPolicy::Penalize(-1.5));
    let a = step(&mut noop, &[2, 1]);
    let b = step(&mut penal, &[2, 1]);
    assert_eq!(b[0], a[0] - 1.5);
    assert_eq!(b[1], a[1]);
}

#[test]
fn test_pools_without_masks_have_no_mask_buffer_and_accept_the_action() {
    let cfg = CountConfig {
        length: 2,
        ..CountConfig::default()
    };
    let mut p = EnvPool::<Counter>::new(cfg, options(IllegalActionPolicy::Noop)).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(p.outputs().masks(), []);
    step(&mut p, &[2, 1]);
    assert_eq!(p.header(0).unwrap().agents[0].illegal_actions, 0);
}

#[test]
fn test_non_finite_penalty_is_rejected() {
    let cfg = CountConfig::default();
    let bad = options(IllegalActionPolicy::Penalize(f32::NAN));
    assert!(matches!(
        EnvPool::<Counter, Masked>::with_masks(cfg, bad),
        Err(EngineError::Options { .. })
    ));
}

#[test]
fn test_reset_clears_illegal_counters() {
    let mut p = masked(IllegalActionPolicy::Noop);
    step(&mut p, &[2, 0]);
    p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(p.header(0).unwrap().agents[0].illegal_actions, 0);
}
