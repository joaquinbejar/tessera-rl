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
    assert_eq!(p.outputs().unwrap().masks(), [t, t, Bool8::FALSE, t, t, t]);
}

#[cfg(not(feature = "debug-checks"))]
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
    assert_eq!(illegal.outputs().unwrap().episode_illegal_actions(), [2.0]);
}

#[cfg(not(feature = "debug-checks"))]
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
    assert_eq!(p.outputs().unwrap().masks(), []);
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

#[cfg(not(feature = "debug-checks"))]
#[test]
fn test_reset_clears_illegal_counters() {
    let mut p = masked(IllegalActionPolicy::Noop);
    step(&mut p, &[2, 0]);
    p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(p.header(0).unwrap().agents[0].illegal_actions, 0);
}

fn masked_with(cfg: CountConfig, policy: IllegalActionPolicy) -> EnvPool<Counter, Masked> {
    let mut p = EnvPool::with_masks(cfg, options(policy)).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

#[cfg(not(feature = "debug-checks"))]
#[test]
fn test_negative_zero_reward_survives_without_illegal_actions() {
    let cfg = CountConfig {
        negative_zero_reward: true,
        length: 9,
        ..CountConfig::default()
    };
    let mut p = masked_with(cfg, IllegalActionPolicy::Penalize(0.0));
    let rewards = step(&mut p, &[0, 1]);
    assert!(rewards.iter().all(|r| r.to_bits() == (-0.0f32).to_bits()));
}

#[cfg(not(feature = "debug-checks"))]
#[test]
fn test_panic_in_noop_poisons_and_recovers() {
    let cfg = CountConfig {
        panic_in_noop: true,
        length: 9,
        ..CountConfig::default()
    };
    let mut p = masked_with(cfg, IllegalActionPolicy::Noop);
    let err = p
        .step(ActionInput {
            discrete: &[2, 0],
            continuous: &[],
        })
        .map(|_| ());
    assert_eq!(err, Err(EngineError::Panicked));
    assert!(p.is_poisoned());
    // Records of a poisoned pool are not readable (they may be partly
    // written; there is no rollback).
    assert!(p.header(0).is_none());
    assert_eq!(
        p.step(ActionInput {
            discrete: &[0, 0],
            continuous: &[]
        })
        .map(|_| ()),
        Err(EngineError::Poisoned)
    );
    p.reset_all(Reseed::Keep).unwrap();
    assert!(!p.is_poisoned());
}

#[test]
fn test_panic_in_mask_writing_poisons_and_failed_recovery_stays_poisoned() {
    let cfg = CountConfig {
        panic_in_mask: true,
        length: 9,
        ..CountConfig::default()
    };
    let mut p = masked_with(cfg, IllegalActionPolicy::Noop);
    let err = p
        .step(ActionInput {
            discrete: &[0, 0],
            continuous: &[],
        })
        .map(|_| ());
    assert_eq!(err, Err(EngineError::Panicked));
    assert!(p.is_poisoned());
    // Recovery re-observes the reset state (steps = 0), where masks work.
    p.reset_all(Reseed::Keep).unwrap();
    assert!(!p.is_poisoned());
}

#[test]
fn test_set_lineage_is_rejected_while_poisoned_without_mutation() {
    let cfg = CountConfig {
        panic_in_mask: true,
        length: 9,
        ..CountConfig::default()
    };
    let mut p = masked_with(cfg, IllegalActionPolicy::Noop);
    let _ = p.step(ActionInput {
        discrete: &[0, 0],
        continuous: &[],
    });
    assert_eq!(
        p.set_lineage(0, tessera_rl_num::identity::LogicalId::new(3)),
        Err(EngineError::Poisoned)
    );
    // The lineage survives resets, so a rejected change is visible after
    // recovery if it had been applied.
    p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(p.header(0).unwrap().engine.identity.logical_id().get(), 0);
}

#[test]
fn test_mask_budget_is_checked_before_allocating() {
    let cfg = || CountConfig::default();
    // One instance, two agents, three choices: 6 mask bytes.
    let at = |budget| PoolOptions {
        mask_budget_bytes: budget,
        ..options(IllegalActionPolicy::Noop)
    };
    assert!(EnvPool::<Counter, Masked>::with_masks(cfg(), at(6)).is_ok());
    assert_eq!(
        EnvPool::<Counter, Masked>::with_masks(cfg(), at(5)).map(|_| ()),
        Err(EngineError::MaskBudgetExceeded {
            bytes: 6,
            budget: 5
        })
    );
    // Large but representable: 2 * (2^32 - 1) rows times 2^31 choices is
    // just under 2^64, rejected by a normal budget before allocating.
    let big = CountConfig {
        actions_n: 1 << 31,
        ..CountConfig::default()
    };
    let wide = |budget| PoolOptions {
        num_envs: NonZeroU32::new(u32::MAX).unwrap(),
        ..at(budget)
    };
    assert!(matches!(
        EnvPool::<Counter, Masked>::with_masks(big, wide(1 << 30)),
        Err(EngineError::MaskBudgetExceeded {
            budget: 1_073_741_824,
            ..
        })
    ));
    // Two heads of 2^31 choices: 2^33 rows times 2^32 overflows u64.
    let huge = CountConfig {
        actions_n: 1 << 31,
        action_heads: 2,
        ..CountConfig::default()
    };
    assert_eq!(
        EnvPool::<Counter, Masked>::with_masks(huge, wide(u64::MAX)).map(|_| ()),
        Err(EngineError::MaskBudgetExceeded {
            bytes: u64::MAX,
            budget: u64::MAX
        })
    );
    // Pools without masks do not use the budget.
    assert!(EnvPool::<Counter>::new(cfg(), at(0)).is_ok());
}

#[test]
fn test_check_actions_reports_only_required_illegal_actions() {
    let mut p = masked(IllegalActionPolicy::Noop);
    let input = |d: &'static [i32]| ActionInput {
        discrete: d,
        continuous: &[],
    };
    assert_eq!(p.check_actions(input(&[0, 2])), Ok(()));
    assert_eq!(
        p.check_actions(input(&[2, 0])),
        Err(EngineError::IllegalAction {
            instance: 0,
            agent: 0,
            head: 0,
            value: 2
        })
    );
    // Non-mutating.
    assert_eq!(p.header(0).unwrap().agents[0].illegal_actions, 0);
    // A pending instance's actions are not checked.
    step(&mut p, &[0, 0]);
    step(&mut p, &[0, 0]);
    assert_eq!(
        p.check_actions(input(&[2, 2])),
        Err(EngineError::PendingReset { id: 0 })
    );
}

#[cfg(feature = "debug-checks")]
#[test]
fn test_debug_checks_reject_illegal_actions_atomically() {
    let mut p = masked(IllegalActionPolicy::Noop);
    let before = p.state_hash_for_test();
    let err = p
        .step(ActionInput {
            discrete: &[2, 0],
            continuous: &[],
        })
        .map(|_| ());
    assert_eq!(
        err,
        Err(EngineError::IllegalAction {
            instance: 0,
            agent: 0,
            head: 0,
            value: 2
        })
    );
    assert_eq!(p.state_hash_for_test(), before);
    assert_eq!(p.header(0).unwrap().agents[0].illegal_actions, 0);
}

#[cfg(feature = "debug-checks")]
trait HashForTest {
    fn state_hash_for_test(&self) -> u64;
}

#[cfg(feature = "debug-checks")]
impl HashForTest for EnvPool<Counter, Masked> {
    fn state_hash_for_test(&self) -> u64 {
        tessera_rl_core::StateHash::of(self.state(0).unwrap()).get()
    }
}
