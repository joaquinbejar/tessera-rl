// SPDX-License-Identifier: MIT OR Apache-2.0

//! ADR-0055: with `debug-checks`, a NaN in a reward or in a state field not
//! annotated `#[state(allow_nan)]` poisons the pool and fails the call with
//! `NanDetected`, in every mode; an allowed NaN does not.

#![cfg(all(not(loom), feature = "debug-checks"))]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

#[path = "debug_env/mod.rs"]
mod debug_env;

use core::num::{NonZeroU32, NonZeroUsize};

use debug_env::{ProbeConfig, ProbeEnv};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AutoReset, EngineError, EnvPool, NanSite, PipelinedOptions,
    PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 8;

fn options() -> PoolOptions {
    PoolOptions {
        auto_reset: AutoReset::NextStep,
        chunk_size: NonZeroU32::new(2).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(3))
    }
}

fn inline(cfg: ProbeConfig) -> EnvPool<ProbeEnv> {
    EnvPool::new(cfg, options()).unwrap()
}

fn sync(cfg: ProbeConfig) -> EnvPool<ProbeEnv> {
    inline(cfg)
        .with_workers(SyncOptions::new(NonZeroUsize::new(3).unwrap()))
        .unwrap()
}

const ACTIONS: [i32; N as usize] = [0; N as usize];

fn input(discrete: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete,
        continuous: &[],
    }
}

fn state_nan(instance: u32) -> EngineError {
    EngineError::NanDetected {
        instance,
        site: NanSite::State { field: "level" },
    }
}

fn reward_nan(instance: u32) -> EngineError {
    EngineError::NanDetected {
        instance,
        site: NanSite::Reward { agent: 0 },
    }
}

/// Steps until the first error; returns it with the number of `Ok` steps.
fn step_until_error(p: &mut EnvPool<ProbeEnv>, max: usize) -> (usize, Option<EngineError>) {
    for k in 0..max {
        if let Err(e) = p.step(input(&ACTIONS)) {
            return (k, Some(e));
        }
    }
    (max, None)
}

#[test]
fn test_debug_nan_inline_state_nan_fails_and_poisons() {
    state_nan_fails_and_poisons(inline);
}

#[test]
fn test_debug_nan_sync_state_nan_fails_and_poisons() {
    state_nan_fails_and_poisons(sync);
}

#[test]
fn test_debug_nan_inline_reward_nan_fails_and_poisons() {
    reward_nan_fails_and_poisons(inline);
}

#[test]
fn test_debug_nan_sync_reward_nan_fails_and_poisons() {
    reward_nan_fails_and_poisons(sync);
}

fn state_nan_fails_and_poisons(make: fn(ProbeConfig) -> EnvPool<ProbeEnv>) {
    {
        let mut p = make(ProbeConfig {
            nan_state_at: 3,
            ..ProbeConfig::default()
        });
        p.reset_all(Reseed::Keep).unwrap();
        assert_eq!(step_until_error(&mut p, 10), (2, Some(state_nan(0))));
        assert!(p.is_poisoned());
        assert_eq!(
            p.step(input(&ACTIONS)).map(|_| ()),
            Err(EngineError::Poisoned)
        );
        // A full reset recovers.
        p.reset_all(Reseed::Keep).unwrap();
        assert!(!p.is_poisoned());
    }
}

fn reward_nan_fails_and_poisons(make: fn(ProbeConfig) -> EnvPool<ProbeEnv>) {
    {
        let mut p = make(ProbeConfig {
            nan_reward_at: 2,
            ..ProbeConfig::default()
        });
        p.reset_all(Reseed::Keep).unwrap();
        assert_eq!(step_until_error(&mut p, 10), (1, Some(reward_nan(0))));
        assert!(p.is_poisoned());
    }
}

/// Within an instance, the reward is checked before the state.
#[test]
fn test_debug_nan_reward_and_state_in_one_step_reports_the_reward() {
    let mut p = inline(ProbeConfig {
        nan_reward_at: 1,
        nan_state_at: 1,
        ..ProbeConfig::default()
    });
    p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(step_until_error(&mut p, 3), (0, Some(reward_nan(0))));
}

/// A NaN in the `allow_nan` field, and no NaN at all, pass in every mode.
#[test]
fn test_debug_nan_allowed_field_or_clean_state_is_not_detected() {
    for cfg in [
        ProbeConfig {
            nan_free: true,
            ..ProbeConfig::default()
        },
        ProbeConfig::default(),
    ] {
        for mut p in [inline(cfg.clone()), sync(cfg.clone())] {
            p.reset_all(Reseed::Keep).unwrap();
            assert_eq!(step_until_error(&mut p, 12), (12, None));
            assert!(!p.is_poisoned());
        }
        let mut piped = inline(cfg.clone())
            .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
            .unwrap();
        let [mut a, mut b] = piped.reset_all(Reseed::Keep).unwrap();
        let half = half_actions(&piped);
        for _ in 0..6 {
            piped.send(a, input(&half)).map_err(|e| e.error).unwrap();
            piped.send(b, input(&half)).map_err(|e| e.error).unwrap();
            a = piped.recv().unwrap();
            b = piped.recv().unwrap();
        }
        let mut asy = inline(cfg)
            .async_mode(AsyncOptions::new(
                NonZeroUsize::new(2).unwrap(),
                NonZeroUsize::new(N as usize).unwrap(),
            ))
            .unwrap();
        let mut token = asy.reset_all(Reseed::Keep).unwrap();
        for _ in 0..6 {
            asy.send_batch(token, input(&ACTIONS))
                .map_err(|e| e.error)
                .unwrap();
            token = asy.recv().unwrap();
        }
        assert!(!asy.is_poisoned());
    }
}

#[test]
fn test_debug_nan_reset_all_and_reset_report_the_first_reset_instance() {
    let cfg = ProbeConfig {
        nan_on_reset: true,
        ..ProbeConfig::default()
    };
    let mut p = sync(cfg.clone());
    assert_eq!(p.reset_all(Reseed::Keep).map(|_| ()), Err(state_nan(0)));
    assert!(p.is_poisoned());
    // A partial reset checks the ids in the order given.
    let mut p = inline(cfg);
    assert_eq!(
        p.reset(&[5, 3], Reseed::Keep).map(|_| ()),
        Err(state_nan(5))
    );
    assert!(p.is_poisoned());
}

fn half_actions<M: tessera_rl_engine::MaskProvider<ProbeEnv>>(
    p: &tessera_rl_engine::PipelinedPool<ProbeEnv, M>,
) -> Vec<i32> {
    let h = &p.halves()[0];
    vec![0; usize::try_from(h.end - h.start).unwrap()]
}

#[test]
fn test_debug_nan_pipelined_recv_fails_with_the_halfs_first_instance() {
    for (cfg, err) in [
        (
            ProbeConfig {
                nan_state_at: 1,
                ..ProbeConfig::default()
            },
            state_nan as fn(u32) -> EngineError,
        ),
        (
            ProbeConfig {
                nan_reward_at: 1,
                ..ProbeConfig::default()
            },
            reward_nan,
        ),
    ] {
        let mut p = inline(cfg)
            .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
            .unwrap();
        let [a, b] = p.reset_all(Reseed::Keep).unwrap();
        drop(a);
        let second = p.halves()[1].start;
        let half = half_actions(&p);
        p.send(b, input(&half)).map_err(|e| e.error).unwrap();
        assert_eq!(p.recv().map(|_| ()), Err(err(second)));
        assert!(p.is_poisoned());
    }
}

#[test]
fn test_debug_nan_async_recv_fails_with_the_lowest_id_of_the_batch() {
    for (cfg, err) in [
        (
            ProbeConfig {
                nan_state_at: 1,
                ..ProbeConfig::default()
            },
            state_nan as fn(u32) -> EngineError,
        ),
        (
            ProbeConfig {
                nan_reward_at: 1,
                ..ProbeConfig::default()
            },
            reward_nan,
        ),
    ] {
        let mut p = inline(cfg)
            .async_mode(AsyncOptions::new(
                NonZeroUsize::new(2).unwrap(),
                NonZeroUsize::new(3).unwrap(),
            ))
            .unwrap();
        let token = p.reset_all(Reseed::Keep).unwrap();
        drop(token);
        // Ids sent in descending order; the batch of three completes in
        // any order, and the lowest id is reported.
        p.send(&[7, 5, 6], input(&[0, 0, 0])).unwrap();
        assert_eq!(p.recv().map(|_| ()), Err(err(5)));
        assert!(p.is_poisoned());
    }
}
