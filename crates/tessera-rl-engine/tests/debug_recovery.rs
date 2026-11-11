// SPDX-License-Identifier: MIT OR Apache-2.0

//! ADR-0055: a full reset recovers a pool from any failed call, and the
//! next clean step passes. A call that fails before its allocation check
//! (a panic, whose unwinding allocates inside the chunk step, or an
//! invalid agent event in the same call as an allocation) must not leave
//! its count behind for the next call, in every mode.

#![cfg(all(not(loom), feature = "debug-checks"))]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

#[path = "debug_env/mod.rs"]
mod debug_env;

use core::num::{NonZeroU32, NonZeroUsize};

use debug_env::{ProbeConfig, ProbeEnv};
use tessera_rl_engine::debug::TrackingAllocator;
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AsyncPool, AutoReset, BatchToken, EngineError, EnvPool, HalfToken,
    NoMasks, PipelinedOptions, PipelinedPool, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

const N: u32 = 8;
const ACTIONS: [i32; N as usize] = [0; N as usize];

fn input(discrete: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete,
        continuous: &[],
    }
}

fn plain(cfg: ProbeConfig) -> EnvPool<ProbeEnv> {
    let options = PoolOptions {
        auto_reset: AutoReset::NextStep,
        chunk_size: NonZeroU32::new(2).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(11))
    };
    EnvPool::new(cfg, options).unwrap()
}

/// One pool in one mode, driven one step (or one cycle of half 0, or one
/// batch of every id) at a time.
enum Driver {
    Plain(Box<EnvPool<ProbeEnv>>),
    Piped(
        PipelinedPool<ProbeEnv, NoMasks>,
        Option<HalfToken<ProbeEnv, NoMasks>>,
    ),
    Asy(
        AsyncPool<ProbeEnv, NoMasks>,
        Option<BatchToken<ProbeEnv, NoMasks>>,
    ),
}

#[derive(Clone, Copy)]
enum Mode {
    Inline,
    Sync,
    Pipelined,
    Async,
}

impl Driver {
    fn new(mode: Mode, cfg: ProbeConfig) -> Self {
        let workers = NonZeroUsize::new(3).unwrap();
        match mode {
            Mode::Inline => Self::Plain(Box::new(plain(cfg))),
            Mode::Sync => Self::Plain(Box::new(
                plain(cfg).with_workers(SyncOptions::new(workers)).unwrap(),
            )),
            Mode::Pipelined => Self::Piped(
                plain(cfg)
                    .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
                    .unwrap(),
                None,
            ),
            Mode::Async => Self::Asy(
                plain(cfg)
                    .async_mode(AsyncOptions::new(
                        workers,
                        NonZeroUsize::new(N as usize).unwrap(),
                    ))
                    .unwrap(),
                None,
            ),
        }
    }

    /// A full reset (the recovery path).
    fn reset(&mut self) -> Result<(), EngineError> {
        match self {
            Self::Plain(p) => p.reset_all(Reseed::Keep).map(|_| ()),
            Self::Piped(p, token) => {
                *token = None;
                let [a, _b] = p.reset_all(Reseed::Keep)?;
                *token = Some(a);
                Ok(())
            }
            Self::Asy(p, token) => {
                *token = None;
                *token = Some(p.reset_all(Reseed::Keep)?);
                Ok(())
            }
        }
    }

    /// One step of every instance this driver steps.
    fn step(&mut self) -> Result<(), EngineError> {
        match self {
            Self::Plain(p) => p.step(input(&ACTIONS)).map(|_| ()),
            Self::Piped(p, token) => {
                let t = token.take().unwrap();
                let h = &p.halves()[0];
                let half = vec![0; usize::try_from(h.end - h.start).unwrap()];
                p.send(t, input(&half)).map_err(|e| e.error)?;
                *token = Some(p.recv()?);
                Ok(())
            }
            Self::Asy(p, token) => {
                let t = token.take().unwrap();
                p.send_batch(t, input(&ACTIONS)).map_err(|e| e.error)?;
                *token = Some(p.recv()?);
                Ok(())
            }
        }
    }
}

/// Step 1 passes, step 2 fails with `first`, a full reset recovers, and
/// the clean step after it passes.
fn recovers(mode: Mode, cfg: ProbeConfig, first: fn(&EngineError) -> bool) {
    let mut d = Driver::new(mode, cfg);
    d.reset().unwrap();
    d.step().unwrap();
    let err = d.step().unwrap_err();
    assert!(first(&err), "unexpected error {err:?}");
    d.reset().unwrap();
    assert_eq!(d.step(), Ok(()), "a stale count failed the clean step");
}

fn panic_cfg() -> ProbeConfig {
    ProbeConfig {
        panic_at: 2,
        ..ProbeConfig::default()
    }
}

fn event_and_alloc_cfg() -> ProbeConfig {
    ProbeConfig {
        spawn_alive_at: 2,
        alloc_at: 2,
        ..ProbeConfig::default()
    }
}

fn panicked(e: &EngineError) -> bool {
    matches!(e, EngineError::Panicked)
}

fn invalid_event(e: &EngineError) -> bool {
    matches!(e, EngineError::InvalidAgentEvent { .. })
}

#[test]
fn test_debug_recovery_inline_panic_then_full_reset_steps_clean() {
    recovers(Mode::Inline, panic_cfg(), panicked);
}

#[test]
fn test_debug_recovery_sync_panic_then_full_reset_steps_clean() {
    recovers(Mode::Sync, panic_cfg(), panicked);
}

#[test]
fn test_debug_recovery_pipelined_panic_then_full_reset_steps_clean() {
    recovers(Mode::Pipelined, panic_cfg(), panicked);
}

#[test]
fn test_debug_recovery_async_panic_then_full_reset_steps_clean() {
    recovers(Mode::Async, panic_cfg(), panicked);
}

#[test]
fn test_debug_recovery_inline_event_and_alloc_then_full_reset_steps_clean() {
    recovers(Mode::Inline, event_and_alloc_cfg(), invalid_event);
}

#[test]
fn test_debug_recovery_sync_event_and_alloc_then_full_reset_steps_clean() {
    recovers(Mode::Sync, event_and_alloc_cfg(), invalid_event);
}

#[test]
fn test_debug_recovery_pipelined_event_and_alloc_then_full_reset_steps_clean() {
    recovers(Mode::Pipelined, event_and_alloc_cfg(), invalid_event);
}

#[test]
fn test_debug_recovery_async_event_and_alloc_then_full_reset_steps_clean() {
    recovers(Mode::Async, event_and_alloc_cfg(), invalid_event);
}
