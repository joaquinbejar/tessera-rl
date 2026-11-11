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
    ActionInput, AsyncOptions, AsyncPool, AutoReset, BatchToken, CycleState, DecodedRecord,
    EngineError, EnvPool, HalfToken, NoMasks, PipelinedOptions, PipelinedPool, PoolOptions, Record,
    Reseed, SyncOptions,
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
                if let Some(t) = token.take() {
                    p.send_batch(t, input(&ACTIONS)).map_err(|e| e.error)?;
                } else {
                    let ids: Vec<u32> = (0..N).collect();
                    p.send(&ids, input(&ACTIONS))?;
                }
                *token = Some(p.recv()?);
                Ok(())
            }
        }
    }
}

/// The full writer that recovers the pool.
#[derive(Clone, Copy)]
enum Recovery {
    Reset,
    Restore,
    Import,
}

/// What a full restore or import writes back: the records taken right
/// after the first reset, and the Pipelined cycle state.
struct Saved {
    records: Vec<Record<ProbeEnv>>,
    cycles: Option<CycleState>,
}

impl Driver {
    /// Every record (no token may be in flight).
    fn save(&mut self) -> Saved {
        match self {
            Self::Plain(p) => {
                let mut records = vec![p.snapshot(0).unwrap(); N as usize];
                p.snapshot_all(&mut records).unwrap();
                Saved {
                    records,
                    cycles: None,
                }
            }
            Self::Piped(p, _) => {
                let mut records = vec![probe_record(); N as usize];
                let cycles = p.snapshot_all(&mut records).unwrap();
                Saved {
                    records,
                    cycles: Some(cycles),
                }
            }
            Self::Asy(p, _) => {
                let mut records = vec![probe_record(); N as usize];
                p.snapshot_all(&mut records).unwrap();
                Saved {
                    records,
                    cycles: None,
                }
            }
        }
    }

    /// Recovers with a full restore or import of `saved`.
    fn write_back(&mut self, how: Recovery, saved: &Saved) -> Result<(), EngineError> {
        let decoded: Vec<DecodedRecord<ProbeEnv>> =
            saved.records.iter().map(|r| (*r).into()).collect();
        match (self, how) {
            (d, Recovery::Reset) => d.reset(),
            (Self::Plain(p), Recovery::Restore) => p.restore_all(&saved.records),
            (Self::Plain(p), Recovery::Import) => p.import_all(&decoded),
            (Self::Piped(p, token), Recovery::Restore) => {
                *token = None;
                p.restore_all(&saved.records, saved.cycles.as_ref().unwrap())?;
                *token = Some(p.idle_token(0)?);
                Ok(())
            }
            (Self::Piped(p, token), Recovery::Import) => {
                *token = None;
                p.import_all(&decoded)?;
                *token = Some(p.idle_token(0)?);
                Ok(())
            }
            // No token afterwards: the next step sends by id.
            (Self::Asy(p, token), Recovery::Restore) => {
                *token = None;
                p.restore_all(&saved.records)
            }
            (Self::Asy(p, token), Recovery::Import) => {
                *token = None;
                p.import_all(&decoded)
            }
        }
    }
}

/// A record of a freshly reset Inline pool, to size snapshot buffers.
fn probe_record() -> Record<ProbeEnv> {
    let mut p = plain(ProbeConfig::default());
    p.reset_all(Reseed::Keep).unwrap();
    p.snapshot(0).unwrap()
}

/// Step 1 passes, step 2 fails with `first`, a full reset recovers, and
/// the clean step after it passes.
fn recovers(mode: Mode, cfg: ProbeConfig, first: fn(&EngineError) -> bool) {
    recovers_by(mode, cfg, first, Recovery::Reset);
}

/// As [`recovers`], recovering with `how`.
fn recovers_by(mode: Mode, cfg: ProbeConfig, first: fn(&EngineError) -> bool, how: Recovery) {
    let mut d = Driver::new(mode, cfg);
    d.reset().unwrap();
    let saved = d.save();
    d.step().unwrap();
    let err = d.step().unwrap_err();
    assert!(first(&err), "unexpected error {err:?}");
    d.write_back(how, &saved).unwrap();
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

// Recovery through a full restore or import instead of a reset: each must
// drop the count of the failed call as well.

#[test]
fn test_debug_recovery_inline_panic_then_restore_all_steps_clean() {
    recovers_by(Mode::Inline, panic_cfg(), panicked, Recovery::Restore);
}

#[test]
fn test_debug_recovery_sync_panic_then_import_all_steps_clean() {
    recovers_by(Mode::Sync, panic_cfg(), panicked, Recovery::Import);
}

#[test]
fn test_debug_recovery_pipelined_panic_then_restore_all_steps_clean() {
    recovers_by(Mode::Pipelined, panic_cfg(), panicked, Recovery::Restore);
}

#[test]
fn test_debug_recovery_pipelined_event_and_alloc_then_import_all_steps_clean() {
    recovers_by(
        Mode::Pipelined,
        event_and_alloc_cfg(),
        invalid_event,
        Recovery::Import,
    );
}

#[test]
fn test_debug_recovery_async_panic_then_restore_all_steps_clean() {
    recovers_by(Mode::Async, panic_cfg(), panicked, Recovery::Restore);
}

#[test]
fn test_debug_recovery_async_event_and_alloc_then_import_all_steps_clean() {
    recovers_by(
        Mode::Async,
        event_and_alloc_cfg(),
        invalid_event,
        Recovery::Import,
    );
}
