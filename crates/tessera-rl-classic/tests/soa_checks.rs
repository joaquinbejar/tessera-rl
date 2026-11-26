// SPDX-License-Identifier: MIT OR Apache-2.0

//! The per-instance checks that read a pool's state rows, on a lane-wise
//! (Struct-of-Arrays, ADR-0090) pool against the Array-of-Structs one:
//!
//! - the `debug-checks` NaN trap (ADR-0055) must catch a NaN state in
//!   Inline, Sync and Pipelined mode as it does for `CartPole`;
//! - the live watch (ADR-0083 item 7) must serve `snapshot(id)` for
//!   `SoaCartPole` in Inline, Sync and Pipelined mode as it does for
//!   `CartPole`.
//!
//! A lane-wise pool keeps its state in columns and its row store empty, so
//! a check that reads the rows instead of gathering the columns sees no
//! instance (#345: the live watch in every mode and the Pipelined NaN trap
//! did, until they read through `state_at`).
//!
//! The NaN tests need the engine's `debug-checks` feature (this crate has
//! no feature of its own to gate on): without it they return early.

#![cfg(not(loom))]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

#[path = "../benches/support/soa_cartpole.rs"]
mod soa_cartpole;

use soa_cartpole::{SoaCartPole, SoaCartPoleState};
use tessera_rl_classic::cartpole::{CartPole, CartPoleConfig, CartPoleState};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, NoMasks, PipelinedOptions,
    PipelinedPool, PoolOptions, Reseed, SyncOptions, WatchHandle,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 16;

/// Upright with a huge finite pole velocity: the next step computes
/// `theta_dot^2 * sin(0) = inf * 0 = NaN`, so the stepped state holds NaN.
/// The imported state itself is finite, so `Validate` accepts it.
const NAN_SEED: [f64; 4] = [0.0, 0.0, 0.0, 1.0e200];

fn options() -> PoolOptions {
    PoolOptions {
        chunk_size: NonZeroU32::new(8).unwrap(),
        auto_reset: AutoReset::NextStep,
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(41))
    }
}

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

macro_rules! fresh {
    ($env:ty) => {{
        let mut p = EnvPool::<$env>::new(CartPoleConfig {}, options()).unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        p
    }};
}

macro_rules! with_workers {
    ($p:expr) => {
        $p.with_workers(SyncOptions::new(NonZeroUsize::new(2).unwrap()))
            .unwrap()
    };
}

macro_rules! pipelined {
    ($p:expr) => {
        $p.pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
            .unwrap()
    };
}

/// Imports the NaN seed into instance 0 of a reset pool.
macro_rules! seed_nan {
    ($p:expr, $wrap:expr) => {{
        let r = $p.snapshot(0).unwrap();
        let state = $wrap(CartPoleState { s: NAN_SEED });
        let rec = DecodedRecord::from_parts(*r.header(), state).unwrap();
        $p.import(&[0], &[rec]).unwrap();
    }};
}

fn aos(s: CartPoleState) -> CartPoleState {
    s
}

fn is_nan_at_0<T>(r: &Result<T, EngineError>) -> bool {
    matches!(r, Err(EngineError::NanDetected { instance: 0, .. }))
}

/// One Pipelined cycle (both halves sent, both received); returns the
/// first error, if any.
fn pipelined_cycle<P: PipelinedCycle>(p: &mut P) -> Result<(), EngineError> {
    p.cycle()
}

/// The send and receive of one cycle, written per environment (the bounds
/// of `PipelinedPool`'s methods are long; the macro keeps them out).
trait PipelinedCycle {
    fn cycle(&mut self) -> Result<(), EngineError>;
}

macro_rules! impl_cycle {
    ($env:ty) => {
        impl PipelinedCycle for PipelinedPool<$env, NoMasks> {
            fn cycle(&mut self) -> Result<(), EngineError> {
                let halves = self.halves();
                for (h, range) in (0_u32..).zip(halves) {
                    let token = self.idle_token(h)?;
                    let rows = usize::try_from(range.end - range.start).unwrap();
                    let a = vec![1_i32; rows];
                    self.send(token, input(&a)).map_err(|e| e.error)?;
                }
                drop(self.recv()?);
                self.recv().map(drop)
            }
        }
    };
}

impl_cycle!(CartPole);
impl_cycle!(SoaCartPole);

// --- (a) the NaN trap -----------------------------------------------------

#[test]
fn test_soa_checks_nan_trap_inline_and_sync_detect_nan() {
    if !tessera_rl_engine::DEBUG_CHECKS {
        return;
    }
    let all = vec![1_i32; N as usize];
    // AoS and lane-wise, Inline.
    let mut p = fresh!(CartPole);
    seed_nan!(p, aos);
    let r = p.step(input(&all)).map(|_| ());
    assert!(is_nan_at_0(&r), "AoS Inline: {r:?}");
    let mut p = fresh!(SoaCartPole);
    seed_nan!(p, SoaCartPoleState::of);
    let r = p.step(input(&all)).map(|_| ());
    assert!(is_nan_at_0(&r), "lane-wise Inline: {r:?}");
    // AoS and lane-wise, Sync.
    let mut p = with_workers!(fresh!(CartPole));
    seed_nan!(p, aos);
    let r = p.step(input(&all)).map(|_| ());
    assert!(is_nan_at_0(&r), "AoS Sync: {r:?}");
    let mut p = with_workers!(fresh!(SoaCartPole));
    seed_nan!(p, SoaCartPoleState::of);
    let r = p.step(input(&all)).map(|_| ());
    assert!(is_nan_at_0(&r), "lane-wise Sync: {r:?}");
}

#[test]
fn test_soa_checks_nan_trap_aos_pipelined_detects_nan() {
    if !tessera_rl_engine::DEBUG_CHECKS {
        return;
    }
    let mut p = fresh!(CartPole);
    seed_nan!(p, aos);
    let mut p = pipelined!(p);
    let r = pipelined_cycle(&mut p);
    assert!(is_nan_at_0(&r), "AoS Pipelined: {r:?}");
    assert!(p.is_poisoned());
}

#[test]
fn test_soa_checks_nan_trap_lanes_pipelined_detects_nan() {
    if !tessera_rl_engine::DEBUG_CHECKS {
        return;
    }
    let mut p = fresh!(SoaCartPole);
    seed_nan!(p, SoaCartPoleState::of);
    let mut p = pipelined!(p);
    let r = pipelined_cycle(&mut p);
    assert!(is_nan_at_0(&r), "lane-wise Pipelined: {r:?}");
    assert!(p.is_poisoned());
}

// --- (b) the live watch ---------------------------------------------------

fn take_ready(handle: &mut WatchHandle) -> Option<Vec<u8>> {
    let mut out = vec![0; handle.record_len()];
    handle.take(&mut out).unwrap().map(|_| out)
}

/// Inline (`$workers` false) or Sync with 2 workers.
macro_rules! sync_watch_serves_snapshot {
    ($env:ty, $workers:expr) => {{
        let mut p = if $workers {
            with_workers!(fresh!($env))
        } else {
            fresh!($env)
        };
        let mut handle = p.attach_watch().unwrap();
        handle.request(3).unwrap();
        let all = vec![1_i32; N as usize];
        p.step(input(&all)).unwrap();
        let served = take_ready(&mut handle).expect("a served record after the step");
        assert_eq!(served, p.snapshot(3).unwrap().to_bytes());
    }};
}

macro_rules! pipelined_watch_serves_snapshot {
    ($env:ty) => {{
        let mut p = pipelined!(fresh!($env));
        let mut handle = p.attach_watch().unwrap();
        let target = p.halves()[1].start;
        handle.request(target).unwrap();
        pipelined_cycle(&mut p).unwrap();
        let served = take_ready(&mut handle).expect("a served record after the cycle");
        let template = fresh!($env).snapshot(0).unwrap();
        let mut records = vec![template; N as usize];
        p.snapshot_all(&mut records).unwrap();
        assert_eq!(served, records[usize::try_from(target).unwrap()].to_bytes());
    }};
}

#[test]
fn test_soa_checks_watch_aos_inline_serves_snapshot() {
    sync_watch_serves_snapshot!(CartPole, false);
}

#[test]
fn test_soa_checks_watch_lanes_inline_serves_snapshot() {
    sync_watch_serves_snapshot!(SoaCartPole, false);
}

#[test]
fn test_soa_checks_watch_aos_sync_serves_snapshot() {
    sync_watch_serves_snapshot!(CartPole, true);
}

#[test]
fn test_soa_checks_watch_lanes_sync_serves_snapshot() {
    sync_watch_serves_snapshot!(SoaCartPole, true);
}

#[test]
fn test_soa_checks_watch_aos_pipelined_serves_snapshot() {
    pipelined_watch_serves_snapshot!(CartPole);
}

#[test]
fn test_soa_checks_watch_lanes_pipelined_serves_snapshot() {
    pipelined_watch_serves_snapshot!(SoaCartPole);
}
