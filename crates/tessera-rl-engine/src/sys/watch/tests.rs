// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the watch slot (ADR-0083 item 7). Under `cfg(loom)` every
//! `model` explores the interleavings (CI: `LOOM_MAX_PREEMPTIONS=3`);
//! natively and under Miri they run once on real threads. Loom's
//! `UnsafeCell` tracking fails a model if a read of the reply buffer
//! overlaps a write.

use super::{Served, WatchError, WatchSlot, watch_pair};
use crate::sys::sync::{Arc, thread};

/// Records are a counter pattern: every byte of generation `g`'s record is
/// `g`, so a torn read shows two values.
const LEN: usize = 4;

fn model(f: impl Fn() + Sync + Send + 'static) {
    #[cfg(loom)]
    loom::model(f);
    #[cfg(not(loom))]
    f();
}

/// Writes the counter pattern of `generation`.
fn fill(generation: u64) -> impl FnOnce(u32, &mut [u8]) -> bool {
    move |_, buf| {
        let byte = u8::try_from(generation).unwrap_or(u8::MAX);
        buf.fill(byte);
        true
    }
}

/// Serves until a request was served once, yielding in between.
fn serve_once(slot: &WatchSlot, generation: u64) {
    loop {
        match slot.serve(generation, fill(generation)) {
            Served::Ready | Served::Abandoned => return,
            Served::Nothing | Served::Pending => thread::yield_now(),
        }
    }
}

/// A whole record: every byte the same, and equal to its generation.
fn assert_whole(out: &[u8], generation: u64) {
    let byte = u8::try_from(generation).unwrap_or(u8::MAX);
    assert!(
        out.iter().all(|&b| b == byte),
        "torn record {out:?} at {generation}"
    );
}

/// Request, serve and take, twice: no torn read, no lost `READY`.
#[test]
fn test_watch_slot_request_serve_take_never_tears() {
    model(|| {
        let (slot, mut handle) = watch_pair(2, LEN);
        let pool = {
            let slot = Arc::clone(&slot);
            thread::spawn(move || {
                serve_once(&slot, 1);
                serve_once(&slot, 2);
            })
        };
        let mut out = [0_u8; LEN];
        for round in 1..=2_u64 {
            loop {
                match handle.request(0) {
                    Ok(()) => break,
                    Err(WatchError::Busy) => thread::yield_now(),
                    Err(e) => panic!("request refused: {e}"),
                }
            }
            loop {
                if let Some(g) = handle.take(&mut out).unwrap() {
                    assert_eq!(g, round);
                    assert_whole(&out, g);
                    break;
                }
                thread::yield_now();
            }
        }
        pool.join().unwrap();
    });
}

/// The reviewer's interleaving (REVIEW-4d-design-r1): a slot `READY`;
/// consumer A starts a take while consumer B, forced to share the slot
/// through a test-only `unsafe impl Sync` wrapper, requests, and the pool
/// serves whatever it sees. With `TAKING` the request is refused while A
/// copies, and the pool never writes during the copy.
#[test]
fn test_watch_slot_two_consumers_request_refused_while_taking() {
    model(|| two_consumers(false));
}

/// The model's own mutant: without `TAKING`, B's request lands while A
/// copies and the pool writes during the copy; loom's cell tracking
/// reports it.
#[cfg(loom)]
#[test]
#[should_panic]
fn test_watch_slot_without_taking_mutant_races() {
    model(|| two_consumers(true));
}

/// Two consumers forced onto one handle (never possible through the public
/// type, which is `!Sync` and takes `&mut self`).
struct Forced(super::WatchHandle);

// SAFETY: test only: the model shares one handle between two threads on
// purpose, to check that `TAKING` alone keeps the pool from writing during
// a copy; every access goes through the slot's `&self` methods.
#[expect(unsafe_code, reason = "test-only forced second consumer")]
unsafe impl Sync for Forced {}

fn two_consumers(no_taking: bool) {
    let mut slot = WatchSlot::new(1, LEN);
    #[cfg(loom)]
    {
        slot.no_taking = no_taking;
    }
    #[cfg(not(loom))]
    let _ = (no_taking, &mut slot);
    let slot = Arc::new(slot);
    let handle = Arc::new(Forced(super::WatchHandle::new(Arc::clone(&slot))));
    // Start in `READY` with generation 1.
    handle.0.slot().request(0).unwrap();
    assert_eq!(slot.serve(1, fill(1)), Served::Ready);
    let a = {
        let handle = Arc::clone(&handle);
        thread::spawn(move || {
            let mut out = [0_u8; LEN];
            if let Ok(Some(g)) = handle.0.slot().take(&mut out) {
                assert_whole(&out, g);
            }
        })
    };
    let b = {
        let handle = Arc::clone(&handle);
        thread::spawn(move || {
            let _ = handle.0.slot().request(0);
        })
    };
    // The pool: one boundary, writing generation 2 if it sees a request.
    let _ = slot.serve(2, fill(2));
    a.join().unwrap();
    b.join().unwrap();
}

/// A request during `SERVING` returns `Busy`: the encode callback runs in
/// `SERVING` and requests from inside it.
#[test]
fn test_watch_slot_request_during_serving_is_busy() {
    model(|| {
        let (slot, mut handle) = watch_pair(1, LEN);
        handle.request(0).unwrap();
        let served = slot.serve(1, |_, buf| {
            assert_eq!(slot.request(0), Err(WatchError::Busy));
            buf.fill(1);
            true
        });
        assert_eq!(served, Served::Ready);
        // A take then succeeds; a request during it (forced) would be Busy.
        let mut out = [0_u8; LEN];
        assert_eq!(handle.take(&mut out), Ok(Some(1)));
        assert_whole(&out, 1);
    });
}

/// Detach while `REQUESTED`: the pool never serves after detaching, and the
/// handle's next take returns `Detached` without blocking.
#[test]
fn test_watch_slot_detach_while_requested_is_detached() {
    model(|| {
        let (slot, mut handle) = watch_pair(1, LEN);
        handle.request(0).unwrap();
        let pool = {
            let slot = Arc::clone(&slot);
            thread::spawn(move || {
                // `detach_watch`: the pool's side is dropped after this.
                slot.detach();
            })
        };
        let mut out = [0_u8; LEN];
        let first = handle.take(&mut out);
        assert!(matches!(first, Ok(None) | Err(WatchError::Detached)));
        pool.join().unwrap();
        assert_eq!(handle.take(&mut out), Err(WatchError::Detached));
        assert_eq!(handle.request(0), Err(WatchError::Detached));
    });
}

/// Detach while `READY`: a copy in progress completes with a whole record,
/// then every call returns `Detached`; nothing is written after detach.
#[test]
fn test_watch_slot_detach_while_ready_copy_completes() {
    model(|| {
        let (slot, mut handle) = watch_pair(1, LEN);
        handle.request(0).unwrap();
        assert_eq!(slot.serve(1, fill(1)), Served::Ready);
        let pool = {
            let slot = Arc::clone(&slot);
            thread::spawn(move || slot.detach())
        };
        let mut out = [0_u8; LEN];
        match handle.take(&mut out) {
            Ok(Some(g)) => assert_whole(&out, g),
            Err(WatchError::Detached) => {}
            other => panic!("unexpected {other:?}"),
        }
        pool.join().unwrap();
        assert_eq!(handle.take(&mut out), Err(WatchError::Detached));
    });
}

/// Stopped training: a pending request with no boundary leaves `take` at
/// `Ok(None)`, never blocking.
#[cfg(not(loom))]
#[test]
fn test_watch_slot_no_boundary_take_is_none() {
    let (_slot, mut handle) = watch_pair(1, LEN);
    handle.request(0).unwrap();
    let mut out = [0_u8; LEN];
    for _ in 0..3 {
        assert_eq!(handle.take(&mut out), Ok(None));
    }
}

/// A poisoned pool serves nothing and `take` reports `Poisoned`.
#[cfg(not(loom))]
#[test]
fn test_watch_slot_poisoned_take_is_poisoned() {
    let (slot, mut handle) = watch_pair(1, LEN);
    handle.request(0).unwrap();
    slot.set_poisoned(true);
    let mut out = [0_u8; LEN];
    assert_eq!(handle.take(&mut out), Err(WatchError::Poisoned));
    slot.set_poisoned(false);
    assert_eq!(handle.take(&mut out), Ok(None));
}

/// Argument checks: buffer length and instance range.
#[cfg(not(loom))]
#[test]
fn test_watch_slot_bad_arguments_are_refused() {
    let (_slot, mut handle) = watch_pair(2, LEN);
    assert_eq!(
        handle.request(2),
        Err(WatchError::InstanceOutOfRange { id: 2, num_envs: 2 })
    );
    let mut short = [0_u8; LEN - 1];
    assert_eq!(
        handle.take(&mut short),
        Err(WatchError::BufferLength {
            expected: LEN,
            actual: LEN - 1
        })
    );
}

/// A dropped handle is seen by the pool on its next pending request only.
#[cfg(not(loom))]
#[test]
fn test_watch_slot_dropped_handle_abandons_on_hit() {
    let (slot, mut handle) = watch_pair(1, LEN);
    handle.request(0).unwrap();
    drop(handle);
    assert!(slot.requested());
    assert_eq!(slot.serve(1, fill(1)), Served::Abandoned);
    assert!(slot.handle_dropped());
}

/// A panicking encoder leaves the slot `REQUESTED`, not stuck in
/// `SERVING`: the request is served at the next boundary (REVIEW-4d-r1).
#[cfg(not(loom))]
#[test]
fn test_watch_slot_encode_panics_request_stays_pending() {
    let (slot, mut handle) = watch_pair(1, LEN);
    handle.request(0).unwrap();
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        slot.serve(1, |_, _| {
            std::panic::resume_unwind(Box::new("encode failed"))
        })
    }));
    assert!(unwound.is_err());
    let mut out = [0_u8; LEN];
    assert_eq!(handle.take(&mut out), Ok(None));
    assert_eq!(slot.request(0), Err(WatchError::Busy));
    assert_eq!(slot.serve(2, fill(2)), Served::Ready);
    assert_eq!(handle.take(&mut out), Ok(Some(2)));
    assert_whole(&out, 2);
}
