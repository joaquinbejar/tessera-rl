// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the id rings and the signal. Under `cfg(loom)` every `model`
//! explores the interleavings, so the models stay small; natively and under
//! Miri they run once.

#[cfg(not(loom))]
use super::{CompletionRing, SubmitRing};
use super::{RingError, Signal};
use crate::sys::sync::{Arc, AtomicU32, Ordering, thread};

fn model(f: impl Fn() + Sync + Send + 'static) {
    #[cfg(loom)]
    loom::model(f);
    #[cfg(not(loom))]
    f();
}

#[cfg(not(loom))]
#[test]
fn test_ring_capacity_is_the_next_power_of_two() {
    assert_eq!(SubmitRing::new(1).unwrap().capacity(), 2);
    assert_eq!(SubmitRing::new(2).unwrap().capacity(), 2);
    assert_eq!(SubmitRing::new(3).unwrap().capacity(), 4);
    assert_eq!(CompletionRing::new(4).unwrap().capacity(), 4);
    assert_eq!(CompletionRing::new(5).unwrap().capacity(), 8);
}

#[cfg(not(loom))]
#[test]
fn test_ring_single_id_never_overwrites_and_refuses_a_third_push() {
    // N = 1 gets two slots, so a second push without a pop cannot reuse the
    // first slot (with one slot it would overwrite id 7).
    let mut ring = SubmitRing::new(1).unwrap();
    ring.push(7).unwrap();
    ring.push(8).unwrap();
    assert_eq!(ring.push(9), Err(RingError::Full { capacity: 2 }));
    assert_eq!(ring.claim().unwrap(), Some(7));
    assert_eq!(ring.claim().unwrap(), Some(8));
    assert_eq!(ring.claim().unwrap(), None);
}

#[test]
fn test_ring_single_id_producer_and_consumer_deliver_every_lap() {
    model(|| {
        let ring = Arc::new(SharedCompletion::new(1));
        let producer = {
            let ring = Arc::clone(&ring);
            thread::spawn(move || {
                for id in 0..2 {
                    while ring.push(id).is_err() {
                        thread::yield_now();
                    }
                }
            })
        };
        let mut got = Vec::new();
        while got.len() < 2 {
            match ring.pop() {
                Some(id) => got.push(id),
                None => thread::yield_now(),
            }
        }
        producer.join().unwrap();
        assert_eq!(got, [0, 1], "in order, nothing overwritten");
    });
}

/// A pop paused between taking its position and freeing the slot makes a
/// concurrent push see `Full` although only one id is then live; the push
/// succeeds once the pop finishes, and nothing is overwritten or lost.
#[test]
fn test_ring_push_behind_a_paused_pop_sees_full_transiently_then_succeeds() {
    model(|| {
        let ring = Arc::new(SharedCompletion::new(2));
        ring.push(0).unwrap();
        ring.push(1).unwrap();
        let consumer = {
            let ring = Arc::clone(&ring);
            thread::spawn(move || ring.pop())
        };
        while ring.push(2).is_err() {
            thread::yield_now();
        }
        let first = consumer.join().unwrap();
        assert_eq!(first, Some(0));
        assert_eq!(ring.pop(), Some(1));
        assert_eq!(ring.pop(), Some(2));
        assert_eq!(ring.pop(), None);
    });
}

/// A push paused between reserving its position and filling the slot makes
/// a concurrent pop see an empty ring; the id appears once the push
/// finishes.
#[test]
fn test_ring_pop_behind_a_paused_push_sees_empty_transiently_then_gets_the_id() {
    model(|| {
        let ring = Arc::new(SharedCompletion::new(2));
        let producer = {
            let ring = Arc::clone(&ring);
            thread::spawn(move || ring.push(5).unwrap())
        };
        let id = loop {
            if let Some(id) = ring.pop() {
                break id;
            }
            thread::yield_now();
        };
        producer.join().unwrap();
        assert_eq!(id, 5);
    });
}

#[cfg(not(loom))]
#[test]
fn test_ring_zero_ids_is_refused() {
    assert!(matches!(
        SubmitRing::new(0),
        Err(RingError::Size { n: 0, .. })
    ));
    assert!(matches!(
        CompletionRing::new(0),
        Err(RingError::Size { n: 0, .. })
    ));
}

#[cfg(not(loom))]
#[test]
fn test_ring_full_push_is_refused_and_overwrites_nothing() {
    let mut ring = SubmitRing::new(2).unwrap();
    ring.push(7).unwrap();
    ring.push(8).unwrap();
    assert_eq!(ring.push(9), Err(RingError::Full { capacity: 2 }));
    assert_eq!(ring.claim().unwrap(), Some(7));
    assert_eq!(ring.claim().unwrap(), Some(8));
    assert_eq!(ring.claim().unwrap(), None);
}

#[cfg(not(loom))]
#[test]
fn test_ring_fifo_order_holds_across_many_laps() {
    let mut ring = CompletionRing::new(3).unwrap();
    let laps = if cfg!(miri) { 20 } else { 1000 };
    for lap in 0..laps {
        for k in 0..3 {
            ring.push(lap * 3 + k).unwrap();
        }
        for k in 0..3 {
            assert_eq!(ring.pop().unwrap(), Some(lap * 3 + k));
        }
        assert_eq!(ring.pop().unwrap(), None);
    }
}

#[test]
fn test_ring_submit_one_producer_two_claimers_wrap_and_deliver_exactly_once() {
    model(|| {
        let ring = Arc::new(SharedSubmit::new(2));
        // Advance the positions by one lap first, so the racing pushes
        // reuse slots: the ring has wrapped.
        // The first lap uses other ids, so a stale slot value is caught.
        for id in 10..12 {
            ring.push(id).unwrap();
            assert_eq!(ring.claim(), Some(id));
        }
        let seen = Arc::new([AtomicU32::new(0), AtomicU32::new(0)]);
        let claimers: Vec<_> = (0..2)
            .map(|_| {
                let ring = Arc::clone(&ring);
                let seen = Arc::clone(&seen);
                thread::spawn(move || {
                    if let Some(id) = ring.claim() {
                        seen.get(id as usize)
                            .expect("a stale id")
                            .fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        // The single producer races the claimers.
        for id in 0..2 {
            ring.push(id).unwrap();
        }
        for c in claimers {
            c.join().unwrap();
        }
        // Whatever the claimers did not take is still in the ring.
        while let Some(id) = ring.claim() {
            seen.get(id as usize)
                .expect("a stale id")
                .fetch_add(1, Ordering::Relaxed);
        }
        for count in seen.iter() {
            assert_eq!(count.load(Ordering::Relaxed), 1, "every id exactly once");
        }
    });
}

#[test]
fn test_ring_completion_two_producers_one_consumer_deliver_exactly_once() {
    model(|| {
        let ring = Arc::new(SharedCompletion::new(2));
        // One lap first, so the racing pushes reuse slots.
        for id in 10..12 {
            ring.push(id).unwrap();
            assert_eq!(ring.pop(), Some(id));
        }
        let producers: Vec<_> = (0..2u32)
            .map(|id| {
                let ring = Arc::clone(&ring);
                thread::spawn(move || ring.push(id).unwrap())
            })
            .collect();
        // The single consumer races the producers.
        let mut seen = [0u32; 2];
        if let Some(id) = ring.pop() {
            *seen.get_mut(id as usize).expect("a stale id") += 1;
        }
        for p in producers {
            p.join().unwrap();
        }
        while let Some(id) = ring.pop() {
            *seen.get_mut(id as usize).expect("a stale id") += 1;
        }
        assert_eq!(seen, [1, 1], "every id exactly once");
    });
}

#[test]
fn test_signal_waiter_never_misses_a_notify() {
    model(|| {
        let signal = Arc::new(Signal::new());
        let ready = Arc::new(AtomicU32::new(0));
        let notifier = {
            let signal = Arc::clone(&signal);
            let ready = Arc::clone(&ready);
            thread::spawn(move || {
                ready.store(1, Ordering::SeqCst);
                signal.notify()
            })
        };
        // The waiter: register, recheck, then wait only while the word is
        // unchanged.
        loop {
            let value = signal.register();
            if ready.load(Ordering::SeqCst) == 1 {
                signal.unregister();
                break;
            }
            while signal.word().load(Ordering::SeqCst) == value {
                thread::yield_now();
            }
            signal.unregister();
        }
        let _ = notifier.join().unwrap();
    });
}

/// The counted-wake protocol as a backend uses it: the waiter blocks, and
/// only a notifier whose `notify` returned `true` wakes it. A notify that
/// skips a registered waiter (or a registration the notifier cannot see)
/// leaves the waiter blocked, which the model reports.
#[test]
fn test_signal_counted_wake_wakes_exactly_when_a_waiter_is_registered() {
    model(|| {
        let signal = Arc::new(Signal::new());
        let ready = Arc::new(AtomicU32::new(0));
        let woken = Arc::new(AtomicU32::new(0));
        let notifier = {
            let signal = Arc::clone(&signal);
            let ready = Arc::clone(&ready);
            let woken = Arc::clone(&woken);
            thread::spawn(move || {
                ready.store(1, Ordering::SeqCst);
                if signal.notify() {
                    // The backend's wake.
                    woken.store(1, Ordering::SeqCst);
                }
            })
        };
        let _value = signal.register();
        if ready.load(Ordering::SeqCst) == 0 {
            // Blocked until the backend wakes this waiter.
            while woken.load(Ordering::SeqCst) == 0 {
                thread::yield_now();
            }
        }
        signal.unregister();
        notifier.join().unwrap();
    });
}

/// A submission ring shared by `Arc` in the tests: the real caller owns
/// its ring with `&mut`, which a test cannot share with spawned threads, so
/// this adapter reaches the producer side through `&self`. Only the model's
/// main thread ever pushes.
struct SharedSubmit {
    ring: super::IdRing,
}

impl SharedSubmit {
    fn new(n: usize) -> Self {
        Self {
            ring: super::IdRing::new(n).unwrap(),
        }
    }
    fn push(&self, id: u32) -> Result<(), RingError> {
        self.ring.push(id)
    }
    fn claim(&self) -> Option<u32> {
        self.ring.pop().unwrap()
    }
}

/// The completion ring with its single consumer reached through `&self`,
/// for the same reason. Only the model's main thread pops.
struct SharedCompletion {
    ring: super::IdRing,
}

impl SharedCompletion {
    fn new(n: usize) -> Self {
        Self {
            ring: super::IdRing::new(n).unwrap(),
        }
    }
    fn push(&self, id: u32) -> Result<(), RingError> {
        self.ring.push(id)
    }
    fn pop(&self) -> Option<u32> {
        self.ring.pop().unwrap()
    }
}

#[cfg(not(loom))]
#[test]
fn test_ring_stalled_id_while_others_cycle_loses_nothing() {
    // One id is claimed and held (a stalled instance) while the others are
    // submitted, claimed, completed and resubmitted many times.
    let n = 4;
    let rounds = if cfg!(miri) { 50 } else { 100_000 };
    let mut submit = SubmitRing::new(n).unwrap();
    let mut done = CompletionRing::new(n).unwrap();
    for id in 0..u32::try_from(n).unwrap() {
        submit.push(id).unwrap();
    }
    let stalled = submit.claim().unwrap().unwrap();
    for _ in 0..rounds {
        for _ in 1..n {
            let id = submit.claim().unwrap().unwrap();
            assert_ne!(id, stalled);
            done.push(id).unwrap();
        }
        for _ in 1..n {
            let id = done.pop().unwrap().unwrap();
            submit.push(id).unwrap();
        }
    }
    done.push(stalled).unwrap();
    assert_eq!(done.pop().unwrap(), Some(stalled));
    assert_eq!(done.pop().unwrap(), None);
}

/// A pop whose free position would overflow reports `Exhausted` before it
/// takes the position: the head does not move and the id stays in the ring
/// (the order of the checks in `pop`, TR on #79).
#[cfg(not(loom))]
#[test]
fn test_ring_pop_reports_exhaustion_before_taking_a_position() {
    let pos = usize::MAX - 1;
    let ring = super::IdRing::at_position(2, pos).unwrap();
    ring.push(7).unwrap();
    // `pos + capacity` overflows, so the pop cannot compute the slot's next
    // free sequence: it must refuse without consuming the position.
    for _ in 0..3 {
        assert_eq!(ring.pop(), Err(RingError::Exhausted));
        assert_eq!(ring.head.load(Ordering::Relaxed), pos, "no position taken");
    }
    // A push at the last position cannot compute its next tail either.
    assert_eq!(ring.push(8), Err(RingError::Exhausted));
}

/// A claim paused inside its slot (after taking the position, before
/// freeing the slot) while the other ids cycle through the ring: the
/// producer may see `Full` transiently and retries, as the Async send does;
/// every id is delivered exactly once and nothing is lost (ADR-0047 section
/// 7, the carried requirement from #78).
#[test]
fn test_ring_claim_paused_while_other_ids_cycle_delivers_every_id_once() {
    model(|| {
        let ring = Arc::new(SharedSubmit::new(2));
        ring.push(0).unwrap();
        ring.push(1).unwrap();
        let seen = Arc::new([AtomicU32::new(0), AtomicU32::new(0)]);
        let paused = {
            let ring = Arc::clone(&ring);
            let seen = Arc::clone(&seen);
            thread::spawn(move || {
                if let Some(id) = ring.claim() {
                    seen.get(id as usize)
                        .expect("a stale id")
                        .fetch_add(1, Ordering::Relaxed);
                }
            })
        };
        // The main thread cycles whichever id it gets: claim it, count it,
        // and push it back once (retrying while the paused claim still holds
        // its slot), then claim it again.
        let mut cycled = None;
        if let Some(id) = ring.claim() {
            seen.get(id as usize)
                .expect("a stale id")
                .fetch_add(1, Ordering::Relaxed);
            while ring.push(id).is_err() {
                thread::yield_now();
            }
            cycled = Some(id);
        }
        paused.join().unwrap();
        while let Some(id) = ring.claim() {
            seen.get(id as usize)
                .expect("a stale id")
                .fetch_add(1, Ordering::Relaxed);
        }
        for (id, count) in seen.iter().enumerate() {
            let want = if cycled == u32::try_from(id).ok() {
                2
            } else {
                1
            };
            assert_eq!(count.load(Ordering::Relaxed), want, "id {id}");
        }
    });
}

/// A run claim takes `ceil(k / share)` of the `k` ids it finds, at most
/// `out.len()`, oldest first; an empty ring gives zero.
#[cfg(not(loom))]
#[test]
fn test_ring_claim_run_takes_its_share_in_order() {
    let mut ring = SubmitRing::new(16).unwrap();
    for id in 0..10 {
        ring.push(id).unwrap();
    }
    let mut out = [u32::MAX; 8];
    // 8 found (bounded by `out`), shared by 3: ceil(8 / 3) = 3.
    assert_eq!(ring.claim_run(&mut out, 3).unwrap(), 3);
    assert_eq!(out[..3], [0, 1, 2]);
    assert_eq!(out[3], u32::MAX, "nothing written past the run");
    // 7 left, one claimer: all 7.
    assert_eq!(ring.claim_run(&mut out, 1).unwrap(), 7);
    assert_eq!(out[..7], [3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(ring.claim_run(&mut out, 1).unwrap(), 0);
    // A share of zero counts as one; an empty `out` claims nothing.
    ring.push(11).unwrap();
    assert_eq!(ring.claim_run(&mut [], 1).unwrap(), 0);
    assert_eq!(ring.claim_run(&mut out, 0).unwrap(), 1);
    assert_eq!(out[0], 11);
}

/// A run push fills only the free slots and reports how many; a full ring
/// refuses the run and overwrites nothing.
#[cfg(not(loom))]
#[test]
fn test_ring_push_run_fills_a_prefix_then_reports_full() {
    let mut ring = CompletionRing::new(4).unwrap();
    ring.push(9).unwrap();
    assert_eq!(ring.push_run(&[0, 1, 2, 3, 4]).unwrap(), 3, "3 slots free");
    assert_eq!(ring.push_run(&[5]), Err(RingError::Full { capacity: 4 }));
    assert_eq!(ring.push_run(&[]).unwrap(), 0);
    for want in [9, 0, 1, 2] {
        assert_eq!(ring.pop().unwrap(), Some(want));
    }
    assert_eq!(ring.pop().unwrap(), None);
    // Across the wrap: a run that crosses the end of the slot array.
    assert_eq!(ring.push_run(&[5, 6, 7]).unwrap(), 3);
    let mut seen = Vec::new();
    while let Some(id) = ring.pop().unwrap() {
        seen.push(id);
    }
    assert_eq!(seen, [5, 6, 7]);
}

/// Run operations at the end of the positions report `Exhausted` before
/// taking a position, as the single ones do.
#[cfg(not(loom))]
#[test]
fn test_ring_run_ops_report_exhaustion_before_taking_a_position() {
    let pos = usize::MAX - 1;
    let ring = super::IdRing::at_position(2, pos).unwrap();
    ring.push(7).unwrap();
    let mut out = [0u32; 2];
    assert_eq!(ring.pop_run(&mut out, 1), Err(RingError::Exhausted));
    assert_eq!(ring.head.load(Ordering::Relaxed), pos, "no position taken");
    assert_eq!(ring.push_run(&[8]), Err(RingError::Exhausted));
    assert_eq!(
        ring.tail.load(Ordering::Relaxed),
        pos + 1,
        "no position taken"
    );
}

/// Two run claimers race the single producer's run push across a wrapped
/// ring: every id is delivered exactly once, whatever the claims' sizes.
#[test]
fn test_ring_claim_run_two_claimers_wrap_and_deliver_exactly_once() {
    fn count(seen: &[AtomicU32; 2], ids: &[u32]) {
        for &id in ids {
            seen.get(id as usize)
                .expect("a stale id")
                .fetch_add(1, Ordering::Relaxed);
        }
    }
    model(|| {
        let ring = Arc::new(super::IdRing::new(2).unwrap());
        for id in 10..12 {
            ring.push(id).unwrap();
            assert_eq!(ring.pop().unwrap(), Some(id));
        }
        let seen = Arc::new([AtomicU32::new(0), AtomicU32::new(0)]);
        let claimers: Vec<_> = (0..2)
            .map(|_| {
                let ring = Arc::clone(&ring);
                let seen = Arc::clone(&seen);
                thread::spawn(move || {
                    let mut out = [0u32; 2];
                    let k = ring.pop_run(&mut out, 1).unwrap();
                    count(&seen, &out[..k]);
                })
            })
            .collect();
        assert_eq!(ring.push_run(&[0, 1]).unwrap(), 2);
        for c in claimers {
            c.join().unwrap();
        }
        let mut out = [0u32; 2];
        loop {
            let k = ring.pop_run(&mut out, 1).unwrap();
            if k == 0 {
                break;
            }
            count(&seen, &out[..k]);
        }
        for c in seen.iter() {
            assert_eq!(c.load(Ordering::Relaxed), 1, "every id exactly once");
        }
    });
}

/// Two run producers race the single consumer: every id arrives exactly
/// once, and each producer's ids arrive in its order.
#[test]
fn test_ring_push_run_two_producers_one_consumer_deliver_exactly_once() {
    model(|| {
        let ring = Arc::new(super::IdRing::new(4).unwrap());
        let producers: Vec<_> = [[0u32, 1], [2, 3]]
            .into_iter()
            .map(|ids| {
                let ring = Arc::clone(&ring);
                thread::spawn(move || {
                    let mut rest: &[u32] = &ids;
                    while !rest.is_empty() {
                        let k = ring.push_run(rest).unwrap();
                        rest = &rest[k..];
                    }
                })
            })
            .collect();
        let mut got = Vec::new();
        if let Some(id) = ring.pop().unwrap() {
            got.push(id);
        }
        for p in producers {
            p.join().unwrap();
        }
        while let Some(id) = ring.pop().unwrap() {
            got.push(id);
        }
        let mut sorted = got.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, [0, 1, 2, 3], "every id exactly once");
        let pos = |id| got.iter().position(|&g| g == id).unwrap();
        assert!(pos(0) < pos(1) && pos(2) < pos(3), "per-producer order");
    });
}
