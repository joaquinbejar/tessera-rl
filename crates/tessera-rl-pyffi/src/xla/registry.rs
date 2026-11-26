// SPDX-License-Identifier: MIT OR Apache-2.0

//! The pool registry of the XLA FFI path (ADR-0095 point 3): a fixed array
//! of slots, each with an atomic occupied flag, an atomic busy flag, an
//! atomic generation and an entry reached only while busy is held. A handle
//! is (slot, generation).
//! No mutex: a call that finds its slot busy is refused, never waits; only
//! [`Registry::close`], which is off the hot path, waits for one bounded
//! call to finish.
//!
//! Protocol (normative, ADR-0095 point 3):
//!
//! - `open`: `occupied` compare-exchange `false -> true` (Acquire), the next
//!   slot if held, so an occupied slot's `busy` is never touched; `busy`
//!   claimed (Acquire), retried with a spin and then a yield (only a call
//!   refusing a stale handle can hold it); if the cell already holds an
//!   entry, the slot is left occupied and skipped (never overwritten);
//!   otherwise the entry stored; `busy` released (Release).
//!
//! A handle is not a capability: handles are built from caller-supplied
//! values (the FFI attributes, and Python integers for `_xla_close`), so
//! every rule below holds for a handle `open` never issued.
//! - A call: step 0, `generation` load (Acquire), refused on a mismatch
//!   without claiming, so a stale handle never makes a reused slot busy;
//!   step 1, `busy` compare-exchange `false -> true` (Acquire), refused if
//!   held, as closed if a reloaded `generation` no longer matches and as busy
//!   otherwise; step 2, `generation` loaded again (Acquire), refused and
//!   released on a mismatch; use the entry; `busy` store `false` (Release).
//! - `close`: `generation` compare-exchange from the handle's value to the
//!   next (AcqRel; Relaxed on failure), returning if it fails (idempotent; a
//!   stale handle does nothing); `busy` claimed (Acquire), retried with a
//!   spin and then a yield; the entry taken; if there was one, `occupied`
//!   store `false` (Release); the entry dropped; `busy` released (Release).
//!   Handles reach `close` from Python (`_xla_close(slot, generation)`), so a
//!   handle need not have been issued by `open`: one that matches a free
//!   slot's generation only bumps it.
//!
//! `Busy` therefore means another call holds this handle's slot. Residual
//! (#738): a call that passed step 0 before a close's bump may still be
//! refused `Busy` while its own handle's close holds the slot and the
//! reload still matches, or, for a stale handle, it may hold the reused
//! slot's `busy` for the instant before it refuses `Closed`, refusing a call
//! on the new entry `Busy`. Neither touches an entry.

#![expect(
    unsafe_code,
    reason = "audited module: the registry's entry cell (ADR-0095)"
)]

use super::sync::{AtomicBool, AtomicU64, Ordering, UnsafeCell, spin_loop, yield_now};

/// Slots of the first-party registry.
pub const SLOTS: usize = 64;

/// A registered entry: its slot and the generation it was opened in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handle {
    /// The slot index.
    pub slot: u32,
    /// The slot's generation when the entry was opened.
    pub generation: u64,
}

/// Why the registry refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistryError {
    /// Every slot holds an entry.
    Full,
    /// Another call or a close holds the slot.
    Busy,
    /// The handle's entry was closed (or never existed).
    Closed,
}

struct Slot<E> {
    /// Set by `open`, cleared by `close`: an occupied slot's `busy` is never
    /// claimed by `open` (#738).
    occupied: AtomicBool,
    busy: AtomicBool,
    generation: AtomicU64,
    /// Read or written only while `busy` is held.
    entry: UnsafeCell<Option<E>>,
}

impl<E> Slot<E> {
    #[cfg(not(tessera_xla_loom))]
    const fn new() -> Self {
        Self {
            occupied: AtomicBool::new(false),
            busy: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            entry: UnsafeCell::new(None),
        }
    }

    #[cfg(tessera_xla_loom)]
    fn new() -> Self {
        Self {
            occupied: AtomicBool::new(false),
            busy: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            entry: UnsafeCell::new(None),
        }
    }

    /// Claims `busy`; the claim releases it when dropped, on every path
    /// out (an early return or an unwind from the caller's closure), so a
    /// panic can never leave a slot busy forever.
    fn try_claim(&self) -> Option<Claim<'_, E>> {
        // Acquire: pairs with the Release in `Claim::drop`.
        let claim = self
            .busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
            .then(|| Claim(self));
        #[cfg(all(test, not(tessera_xla_loom)))]
        if claim.is_some() {
            tests::after_claim();
        }
        claim
    }

    /// Claims `busy`, retrying with a spin and then a yield: for `open` and
    /// `close`, off the hot path, which wait for a bounded holder.
    fn claim_waiting(&self) -> Claim<'_, E> {
        let mut spins = 0u32;
        loop {
            if let Some(claim) = self.try_claim() {
                return claim;
            }
            if spins < 64 {
                spin_loop();
                spins += 1;
            } else {
                yield_now();
            }
        }
    }

    /// Runs `f` on the entry cell. The caller holds `busy`.
    fn with_entry<R>(&self, f: impl FnOnce(&mut Option<E>) -> R) -> R {
        // SAFETY: every caller holds `busy`, and `entry` is touched only
        // under it, so this is the only reference to the cell's contents.
        self.entry.with_mut(|e| f(unsafe { &mut *e }))
    }
}

/// A held `busy` flag of one slot.
struct Claim<'a, E>(&'a Slot<E>);

impl<E> Drop for Claim<'_, E> {
    fn drop(&mut self) {
        // Release: publishes the entry's writes to the next holder.
        self.0.busy.store(false, Ordering::Release);
    }
}

/// A fixed array of `N` slots of entries `E`.
pub struct Registry<E, const N: usize = SLOTS> {
    slots: [Slot<E>; N],
}

// SAFETY: a slot's entry is reached only while its `busy` flag is held
// (`with_entry`), so at most one thread touches an entry at a time; sharing
// `&Registry` then only moves entries between threads, which `E: Send`
// allows.
unsafe impl<E: Send, const N: usize> Sync for Registry<E, N> {}

impl<E, const N: usize> Registry<E, N> {
    /// An empty registry.
    #[cfg(not(tessera_xla_loom))]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: [const { Slot::new() }; N],
        }
    }

    /// An empty registry.
    #[cfg(tessera_xla_loom)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: core::array::from_fn(|_| Slot::new()),
        }
    }

    /// Registers `entry` in a free slot; `Err` gives it back when every
    /// slot is occupied. A slot being closed is occupied until its entry is
    /// taken out.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Full`], with the entry, when no slot is free.
    pub fn open(&self, entry: E) -> Result<Handle, (E, RegistryError)> {
        for (index, slot) in self.slots.iter().enumerate() {
            // Acquire: pairs with the Release in `close` that freed the slot.
            // A held slot is skipped without touching its `busy` (#738).
            if slot
                .occupied
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            // Only a call refusing a stale handle can hold `busy` here.
            let _claim = slot.claim_waiting();
            // Never overwrite, in any build: a slot whose cell holds an entry
            // is in use whatever its flag says, so it stays occupied and is
            // skipped (#738, second reader F1).
            if slot.with_entry(|cell| cell.is_some()) {
                continue;
            }
            slot.with_entry(|cell| *cell = Some(entry));
            // Acquire: the generation a close of the previous entry bumped.
            let generation = slot.generation.load(Ordering::Acquire);
            let slot = u32::try_from(index).unwrap_or(u32::MAX);
            return Ok(Handle { slot, generation });
        }
        Err((entry, RegistryError::Full))
    }

    /// Runs `f` on `handle`'s entry: refused with `Busy` while another
    /// call or a close holds the slot, with `Closed` once it was closed.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Busy`] or [`RegistryError::Closed`]; `f` did not run.
    pub fn call<R>(&self, handle: Handle, f: impl FnOnce(&mut E) -> R) -> Result<R, RegistryError> {
        let slot = self.slot(handle)?;
        // Acquire: pairs with the AcqRel bump in `close`. A stale handle is
        // refused here, before claiming, so it never makes the slot's next
        // entry busy (#738).
        if slot.generation.load(Ordering::Acquire) != handle.generation {
            return Err(RegistryError::Closed);
        }
        let Some(_claim) = slot.try_claim() else {
            // A close of this handle may hold the slot: report it as closed
            // rather than busy when the generation has moved on.
            if slot.generation.load(Ordering::Acquire) != handle.generation {
                return Err(RegistryError::Closed);
            }
            return Err(RegistryError::Busy);
        };
        // Again under the claim: a close may have bumped it in between.
        if slot.generation.load(Ordering::Acquire) != handle.generation {
            return Err(RegistryError::Closed);
        }
        slot.with_entry(|cell| cell.as_mut().map(f))
            .ok_or(RegistryError::Closed)
    }

    /// Closes `handle`'s entry: later calls refuse with `Closed`, a call in
    /// flight finishes first, and the entry is dropped. Idempotent; a stale
    /// handle does nothing.
    pub fn close(&self, handle: Handle) {
        let Ok(slot) = self.slot(handle) else {
            return;
        };
        let next = handle.generation.wrapping_add(1);
        // AcqRel: a call that claims the slot after this close releases it
        // sees the new generation.
        if slot
            .generation
            .compare_exchange(handle.generation, next, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        let _claim = slot.claim_waiting();
        let entry = slot.with_entry(Option::take);
        // Release: the slot is free for `open` once the entry is out; `open`
        // still waits for `busy`, so the drop below finishes first. Only a
        // close that took an entry frees the slot: a handle built from
        // caller-supplied values (`_xla_close(slot, generation)` is reachable
        // from Python) whose generation matches a free slot, or a slot an
        // `open` is still filling, takes nothing and leaves `occupied` as it
        // is, so it never marks a slot free that holds, or will hold, an
        // entry (#738, second reader F1).
        if entry.is_some() {
            slot.occupied.store(false, Ordering::Release);
        }
        drop(entry);
    }

    fn slot(&self, handle: Handle) -> Result<&Slot<E>, RegistryError> {
        usize::try_from(handle.slot)
            .ok()
            .and_then(|i| self.slots.get(i))
            .ok_or(RegistryError::Closed)
    }
}

#[cfg(not(tessera_xla_loom))]
impl<E, const N: usize> Default for Registry<E, N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, not(tessera_xla_loom)))]
mod tests {
    use super::{Handle, Registry, RegistryError};
    use std::cell::RefCell;
    use std::sync::Mutex;

    type Hook = Box<dyn FnMut()>;

    std::thread_local! {
        /// Runs on this thread right after any slot's `busy` is claimed, so a
        /// test can act while a claim is held (#738). Taken out while it
        /// runs, so the registry calls it makes do not re-enter it.
        static AFTER_CLAIM: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }

    pub(super) fn after_claim() {
        let hook = AFTER_CLAIM.with(|h| h.borrow_mut().take());
        if let Some(mut f) = hook {
            f();
            AFTER_CLAIM.with(|h| {
                let mut h = h.borrow_mut();
                if h.is_none() {
                    *h = Some(f);
                }
            });
        }
    }

    fn set_after_claim(f: impl FnMut() + 'static) {
        AFTER_CLAIM.with(|h| *h.borrow_mut() = Some(Box::new(f)));
    }

    fn clear_after_claim() {
        AFTER_CLAIM.with(|h| *h.borrow_mut() = None);
    }

    /// #738: opening a handle never makes another handle's live slot busy.
    /// The hook steps the live handle while `open` holds whatever claim it
    /// takes; the step must never be refused `Busy`.
    #[test]
    fn test_registry_open_never_claims_an_occupied_slot() {
        static R: Registry<u64, 2> = Registry::new();
        static SEEN: Mutex<Vec<Result<u64, RegistryError>>> = Mutex::new(Vec::new());
        let h0 = R.open(10).unwrap();
        set_after_claim(move || {
            let r = R.call(h0, |v| *v);
            SEEN.lock().unwrap().push(r);
        });
        let h1 = R.open(20).unwrap();
        clear_after_claim();
        let seen = SEEN.lock().unwrap().clone();
        assert!(!seen.contains(&Err(RegistryError::Busy)), "{seen:?}");
        assert_eq!(R.call(h0, |v| *v), Ok(10));
        assert_eq!(R.call(h1, |v| *v), Ok(20));
    }

    /// #738: a call on a closed handle never makes the slot's new entry
    /// busy. The hook steps the live handle while the stale call holds
    /// whatever claim it takes; the step must never be refused `Busy`.
    #[test]
    fn test_registry_stale_call_never_claims_a_reused_slot() {
        static R: Registry<u64, 1> = Registry::new();
        static SEEN: Mutex<Vec<Result<u64, RegistryError>>> = Mutex::new(Vec::new());
        let stale = R.open(1).unwrap();
        R.close(stale);
        let live = R.open(2).unwrap();
        assert_eq!(live.slot, stale.slot);
        set_after_claim(move || {
            let r = R.call(live, |v| *v);
            SEEN.lock().unwrap().push(r);
        });
        assert_eq!(R.call(stale, |v| *v), Err(RegistryError::Closed));
        clear_after_claim();
        let seen = SEEN.lock().unwrap().clone();
        assert!(!seen.contains(&Err(RegistryError::Busy)), "{seen:?}");
        assert_eq!(R.call(live, |v| *v), Ok(2));
    }

    /// #738 (second reader F4): `open` waits for a stale call's claim on a
    /// free slot instead of storing under it. A thread holds the claim
    /// (in the hook) of a call with a handle that matches the free slot's
    /// generation; an `open` on another thread cannot finish until the claim
    /// is released, and then registers in that slot.
    #[test]
    fn test_registry_open_waits_for_a_stale_claim() {
        use std::sync::atomic::Ordering;
        use std::sync::mpsc;
        static R: Registry<u64, 1> = Registry::new();
        let forged = Handle {
            slot: 0,
            generation: 0,
        };
        let (held_tx, held_rx) = mpsc::channel::<()>();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let holder = std::thread::spawn(move || {
            set_after_claim(move || {
                held_tx.send(()).unwrap();
                go_rx.recv().unwrap();
            });
            let r = R.call(forged, |v| *v);
            clear_after_claim();
            r
        });
        held_rx.recv().unwrap();
        let opener = std::thread::spawn(|| R.open(5).map_err(|(_, e)| e));
        while !R.slots[0].occupied.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!opener.is_finished(), "open stored while a claim was held");
        go_tx.send(()).unwrap();
        assert_eq!(holder.join().unwrap(), Err(RegistryError::Closed));
        let h = opener.join().unwrap().unwrap();
        assert_eq!(h.slot, 0);
        assert_eq!(R.call(h, |v| *v), Ok(5));
    }

    /// #738 (second reader F1), the reader's sequence, made deterministic:
    /// a forged close (matching the free slot's generation) holds `busy`
    /// (in the hook) while an `open` has set `occupied` and waits; the close
    /// takes nothing and must not clear `occupied`. Afterwards the slot
    /// holds the opened entry, is still marked occupied, and a second open
    /// finds no slot.
    #[test]
    fn test_registry_forged_close_racing_open_keeps_the_slot_occupied() {
        use std::sync::atomic::Ordering;
        use std::sync::mpsc;
        static R: Registry<u64, 1> = Registry::new();
        let forged = Handle {
            slot: 0,
            generation: 0,
        };
        let (held_tx, held_rx) = mpsc::channel::<()>();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let closer = std::thread::spawn(move || {
            set_after_claim(move || {
                held_tx.send(()).unwrap();
                go_rx.recv().unwrap();
            });
            R.close(forged);
            clear_after_claim();
        });
        held_rx.recv().unwrap();
        let opener = std::thread::spawn(|| R.open(7).map_err(|(_, e)| e));
        while !R.slots[0].occupied.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        go_tx.send(()).unwrap();
        closer.join().unwrap();
        let h = opener.join().unwrap().unwrap();
        assert!(
            R.slots[0].occupied.load(Ordering::Acquire),
            "a slot holding an entry is marked free"
        );
        assert_eq!(R.call(h, |v| *v), Ok(7));
        assert_eq!(R.open(8).unwrap_err(), (8, RegistryError::Full));
    }

    /// #738 (second reader F1): a close with a handle built from values that
    /// match a free slot's generation (reachable from Python through
    /// `_xla_close`) takes nothing and does not free the slot for a second
    /// entry.
    #[test]
    fn test_registry_forged_close_of_a_free_slot_frees_nothing() {
        let r: Registry<u64, 1> = Registry::new();
        r.close(Handle {
            slot: 0,
            generation: 0,
        });
        let h = r.open(1).unwrap();
        assert_eq!(h.generation, 1);
        assert_eq!(r.open(2).unwrap_err(), (2, RegistryError::Full));
        assert_eq!(r.call(h, |v| *v), Ok(1));
    }

    /// A call made while its own handle's close holds the slot reports
    /// `Closed`, not `Busy` (here through step 0: the close's bump came
    /// first). The reload after a failed claim covers a call that passed
    /// step 0 before the bump; the loom models accept either refusal there.
    #[test]
    fn test_registry_call_during_its_close_reports_closed() {
        static R: Registry<u64, 1> = Registry::new();
        static SEEN: Mutex<Vec<Result<u64, RegistryError>>> = Mutex::new(Vec::new());
        let h = R.open(4).unwrap();
        set_after_claim(move || {
            let r = R.call(h, |v| *v);
            SEEN.lock().unwrap().push(r);
        });
        R.close(h);
        clear_after_claim();
        let seen = SEEN.lock().unwrap().clone();
        assert_eq!(seen, vec![Err(RegistryError::Closed)]);
    }

    #[test]
    fn test_registry_open_call_close_refuses_after_close() {
        let r: Registry<u64, 2> = Registry::new();
        let h = r.open(7).unwrap();
        assert_eq!(r.call(h, |v| *v + 1), Ok(8));
        r.close(h);
        assert_eq!(r.call(h, |v| *v), Err(RegistryError::Closed));
        // Idempotent, and a reused slot gets a new generation.
        r.close(h);
        let h2 = r.open(9).unwrap();
        assert_eq!(h2.slot, h.slot);
        assert_ne!(h2.generation, h.generation);
        assert_eq!(r.call(h, |v| *v), Err(RegistryError::Closed));
        assert_eq!(r.call(h2, |v| *v), Ok(9));
    }

    #[test]
    fn test_registry_full_gives_the_entry_back() {
        let r: Registry<u64, 1> = Registry::new();
        let _h = r.open(1).unwrap();
        assert_eq!(r.open(2).unwrap_err(), (2, RegistryError::Full));
    }

    #[test]
    fn test_registry_nested_call_is_busy() {
        let r: Registry<u64, 1> = Registry::new();
        let h = r.open(1).unwrap();
        let inner = r.call(h, |_| r.call(h, |v| *v));
        assert_eq!(inner, Ok(Err(RegistryError::Busy)));
    }

    /// A panic inside a call releases the slot (its claim drops on unwind),
    /// so later calls and a close never hang.
    #[test]
    fn test_registry_panic_in_call_releases_the_slot() {
        let r: Registry<u64, 1> = Registry::new();
        let h = r.open(3).unwrap();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = r.call(h, |_| panic!("inside a call"));
        }));
        assert!(caught.is_err());
        assert_eq!(r.call(h, |v| *v), Ok(3));
        r.close(h);
        assert_eq!(r.call(h, |v| *v), Err(RegistryError::Closed));
    }

    #[test]
    fn test_registry_unknown_slot_is_closed() {
        let r: Registry<u64, 1> = Registry::new();
        let h = Handle {
            slot: 5,
            generation: 0,
        };
        assert_eq!(r.call(h, |v| *v), Err(RegistryError::Closed));
        r.close(h);
    }
}

/// The loom model of ADR-0095 point 3: steps against a close.
#[cfg(all(test, tessera_xla_loom))]
mod loom_model {
    use super::{Registry, RegistryError};
    use loom::sync::Arc;
    use loom::thread;

    /// The entry: a counter in a loom cell, so an access racing the drop,
    /// or two accesses at once, is a detected data race.
    struct Pool(loom::cell::UnsafeCell<u64>);

    fn step(r: &Registry<Pool, 1>, h: super::Handle) -> Result<u64, RegistryError> {
        r.call(h, |p| {
            p.0.with_mut(|v| {
                // SAFETY: loom tracks this access; the registry's busy flag
                // makes it exclusive.
                let v = unsafe { &mut *v };
                *v += 1;
                *v
            })
        })
    }

    #[test]
    fn test_registry_loom_step_against_close() {
        loom::model(|| {
            let r = Arc::new(Registry::<Pool, 1>::new());
            let h = r
                .open(Pool(loom::cell::UnsafeCell::new(0)))
                .map_err(|_| ())
                .unwrap();
            let stepper = {
                let r = Arc::clone(&r);
                thread::spawn(move || step(&r, h))
            };
            r.close(h);
            // After close returns, no call runs: a late stepper may still
            // hold busy while it reads the new generation and refuses.
            assert!(step(&r, h).is_err());
            // A call admitted before the close completed against the pool.
            match stepper.join().unwrap() {
                Ok(1) | Err(RegistryError::Closed | RegistryError::Busy) => {}
                other => panic!("{other:?}"),
            }
        });
    }

    fn pool() -> Pool {
        Pool(loom::cell::UnsafeCell::new(0))
    }

    fn step_n<const N: usize>(r: &Registry<Pool, N>, h: super::Handle) -> Result<u64, RegistryError> {
        r.call(h, |p| {
            p.0.with_mut(|v| {
                // SAFETY: loom tracks this access; the registry's busy flag
                // makes it exclusive.
                let v = unsafe { &mut *v };
                *v += 1;
                *v
            })
        })
    }

    /// #738: an open racing a step on another slot never refuses the step.
    #[test]
    fn test_registry_loom_open_against_step() {
        loom::model(|| {
            let r = Arc::new(Registry::<Pool, 2>::new());
            let h0 = r.open(pool()).map_err(|_| ()).unwrap();
            let stepper = {
                let r = Arc::clone(&r);
                thread::spawn(move || step_n(&r, h0))
            };
            let h1 = r.open(pool()).map_err(|_| ()).unwrap();
            assert_ne!(h1.slot, h0.slot);
            assert_eq!(stepper.join().unwrap(), Ok(1));
            assert_eq!(step_n(&r, h1), Ok(1));
        });
    }

    /// #738: a call on a closed handle racing a step on the reopened slot
    /// is refused `Closed` and never refuses the step.
    #[test]
    fn test_registry_loom_stale_call_against_reopened_step() {
        loom::model(|| {
            let r = Arc::new(Registry::<Pool, 1>::new());
            let stale = r.open(pool()).map_err(|_| ()).unwrap();
            r.close(stale);
            let live = r.open(pool()).map_err(|_| ()).unwrap();
            let caller = {
                let r = Arc::clone(&r);
                thread::spawn(move || step_n(&r, stale))
            };
            assert_eq!(step_n(&r, live), Ok(1));
            assert_eq!(caller.join().unwrap(), Err(RegistryError::Closed));
        });
    }

    /// #738: an open racing a close of the only slot either finds it still
    /// occupied (`Full`) or takes it after the entry is out; never both
    /// entries in it, and the old entry's drop never races the new one.
    #[test]
    fn test_registry_loom_open_against_close() {
        loom::model(|| {
            let r = Arc::new(Registry::<Pool, 1>::new());
            let h0 = r.open(pool()).map_err(|_| ()).unwrap();
            let closer = {
                let r = Arc::clone(&r);
                thread::spawn(move || r.close(h0))
            };
            let opened = r.open(pool());
            closer.join().unwrap();
            let h1 = match opened {
                Ok(h) => h,
                Err((p, RegistryError::Full)) => r.open(p).map_err(|_| ()).unwrap(),
                Err((_, e)) => panic!("{e:?}"),
            };
            assert_ne!(h1.generation, h0.generation);
            assert_eq!(step_n(&r, h1), Ok(1));
            assert_eq!(step_n(&r, h0), Err(RegistryError::Closed));
        });
    }

    /// #738 (second reader F4): a call on the old handle racing its close
    /// and a reopen of the same slot. The old call completes or refuses;
    /// the new handle's step completes or is refused busy for the instant the
    /// old call holds the slot; the old pool's drop never races the new
    /// pool's store (loom's cell tracking).
    #[test]
    fn test_registry_loom_old_call_against_close_and_reopen() {
        loom::model(|| {
            let r = Arc::new(Registry::<Pool, 1>::new());
            let h0 = r.open(pool()).map_err(|_| ()).unwrap();
            let stepper = {
                let r = Arc::clone(&r);
                thread::spawn(move || step_n(&r, h0))
            };
            r.close(h0);
            let h1 = r.open(pool()).map_err(|_| ()).unwrap();
            match step_n(&r, h1) {
                Ok(1) | Err(RegistryError::Busy) => {}
                other => panic!("new handle: {other:?}"),
            }
            match stepper.join().unwrap() {
                Ok(1) | Err(RegistryError::Closed | RegistryError::Busy) => {}
                other => panic!("old handle: {other:?}"),
            }
        });
    }

    /// #738 (second reader F1): a close with a forged handle matching a free
    /// slot's generation, racing an open, never frees the slot the open
    /// fills: afterwards the slot holds the opened entry and a second open
    /// finds no slot.
    #[test]
    fn test_registry_loom_forged_close_against_open() {
        loom::model(|| {
            let r = Arc::new(Registry::<Pool, 1>::new());
            let forged = super::Handle {
                slot: 0,
                generation: 0,
            };
            let closer = {
                let r = Arc::clone(&r);
                thread::spawn(move || r.close(forged))
            };
            let h = r.open(pool()).map_err(|_| ()).unwrap();
            closer.join().unwrap();
            // The forged close either bumped the free slot before the open
            // read its generation, or closed the issued handle (it then
            // matches it exactly); it never frees a slot holding an entry.
            match step_n(&r, h) {
                Ok(1) => {
                    assert!(r.open(pool()).is_err(), "a second entry in an occupied slot");
                }
                Err(RegistryError::Closed) => {
                    let h2 = r.open(pool()).map_err(|_| ()).unwrap();
                    assert_eq!(step_n(&r, h2), Ok(1));
                }
                other => panic!("{other:?}"),
            }
        });
    }

    #[test]
    fn test_registry_loom_two_steps_against_close() {
        loom::model(|| {
            let r = Arc::new(Registry::<Pool, 1>::new());
            let h = r
                .open(Pool(loom::cell::UnsafeCell::new(0)))
                .map_err(|_| ())
                .unwrap();
            let steppers: Vec<_> = (0..2)
                .map(|_| {
                    let r = Arc::clone(&r);
                    thread::spawn(move || step(&r, h))
                })
                .collect();
            r.close(h);
            assert!(step(&r, h).is_err());
            let mut done = Vec::new();
            for s in steppers {
                if let Ok(v) = s.join().unwrap() {
                    done.push(v);
                }
            }
            // Calls that ran saw distinct counter values: never two at once.
            done.sort_unstable();
            assert!(done.windows(2).all(|w| w[0] != w[1]), "{done:?}");
        });
    }
}
