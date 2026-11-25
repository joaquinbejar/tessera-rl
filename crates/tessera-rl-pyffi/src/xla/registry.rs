// SPDX-License-Identifier: MIT OR Apache-2.0

//! The pool registry of the XLA FFI path (ADR-0095 point 3): a fixed array
//! of slots, each with an atomic busy flag, an atomic generation and an
//! entry reached only while busy is held. A handle is (slot, generation).
//! No mutex: a call that finds its slot busy is refused, never waits; only
//! [`Registry::close`], which is off the hot path, waits for one bounded
//! call to finish.
//!
//! Protocol (normative, ADR-0095 point 3):
//!
//! - A call: `busy` compare-exchange `false -> true` (Acquire), refused if
//!   held; `generation` load (Acquire), refused and released on a mismatch;
//!   use the entry; `busy` store `false` (Release).
//! - `close`: `generation` bumped from the handle's value (AcqRel), a no-op
//!   if another close already bumped it; `busy` claimed by compare-exchange
//!   (Acquire), retried with a spin and then a yield; the entry dropped;
//!   `busy` released (Release).
//!
//! The entry is dropped only while `close` holds `busy`, whose acquiring
//! compare-exchange synchronizes with the releasing store of the last call
//! that held it, so every use happens before the drop.

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
    busy: AtomicBool,
    generation: AtomicU64,
    /// Read or written only while `busy` is held.
    entry: UnsafeCell<Option<E>>,
}

impl<E> Slot<E> {
    #[cfg(not(tessera_xla_loom))]
    const fn new() -> Self {
        Self {
            busy: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            entry: UnsafeCell::new(None),
        }
    }

    #[cfg(tessera_xla_loom)]
    fn new() -> Self {
        Self {
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
        self.busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
            .then(|| Claim(self))
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
    /// slot is taken (a free slot that is busy counts as taken).
    ///
    /// # Errors
    ///
    /// [`RegistryError::Full`], with the entry, when no slot is free.
    pub fn open(&self, entry: E) -> Result<Handle, (E, RegistryError)> {
        for (index, slot) in self.slots.iter().enumerate() {
            let Some(_claim) = slot.try_claim() else {
                continue;
            };
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
        let Some(_claim) = slot.try_claim() else {
            return Err(RegistryError::Busy);
        };
        // Acquire: pairs with the AcqRel bump in `close`.
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
            .compare_exchange(handle.generation, next, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let mut spins = 0u32;
        let _claim = loop {
            if let Some(claim) = slot.try_claim() {
                break claim;
            }
            if spins < 64 {
                spin_loop();
                spins += 1;
            } else {
                yield_now();
            }
        };
        let entry = slot.with_entry(Option::take);
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
