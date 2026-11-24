// SPDX-License-Identifier: MIT OR Apache-2.0

//! The pool's side of a live watch (ADR-0083 item 7, issue #378 stage
//! 4d): attach and detach, and the service of the slot at a batch boundary.
//! The slot and its `unsafe` reply buffer live in `sys::watch`.
//!
//! Cost on the step path:
//!
//! - no viewer: `if let Some(watch) = &mut pool.watch`, a test of one
//!   pointer (`Watch` holds an `Arc`, so the `Option` is a nullable
//!   pointer) and nothing else;
//! - a viewer attached, nothing requested: the boundary count and the
//!   poison mark (pool-owned plain fields, no atomic), then one `Relaxed`
//!   load of the slot's state;
//! - a request pending: the out-of-line [`Watch::serve`], which encodes one
//!   record into the buffer preallocated at attach (no allocation).
//!
//! The step itself never reads the slot: only the boundary after the call
//! does, so outputs, RNG, determinism and the snapshot contract are
//! untouched (the record served is exactly `snapshot(id)`'s).

use tessera_rl_core::Environment;

use crate::error::EngineError;
use crate::masks::MaskProvider;
use crate::pool::EnvPool;
use crate::record::encode_parts_into;
use crate::sys::sync::Arc;
use crate::sys::watch::{Served, WatchHandle, WatchSlot, watch_pair};

/// The pool's side of an attached watch.
#[derive(Debug)]
pub(crate) struct Watch {
    slot: Arc<WatchSlot>,
    /// Boundaries passed since attach (successful calls): the generation
    /// of a served record.
    calls: u64,
    /// The poison state last published to the slot.
    poison_marked: bool,
}

impl Watch {
    /// A new slot for `num_envs` instances of `len`-byte records, and its
    /// handle. Allocates the reply buffer (never on the step path).
    #[must_use]
    pub(crate) fn attach(num_envs: u32, len: usize) -> (Self, WatchHandle) {
        let (slot, handle) = watch_pair(num_envs, len);
        (
            Self {
                slot,
                calls: 0,
                poison_marked: false,
            },
            handle,
        )
    }

    /// Whether the handle was dropped, so a new attach may replace this.
    #[must_use]
    pub(crate) fn abandoned(&self) -> bool {
        self.slot.handle_dropped()
    }

    /// One batch boundary. `encode(id, buf)` writes instance `id`'s record
    /// and returns false if `id` is not idle at this boundary. Returns
    /// false when the handle is gone and the pool should drop this side.
    #[inline]
    pub(crate) fn boundary(
        &mut self,
        poisoned: bool,
        encode: impl FnOnce(u32, &mut [u8]) -> bool,
    ) -> bool {
        // A u64 count of calls does not overflow in practice; saturating
        // keeps the last generation rather than wrapping to an old one.
        self.calls = self.calls.saturating_add(1);
        if poisoned != self.poison_marked {
            self.poison_marked = poisoned;
            self.slot.set_poisoned(poisoned);
        }
        // A poisoned pool serves nothing (ADR-0028).
        if poisoned || !self.slot.requested() {
            return true;
        }
        self.serve(encode)
    }

    /// A refused or failed call: not a boundary (ADR-0083 item 7 serves
    /// nothing from a refused writer), but a poison it caused is published.
    #[inline]
    pub(crate) fn refused(&mut self, poisoned: bool) {
        if poisoned != self.poison_marked {
            self.poison_marked = poisoned;
            self.slot.set_poisoned(poisoned);
        }
    }

    /// The slow path: a request is pending.
    #[cold]
    #[inline(never)]
    fn serve(&self, encode: impl FnOnce(u32, &mut [u8]) -> bool) -> bool {
        self.slot.serve(self.calls, encode) != Served::Abandoned
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        self.slot.detach();
    }
}

/// Encodes instance `id`'s record from row slices that start at `first`.
/// False if `id` is outside them.
pub(crate) fn encode_rows<E: Environment>(
    headers: &[E::Header],
    states: &[E::State],
    first: u32,
    id: u32,
    buf: &mut [u8],
) -> bool {
    let Some(k) = id.checked_sub(first).and_then(|k| usize::try_from(k).ok()) else {
        return false;
    };
    match (headers.get(k), states.get(k)) {
        (Some(header), Some(state)) => encode_parts_into::<E>(header, state, buf),
        _ => false,
    }
}

impl<E: Environment, M: MaskProvider<E>> EnvPool<E, M> {
    /// Attaches a live watch (ADR-0083 item 7) and returns its single
    /// consumer handle. The pool then serves a pending request after each
    /// `reset`, `reset_all`, `reset_from`, `step`, `restore`, `restore_all`,
    /// `fork` and `fork_branched`, on the caller's thread. Allocates one
    /// record buffer; outside the step path.
    ///
    /// # Errors
    ///
    /// [`EngineError::WatchAttached`] while another handle is alive (a
    /// slot whose handle was dropped is replaced).
    pub fn attach_watch(&mut self) -> Result<WatchHandle, EngineError> {
        if self.watch.as_ref().is_some_and(|w| !w.abandoned()) {
            return Err(EngineError::WatchAttached);
        }
        let (watch, handle) = Watch::attach(
            self.options.num_envs.get(),
            crate::record::Record::<E>::encoded_len(),
        );
        // Dropping a replaced side marks it detached.
        self.watch = Some(watch);
        Ok(handle)
    }

    /// Detaches the watch, if any: the handle's calls then return
    /// `WatchError::Detached`, and a pending request is never served.
    pub fn detach_watch(&mut self) {
        self.watch = None;
    }

    /// Whether a watch is attached.
    #[must_use]
    pub fn has_watch(&self) -> bool {
        self.watch.is_some()
    }

    /// The batch boundary of Sync and Inline mode, after a call that
    /// returned `ok`: every instance is idle. A refused or failed call
    /// serves nothing.
    #[inline]
    pub(crate) fn watch_boundary(&mut self, ok: bool) {
        let Self {
            watch,
            headers,
            states,
            poisoned,
            ..
        } = self;
        if let Some(w) = watch {
            if !ok {
                w.refused(*poisoned);
                return;
            }
            let keep = w.boundary(*poisoned, |id, buf| {
                encode_rows::<E>(headers, states, 0, id, buf)
            });
            if !keep {
                *watch = None;
            }
        }
    }
}
