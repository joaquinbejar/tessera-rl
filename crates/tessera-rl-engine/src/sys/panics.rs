// SPDX-License-Identifier: MIT OR Apache-2.0

//! Catching panics without letting their payloads panic again.
//!
//! A panic payload is an arbitrary value, and its `Drop` may itself panic.
//! Dropping it at the wrong place (after a worker's decrement, or on the
//! caller's thread after a caught kernel panic) would escape every catch.
//! [`catch`] drops the payload inside a second catch; a payload whose
//! `Drop` panics is forgotten (leaked), never dropped again.

use std::panic::{AssertUnwindSafe, catch_unwind};

/// Runs `f`, returning `true` if it completed and `false` if it panicked.
/// The panic payload is disposed of here, safely.
pub(crate) fn catch(f: impl FnOnce()) -> bool {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(()) => true,
        Err(payload) => {
            dispose(payload);
            false
        }
    }
}

/// Drops a panic payload; one whose `Drop` panics is forgotten.
fn dispose(payload: Box<dyn core::any::Any + Send>) {
    if let Err(nested) = catch_unwind(AssertUnwindSafe(move || drop(payload))) {
        // The payload of the payload's own panic is leaked: dropping it could
        // panic again, and nothing may unwind past this point.
        core::mem::forget(nested);
    }
}

#[cfg(test)]
mod tests {
    use super::catch;

    struct Bomb;

    impl Drop for Bomb {
        fn drop(&mut self) {
            std::panic::panic_any(Inner);
        }
    }

    struct Inner;

    #[test]
    fn test_catch_survives_a_payload_whose_drop_panics() {
        assert!(!catch(|| std::panic::panic_any(Bomb)));
        assert!(catch(|| {}));
    }
}
