// SPDX-License-Identifier: MIT OR Apache-2.0

//! Comparison of canonical encodings without allocating.
//!
//! `PartialEq` can differ from encoding equality (`-0.0 == 0.0`, or a lax
//! user implementation), so invariants such as "inactive pool entries hold
//! the initial value" compare canonical bytes. The comparison runs in
//! windows of [`WINDOW`] bytes: for each window both values are encoded
//! again and only that window is kept, so no buffer of `ENCODED_LEN` bytes
//! is needed. The cost is quadratic in `ENCODED_LEN / WINDOW`; it is used
//! only when decoding, never on the hot path.

use super::{Canonical, Encoder, Sink};

const WINDOW: usize = 64;

/// Keeps the bytes of one window of an encoding.
struct WindowSink {
    start: usize,
    pos: usize,
    buf: [u8; WINDOW],
}

impl Sink for WindowSink {
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            if let Some(offset) = self.pos.checked_sub(self.start)
                && let Some(slot) = self.buf.get_mut(offset)
            {
                *slot = byte;
            }
            self.pos = self.pos.saturating_add(1);
        }
    }
}

fn window<T: Canonical>(value: &T, start: usize) -> [u8; WINDOW] {
    let mut out = Encoder::new(WindowSink {
        start,
        pos: 0,
        buf: [0; WINDOW],
    });
    value.encode(&mut out);
    out.into_sink().buf
}

/// Returns whether `a` and `b` have identical canonical encodings.
#[must_use]
pub fn canonical_eq<T: Canonical>(a: &T, b: &T) -> bool {
    (0..T::ENCODED_LEN)
        .step_by(WINDOW)
        .all(|start| window(a, start) == window(b, start))
}

#[cfg(test)]
mod tests {
    use super::canonical_eq;

    #[test]
    fn test_canonical_eq_distinguishes_signed_zero() {
        assert!(canonical_eq(&0.0f64, &0.0f64));
        assert!(!canonical_eq(&-0.0f64, &0.0f64));
    }

    #[test]
    fn test_canonical_eq_spans_several_windows() {
        let a = [7u8; 200];
        let mut b = a;
        assert!(canonical_eq(&a, &b));
        b[150] = 8;
        assert!(!canonical_eq(&a, &b));
    }
}
