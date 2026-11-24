// SPDX-License-Identifier: MIT OR Apache-2.0

//! The monotonic clock of the adaptive worker wait (ADR-0074 item 1): a
//! concrete function, statically dispatched, read only inside a worker's
//! wait, never in a step.
//!
//! Outside loom it is `std::time::Instant` (vDSO `clock_gettime` on Linux,
//! `mach_absolute_time` on macOS, `QueryPerformanceCounter` on Windows),
//! measured from the first read. Under loom it is a deterministic mock: each
//! read on a thread advances that thread's clock by one microsecond, so a
//! model that spins reaches its budget after a bounded number of reads.

/// Nanoseconds since an arbitrary process-wide origin; monotonic.
#[cfg(not(loom))]
#[must_use]
pub(crate) fn now_ns() -> u64 {
    static ORIGIN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let elapsed = ORIGIN.get_or_init(std::time::Instant::now).elapsed();
    // 2^64 ns is about 584 years.
    u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

/// The loom mock: one microsecond per read, per thread.
#[cfg(loom)]
#[must_use]
pub(crate) fn now_ns() -> u64 {
    std::thread_local! {
        static NOW: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
    }
    NOW.with(|now| {
        let next = now.get().checked_add(1_000).unwrap_or(u64::MAX);
        now.set(next);
        next
    })
}

/// Nanoseconds since `start`, a value of [`now_ns`]. The clock is
/// monotonic, so `now >= start`; a backward step, which `Instant` rules
/// out, reads as no time passed.
#[must_use]
pub(crate) fn since(start: u64) -> u64 {
    // Saturating on purpose: the clock is monotonic (`Instant`), so this
    // never saturates; a backward step would read as no time passed. It is
    // scheduler timing, never simulation state.
    now_ns().saturating_sub(start)
}

/// How many spins pass between two clock reads of an adaptive wait: a
/// power of two, so the counter is a mask. One under loom, which explores
/// every spin as a schedule point.
pub(crate) const READ_EVERY: u32 = if cfg!(loom) { 1 } else { 1_024 };

/// The EWMA of a worker's waits after one more `sample` (weight 1/8, in
/// integers). The result lies between `old` and `sample`, so it fits.
#[must_use]
pub(crate) fn ewma(old: u64, sample: u64) -> u64 {
    let mixed = (u128::from(old) * 7 + u128::from(sample)) / 8;
    u64::try_from(mixed).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{READ_EVERY, ewma};

    #[test]
    fn test_ewma_converges_on_a_constant_gap() {
        let mut e = 0;
        for _ in 0..200 {
            e = ewma(e, 3_000_000);
        }
        assert!((2_999_000..=3_000_000).contains(&e), "{e}");
    }

    #[test]
    fn test_ewma_weights_one_eighth() {
        assert_eq!(ewma(800, 0), 700);
        assert_eq!(ewma(0, 800), 100);
        assert_eq!(ewma(u64::MAX, u64::MAX), u64::MAX);
    }

    #[test]
    fn test_read_every_is_a_power_of_two() {
        assert!(READ_EVERY.is_power_of_two());
    }
}
