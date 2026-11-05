// SPDX-License-Identifier: MIT OR Apache-2.0

//! MXCSR access on x86-64.

#[cfg(not(miri))]
use core::arch::asm;

pub(super) type Control = u32;

// Sticky status flags IE, DE, ZE, OE, UE, PE (bits 0 to 5).
const STATUS: u32 = 0b11_1111;
// Denormals-are-zero.
const DAZ: u32 = 1 << 6;
// Exception masks IM, DM, ZM, OM, UM, PM (bits 7 to 12); set means masked.
const MASKS: u32 = 0b11_1111 << 7;
// Rounding control; 0b00 is round to nearest, ties to even.
const RC: u32 = 0b11 << 13;
// Flush-to-zero.
const FTZ: u32 = 1 << 15;

/// Bits the canonical environment clears and sets, and the sticky status
/// flags, for native entry boundaries (ADR-0041).
pub(super) const CLEAR: Control = DAZ | RC | FTZ;
pub(super) const SET: Control = MASKS;
pub(super) const STATUS_FLAGS: Control = STATUS;

/// The register zero-extended to 64 bits.
pub(super) const fn to_bits(control: Control) -> u64 {
    control as u64
}

/// The register from a zero-extended 64-bit value (MXCSR is 32 bits).
#[expect(clippy::cast_possible_truncation, reason = "MXCSR is 32 bits wide")]
pub(super) const fn from_bits(bits: u64) -> Control {
    bits as Control
}

/// Canonical control bits; the status flags are left untouched.
#[inline]
pub(super) const fn canonical(control: Control) -> Control {
    (control & !(DAZ | RC | FTZ)) | MASKS
}

/// Flush-to-zero enabled, for hostile-mode tests.
#[cfg(any(test, feature = "test-hooks"))]
#[inline]
pub(super) const fn flush_to_zero(control: Control) -> Control {
    control | DAZ | FTZ
}

/// The caller's control bits with the current status flags.
#[inline]
pub(super) const fn restored(saved: Control, current: Control) -> Control {
    (saved & !STATUS) | (current & STATUS)
}

// No `nomem`, `readonly` or `preserves_flags`: these instructions must act
// as compiler barriers for floating-point code around them.

#[cfg(not(miri))]
#[inline]
#[expect(unsafe_code, reason = "audited module: reads MXCSR")]
pub(super) fn read() -> Control {
    let mut value: u32 = 0;
    // SAFETY: `stmxcsr` stores the 32-bit MXCSR into the pointed-to memory
    // and has no other effect. The pointer is to a live, aligned, writable
    // local `u32`. SSE is part of the x86-64 baseline.
    unsafe {
        asm!("stmxcsr [{}]", in(reg) &raw mut value, options(nostack));
    }
    value
}

#[cfg(not(miri))]
#[inline]
#[expect(unsafe_code, reason = "audited module: writes MXCSR")]
pub(super) fn write(control: Control) {
    // SAFETY: `ldmxcsr` loads MXCSR from the pointed-to `u32`, which is live
    // and aligned. Every value written is derived from a value read from
    // MXCSR with only defined bits changed, so no reserved bit is set and
    // the instruction cannot fault. Changing the floating-point mode is the
    // purpose of this module (see the module docs for the boundary).
    unsafe {
        asm!("ldmxcsr [{}]", in(reg) &raw const control, options(nostack));
    }
}

// Under Miri, which cannot run inline assembly, the control register is
// emulated per thread, starting at the hardware default, so code that
// enters the canonical environment can be checked for undefined behaviour.
#[cfg(miri)]
std::thread_local! {
    static EMULATED: core::cell::Cell<Control> = const { core::cell::Cell::new(0x1F80) };
}

#[cfg(miri)]
pub(super) fn read() -> Control {
    EMULATED.with(core::cell::Cell::get)
}

#[cfg(miri)]
pub(super) fn write(control: Control) {
    EMULATED.with(|c| c.set(control));
}

#[cfg(any(test, feature = "test-hooks"))]
pub(super) mod test_modes {
    #[cfg(test)]
    use super::MASKS;
    use super::{Control, DAZ, FTZ, RC, STATUS};

    pub(crate) const fn flush_to_zero(control: Control) -> Control {
        control | DAZ | FTZ
    }

    pub(crate) const fn toward_zero(control: Control) -> Control {
        control | RC
    }

    pub(crate) const fn unmask_divide_by_zero(control: Control) -> Control {
        control & !(1 << 9)
    }

    pub(crate) const fn control_bits(control: Control) -> Control {
        control & !STATUS
    }

    #[cfg(test)]
    pub(crate) const fn inexact_raised(control: Control) -> bool {
        control & (1 << 5) != 0
    }

    #[cfg(test)]
    pub(crate) const fn clear_status(control: Control) -> Control {
        control & !STATUS
    }

    #[cfg(test)]
    pub(crate) const fn all_masked(control: Control) -> bool {
        control & MASKS == MASKS
    }
}
