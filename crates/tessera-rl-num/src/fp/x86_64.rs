// SPDX-License-Identifier: MIT OR Apache-2.0

//! MXCSR access on x86-64.

use core::arch::asm;

pub(super) type Control = u32;

const DAZ: u32 = 1 << 6;
const RC: u32 = 0b11 << 13;
const FTZ: u32 = 1 << 15;

#[inline]
pub(super) const fn canonical(control: Control) -> Control {
    control & !(DAZ | RC | FTZ)
}

#[cfg(test)]
pub(super) const fn with_flush_to_zero(control: Control) -> Control {
    control | DAZ | FTZ
}

#[inline]
#[expect(unsafe_code, reason = "audited module: reads MXCSR")]
pub(super) fn read() -> Control {
    let mut value: u32 = 0;
    // SAFETY: `stmxcsr` stores the 32-bit MXCSR into the pointed-to memory
    // and has no other effect. The pointer is to a live, aligned, writable
    // local `u32`. SSE is part of the x86-64 baseline.
    unsafe {
        asm!("stmxcsr [{}]", in(reg) &raw mut value, options(nostack, preserves_flags));
    }
    value
}

#[inline]
#[expect(unsafe_code, reason = "audited module: writes MXCSR")]
pub(super) fn write(control: Control) {
    // SAFETY: `ldmxcsr` loads MXCSR from the pointed-to `u32`, which is live
    // and aligned. Every value written is either one previously read from
    // MXCSR or one with only RC, FTZ and DAZ changed, so no reserved bit is
    // set and the instruction cannot fault. Changing the floating-point mode
    // is the purpose of this module; the guard restores it.
    unsafe {
        asm!("ldmxcsr [{}]", in(reg) &raw const control, options(nostack, preserves_flags, readonly));
    }
}
