// SPDX-License-Identifier: MIT OR Apache-2.0

//! FPCR access on aarch64.

use core::arch::asm;

pub(super) type Control = u64;

const FIZ: u64 = 1 << 0;
const AH: u64 = 1 << 1;
const FZ16: u64 = 1 << 19;
const RMODE: u64 = 0b11 << 22;
const FZ: u64 = 1 << 24;
const DN: u64 = 1 << 25;

#[inline]
pub(super) const fn canonical(control: Control) -> Control {
    control & !(FIZ | AH | FZ16 | RMODE | FZ | DN)
}

#[cfg(test)]
pub(super) const fn with_flush_to_zero(control: Control) -> Control {
    control | FZ
}

#[inline]
#[expect(unsafe_code, reason = "audited module: reads FPCR")]
pub(super) fn read() -> Control {
    let value: u64;
    // SAFETY: `mrs` of FPCR is permitted at EL0 and only copies the register
    // into a general-purpose register.
    unsafe {
        asm!("mrs {}, fpcr", out(reg) value, options(nomem, nostack, preserves_flags));
    }
    value
}

#[inline]
#[expect(unsafe_code, reason = "audited module: writes FPCR")]
pub(super) fn write(control: Control) {
    // SAFETY: `msr` to FPCR is permitted at EL0. Every value written is
    // either one previously read from FPCR or one with only the documented
    // mode bits cleared, so no reserved bit is set. Changing the
    // floating-point mode is the purpose of this module; the guard restores
    // it.
    unsafe {
        asm!("msr fpcr, {}", in(reg) control, options(nomem, nostack, preserves_flags));
    }
}
