// SPDX-License-Identifier: MIT OR Apache-2.0

//! FPCR access on aarch64. Status flags live in FPSR, which this module
//! never writes.

use core::arch::asm;

pub(super) type Control = u64;

// FEAT_AFP: flushes subnormal inputs to zero; must be off. RES0 without
// FEAT_AFP, so clearing it is always valid.
const FIZ: u64 = 1 << 0;
// FEAT_AFP: alternate handling of NaN and min/max; off for IEEE behaviour.
const AH: u64 = 1 << 1;
// Trap enables IOE, DZE, OFE, UFE, IXE (bits 8 to 12) and IDE (bit 15);
// cleared so exceptions never trap. RAZ/WI where traps are unsupported.
const TRAPS: u64 = (0b1_1111 << 8) | (1 << 15);
// Flush-to-zero for half precision; off for IEEE subnormals.
const FZ16: u64 = 1 << 19;
// Rounding mode; 0b00 is round to nearest, ties to even.
const RMODE: u64 = 0b11 << 22;
// Flush-to-zero for single and double precision (also covers denormal
// inputs, the aarch64 equivalent of x86 DAZ); off.
const FZ: u64 = 1 << 24;
// Default NaN; off, so NaN propagation follows IEEE 754.
const DN: u64 = 1 << 25;

/// Canonical control bits.
#[inline]
pub(super) const fn canonical(control: Control) -> Control {
    control & !(FIZ | AH | TRAPS | FZ16 | RMODE | FZ | DN)
}

/// FPCR holds no status flags, so the caller's value is restored as is.
#[inline]
pub(super) const fn restored(saved: Control, _current: Control) -> Control {
    saved
}

// No `nomem` or `preserves_flags`: these instructions must act as compiler
// barriers for floating-point code around them.

#[inline]
#[expect(unsafe_code, reason = "audited module: reads FPCR")]
pub(super) fn read() -> Control {
    let value: u64;
    // SAFETY: `mrs` of FPCR is permitted at EL0 and only copies the register
    // into a general-purpose register.
    unsafe {
        asm!("mrs {}, fpcr", out(reg) value, options(nostack));
    }
    value
}

#[inline]
#[expect(unsafe_code, reason = "audited module: writes FPCR")]
pub(super) fn write(control: Control) {
    // SAFETY: `msr` to FPCR is permitted at EL0. Every value written is
    // derived from a value read from FPCR with only defined bits changed,
    // so no reserved bit is set. Changing the floating-point mode is the
    // purpose of this module (see the module docs for the boundary).
    unsafe {
        asm!("msr fpcr, {}", in(reg) control, options(nostack));
    }
}

#[cfg(test)]
pub(super) mod test_modes {
    use super::{AH, Control, DN, FZ, FZ16, RMODE};

    pub(crate) const fn flush_to_zero(control: Control) -> Control {
        control | FZ
    }

    pub(crate) const fn toward_zero(control: Control) -> Control {
        control | RMODE
    }

    pub(crate) const fn non_ieee_nan_and_half(control: Control) -> Control {
        control | DN | AH | FZ16
    }

    // FPCR has no status flags (they live in FPSR), so every bit is a
    // control bit.
    pub(crate) const fn control_bits(control: Control) -> Control {
        control
    }
}
