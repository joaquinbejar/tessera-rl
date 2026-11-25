// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `i32` and `Fx32` lane types (ADR-0090 decision 5, stage 2). As in
//! [`crate::lanes`], each type is private, and a value exists only inside
//! [`LaneKernel::run`](crate::LaneKernel::run), entered through a `run_*`
//! function after the level's features were detected: that is what makes
//! the intrinsic calls below sound.
//!
//! The scalar types are the reference: [`I32Lanes`] on `i32`'s and `u32`'s
//! own operations, [`Fx32Lanes`] on [`Fx32`]'s. Each SIMD level is tested bit
//! for bit against them. The `Fx32` product is the exact 64-bit product of
//! the raw values, shifted right by 16 with floor rounding, then clamped:
//! - SSE4.2 and AVX2 multiply the even lanes with `pmuldq` and the odd lanes
//!   after a 64-bit shift right by 32. Bits [16, 48) of each product are the
//!   wrapped result. The product overflows `i32` after the shift unless bits
//!   [47, 64) are all equal, which is tested with 32-bit operations only (an
//!   arithmetic shift of the high dword by 16 against the result's sign),
//!   since neither level has a 64-bit arithmetic shift.
//! - NEON multiplies with `vmull_s32`, shifts with `vshrq_n_s64` (arithmetic)
//!   and clamps with the saturating narrow `vqmovn_s64`.
//!
//! # Safety
//!
//! Every `unsafe` block below calls one intrinsic, or offsets a pointer,
//! and its comment cites this section. An intrinsic of a level is sound
//! only on a CPU with that level's features; three facts guarantee it:
//! 1. The level types (`Avx2I32`, `Sse42Fx32`, `NeonI32`, ...) and their
//!    families are `pub(crate)`: no code outside this crate names them.
//! 2. [`Lanes`] is sealed: no other type can stand for a level, so a
//!    [`LaneKernel`] receives only this crate's families.
//! 3. A [`Resolved`](crate::Resolved) holds a level only after
//!    `Level::present` found its features on this CPU (`Resolved::best`,
//!    `force`, `available`), and [`Resolved::dispatch_lanes`](crate::Resolved::dispatch_lanes)
//!    enters a level's family only inside its `run_*_lanes`, compiled with
//!    that level's `#[target_feature]`.
//!
//! So a value of a level's type, and every call below, exists only inside
//! [`LaneKernel::run`] on a CPU that has the level.

#![expect(
    unsafe_code,
    reason = "the audited simd module: intrinsics over checked levels"
)]
#![expect(
    clippy::inline_always,
    reason = "operations must inline into the target_feature kernel"
)]

use tessera_rl_num::Fx32;
use tessera_rl_num::math::lane_int::{Fx32Lanes, I32Lanes};

const fn word(v: bool) -> i32 {
    if v { -1 } else { 0 }
}

#[inline(always)]
fn bits_to_bools(bits: u8) -> [bool; 8] {
    let mut out = [false; 8];
    for (i, o) in out.iter_mut().enumerate() {
        *o = (bits >> i) & 1 == 1;
    }
    out
}

/// Scalar `i32` lanes: the reference.
#[derive(Clone, Copy)]
pub(crate) struct ScalarI32([i32; 8]);

#[inline(always)]
fn map2(a: [i32; 8], b: [i32; 8], f: impl Fn(i32, i32) -> i32) -> [i32; 8] {
    let mut out = a;
    for (o, b) in out.iter_mut().zip(b) {
        *o = f(*o, b);
    }
    out
}

#[inline(always)]
fn test2(a: [i32; 8], b: [i32; 8], f: impl Fn(i32, i32) -> bool) -> [bool; 8] {
    let mut out = [false; 8];
    for ((o, a), b) in out.iter_mut().zip(a).zip(b) {
        *o = f(a, b);
    }
    out
}

#[expect(clippy::cast_sign_loss, reason = "the u32 view of the same bits")]
const fn u(v: i32) -> u32 {
    v as u32
}

#[expect(clippy::cast_possible_wrap, reason = "the i32 bits of a u32 result")]
const fn s(v: u32) -> i32 {
    v as i32
}

impl I32Lanes for ScalarI32 {
    type Mask = [bool; 8];

    #[inline(always)]
    fn splat(v: i32) -> Self {
        Self([v; 8])
    }
    #[inline(always)]
    fn from_array(v: [i32; 8]) -> Self {
        Self(v)
    }
    #[inline(always)]
    fn to_array(self) -> [i32; 8] {
        self.0
    }
    #[inline(always)]
    fn wrapping_add(self, rhs: Self) -> Self {
        Self(map2(self.0, rhs.0, i32::wrapping_add))
    }
    #[inline(always)]
    fn wrapping_sub(self, rhs: Self) -> Self {
        Self(map2(self.0, rhs.0, i32::wrapping_sub))
    }
    #[inline(always)]
    fn wrapping_mul(self, rhs: Self) -> Self {
        Self(map2(self.0, rhs.0, i32::wrapping_mul))
    }
    #[inline(always)]
    fn overflowing_add(self, rhs: Self) -> (Self, [bool; 8]) {
        (
            self.wrapping_add(rhs),
            test2(self.0, rhs.0, |a, b| a.checked_add(b).is_none()),
        )
    }
    #[inline(always)]
    fn overflowing_sub(self, rhs: Self) -> (Self, [bool; 8]) {
        (
            self.wrapping_sub(rhs),
            test2(self.0, rhs.0, |a, b| a.checked_sub(b).is_none()),
        )
    }
    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        Self(map2(self.0, rhs.0, |a, b| a & b))
    }
    #[inline(always)]
    fn or(self, rhs: Self) -> Self {
        Self(map2(self.0, rhs.0, |a, b| a | b))
    }
    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        Self(map2(self.0, rhs.0, |a, b| a ^ b))
    }
    #[inline(always)]
    fn not(self) -> Self {
        Self(self.0.map(|a| !a))
    }
    #[inline(always)]
    fn shl<const N: i32>(self) -> Self {
        const { assert!(N >= 1 && N < 32, "shift amount in 1..32") };
        Self(self.0.map(|a| a << N))
    }
    #[inline(always)]
    fn shr_arith<const N: i32>(self) -> Self {
        const { assert!(N >= 1 && N < 32, "shift amount in 1..32") };
        Self(self.0.map(|a| a >> N))
    }
    #[inline(always)]
    fn shr_logical<const N: i32>(self) -> Self {
        const { assert!(N >= 1 && N < 32, "shift amount in 1..32") };
        Self(self.0.map(|a| s(u(a) >> N)))
    }
    #[inline(always)]
    fn eq(self, rhs: Self) -> [bool; 8] {
        test2(self.0, rhs.0, |a, b| a == b)
    }
    #[inline(always)]
    fn gt(self, rhs: Self) -> [bool; 8] {
        test2(self.0, rhs.0, |a, b| a > b)
    }
    #[inline(always)]
    fn gt_u(self, rhs: Self) -> [bool; 8] {
        test2(self.0, rhs.0, |a, b| u(a) > u(b))
    }
    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self(map2(self.0, rhs.0, i32::min))
    }
    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self(map2(self.0, rhs.0, i32::max))
    }
    #[inline(always)]
    fn min_u(self, rhs: Self) -> Self {
        Self(map2(self.0, rhs.0, |a, b| s(u(a).min(u(b)))))
    }
    #[inline(always)]
    fn max_u(self, rhs: Self) -> Self {
        Self(map2(self.0, rhs.0, |a, b| s(u(a).max(u(b)))))
    }
    #[inline(always)]
    fn select(mask: [bool; 8], yes: Self, no: Self) -> Self {
        let mut out = no.0;
        for ((o, y), m) in out.iter_mut().zip(yes.0).zip(mask) {
            if m {
                *o = y;
            }
        }
        Self(out)
    }
    #[inline(always)]
    fn mask_from_array(v: [bool; 8]) -> [bool; 8] {
        v
    }
    #[inline(always)]
    fn mask_from_bits(bits: u8) -> [bool; 8] {
        bits_to_bools(bits)
    }
    #[inline(always)]
    fn mask_bits(mask: [bool; 8]) -> u8 {
        mask.iter()
            .enumerate()
            .fold(0, |bits, (i, &m)| bits | (u8::from(m) << i))
    }
    #[inline(always)]
    fn mask_and(a: [bool; 8], b: [bool; 8]) -> [bool; 8] {
        let mut out = a;
        for (o, b) in out.iter_mut().zip(b) {
            *o &= b;
        }
        out
    }
    #[inline(always)]
    fn mask_or(a: [bool; 8], b: [bool; 8]) -> [bool; 8] {
        let mut out = a;
        for (o, b) in out.iter_mut().zip(b) {
            *o |= b;
        }
        out
    }
    #[inline(always)]
    fn mask_not(a: [bool; 8]) -> [bool; 8] {
        a.map(|m| !m)
    }
}

/// Scalar `Fx32` lanes: the reference, on [`Fx32`]'s own operations.
#[derive(Clone, Copy)]
pub(crate) struct ScalarFx32(ScalarI32);

#[inline(always)]
fn fx_map2(a: ScalarFx32, b: ScalarFx32, f: impl Fn(Fx32, Fx32) -> Fx32) -> ScalarFx32 {
    let (a, b) = (a.to_array(), b.to_array());
    let mut out = a;
    for (o, b) in out.iter_mut().zip(b) {
        *o = f(*o, b);
    }
    ScalarFx32::from_array(out)
}

impl Fx32Lanes for ScalarFx32 {
    type Raw = ScalarI32;

    #[inline(always)]
    fn from_raw(raw: ScalarI32) -> Self {
        Self(raw)
    }
    #[inline(always)]
    fn to_raw(self) -> ScalarI32 {
        self.0
    }
    #[inline(always)]
    fn saturating_add(self, rhs: Self) -> Self {
        fx_map2(self, rhs, Fx32::saturating_add)
    }
    #[inline(always)]
    fn saturating_sub(self, rhs: Self) -> Self {
        fx_map2(self, rhs, Fx32::saturating_sub)
    }
    #[inline(always)]
    fn saturating_mul(self, rhs: Self) -> Self {
        fx_map2(self, rhs, Fx32::saturating_mul)
    }
}

/// A level's `i32` and `Fx32` lane types over `$width`-lane registers.
///
/// `$a`, `$b`, `$c`: identifiers, one set per register; `$i`: the register's
/// offset into the eight-lane arrays.
macro_rules! register_int_lanes {
    ($i32:ident, $fx:ident, $m:ident, $(($a:ident $b:ident $c:ident $i:literal))+) => {
        #[derive(Clone, Copy)]
        pub(crate) struct $i32([$m::R; { [$($i),+].len() }]);

        #[derive(Clone, Copy)]
        pub(crate) struct $fx($i32);

        impl I32Lanes for $i32 {
            type Mask = [$m::M; { [$($i),+].len() }];

            #[inline(always)]
            fn splat(v: i32) -> Self {
                // SAFETY: the level's features are present (module safety section).
                Self([$({ let _ = $i; unsafe { $m::splat(v) } }),+])
            }
            #[inline(always)]
            fn from_array(v: [i32; 8]) -> Self {
                let p = v.as_ptr();
                Self([$({
                    // SAFETY: `$i + WIDTH <= 8`, so the offset stays inside `v`.
                    let q = unsafe { p.add($i) };
                    // SAFETY: `q` is valid for one register; the level's
                    // features are present (module safety section).
                    unsafe { $m::load(q) }
                }),+])
            }
            #[inline(always)]
            fn to_array(self) -> [i32; 8] {
                let mut out = [0; 8];
                let p = out.as_mut_ptr();
                let Self([$($a),+]) = self;
                $(
                    // SAFETY: `$i + WIDTH <= 8`, so the offset stays inside `out`.
                    let q = unsafe { p.add($i) };
                    // SAFETY: `q` is valid for one register; the level's
                    // features are present (module safety section).
                    unsafe { $m::store(q, $a) };
                )+
                out
            }
            #[inline(always)]
            fn wrapping_add(self, rhs: Self) -> Self {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                Self([$(unsafe { $m::add($a, $b) }),+])
            }
            #[inline(always)]
            fn wrapping_sub(self, rhs: Self) -> Self {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                Self([$(unsafe { $m::sub($a, $b) }),+])
            }
            #[inline(always)]
            fn wrapping_mul(self, rhs: Self) -> Self {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                Self([$(unsafe { $m::mul_lo($a, $b) }),+])
            }
            #[inline(always)]
            fn overflowing_add(self, rhs: Self) -> (Self, Self::Mask) {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                let sum = [$(unsafe { $m::add($a, $b) }),+];
                let Self([$($c),+]) = Self(sum);
                // SAFETY: as above.
                (Self(sum), [$(unsafe { $m::add_overflow($a, $b, $c) }),+])
            }
            #[inline(always)]
            fn overflowing_sub(self, rhs: Self) -> (Self, Self::Mask) {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                let diff = [$(unsafe { $m::sub($a, $b) }),+];
                let Self([$($c),+]) = Self(diff);
                // SAFETY: as above.
                (Self(diff), [$(unsafe { $m::sub_overflow($a, $b, $c) }),+])
            }
            #[inline(always)]
            fn and(self, rhs: Self) -> Self {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                Self([$(unsafe { $m::and($a, $b) }),+])
            }
            #[inline(always)]
            fn or(self, rhs: Self) -> Self {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                Self([$(unsafe { $m::or($a, $b) }),+])
            }
            #[inline(always)]
            fn xor(self, rhs: Self) -> Self {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                Self([$(unsafe { $m::xor($a, $b) }),+])
            }
            #[inline(always)]
            fn not(self) -> Self {
                let Self([$($a),+]) = self;
                // SAFETY: the level's features are present (module safety section).
                Self([$(unsafe { $m::not($a) }),+])
            }
            #[inline(always)]
            fn shl<const N: i32>(self) -> Self {
                const { assert!(N >= 1 && N < 32, "shift amount in 1..32") };
                let Self([$($a),+]) = self;
                // SAFETY: the level's features are present (module safety section).
                Self([$(unsafe { $m::shl::<N>($a) }),+])
            }
            #[inline(always)]
            fn shr_arith<const N: i32>(self) -> Self {
                const { assert!(N >= 1 && N < 32, "shift amount in 1..32") };
                let Self([$($a),+]) = self;
                // SAFETY: the level's features are present (module safety section).
                Self([$(unsafe { $m::sra::<N>($a) }),+])
            }
            #[inline(always)]
            fn shr_logical<const N: i32>(self) -> Self {
                const { assert!(N >= 1 && N < 32, "shift amount in 1..32") };
                let Self([$($a),+]) = self;
                // SAFETY: the level's features are present (module safety section).
                Self([$(unsafe { $m::srl::<N>($a) }),+])
            }
            #[inline(always)]
            fn eq(self, rhs: Self) -> Self::Mask {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                [$(unsafe { $m::eq($a, $b) }),+]
            }
            #[inline(always)]
            fn gt(self, rhs: Self) -> Self::Mask {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                [$(unsafe { $m::gt($a, $b) }),+]
            }
            #[inline(always)]
            fn gt_u(self, rhs: Self) -> Self::Mask {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                [$(unsafe { $m::gt_u($a, $b) }),+]
            }
            #[inline(always)]
            fn min(self, rhs: Self) -> Self {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                Self([$(unsafe { $m::min($a, $b) }),+])
            }
            #[inline(always)]
            fn max(self, rhs: Self) -> Self {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                Self([$(unsafe { $m::max($a, $b) }),+])
            }
            #[inline(always)]
            fn min_u(self, rhs: Self) -> Self {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                Self([$(unsafe { $m::min_u($a, $b) }),+])
            }
            #[inline(always)]
            fn max_u(self, rhs: Self) -> Self {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                Self([$(unsafe { $m::max_u($a, $b) }),+])
            }
            #[inline(always)]
            fn select(mask: Self::Mask, yes: Self, no: Self) -> Self {
                let ([$($c),+], Self([$($a),+]), Self([$($b),+])) = (mask, yes, no);
                // SAFETY: the level's features are present (module safety section).
                Self([$(unsafe { $m::select($c, $a, $b) }),+])
            }
            #[inline(always)]
            fn mask_from_array(v: [bool; 8]) -> Self::Mask {
                let words = v.map(word);
                let p = words.as_ptr();
                [$({
                    // SAFETY: `$i + WIDTH <= 8`, so the offset stays inside `words`.
                    let q = unsafe { p.add($i) };
                    // SAFETY: `q` is valid for one register; the level's
                    // features are present (module safety section).
                    unsafe { $m::load_mask(q) }
                }),+]
            }
            #[inline(always)]
            fn mask_from_bits(bits: u8) -> Self::Mask {
                Self::mask_from_array(bits_to_bools(bits))
            }
            #[inline(always)]
            fn mask_bits(mask: Self::Mask) -> u8 {
                let [$($a),+] = mask;
                // SAFETY: the level's features are present (module safety section).
                0 $(| (unsafe { $m::bits($a) } << $i))+
            }
            #[inline(always)]
            fn mask_and(a: Self::Mask, b: Self::Mask) -> Self::Mask {
                let ([$($a),+], [$($b),+]) = (a, b);
                // SAFETY: the level's features are present (module safety section).
                [$(unsafe { $m::mask_and($a, $b) }),+]
            }
            #[inline(always)]
            fn mask_or(a: Self::Mask, b: Self::Mask) -> Self::Mask {
                let ([$($a),+], [$($b),+]) = (a, b);
                // SAFETY: the level's features are present (module safety section).
                [$(unsafe { $m::mask_or($a, $b) }),+]
            }
            #[inline(always)]
            fn mask_not(a: Self::Mask) -> Self::Mask {
                let [$($a),+] = a;
                // SAFETY: the level's features are present (module safety section).
                [$(unsafe { $m::mask_not($a) }),+]
            }
        }

        impl Fx32Lanes for $fx {
            type Raw = $i32;

            #[inline(always)]
            fn from_raw(raw: $i32) -> Self {
                Self(raw)
            }
            #[inline(always)]
            fn to_raw(self) -> $i32 {
                self.0
            }
            #[inline(always)]
            fn saturating_add(self, rhs: Self) -> Self {
                let (Self($i32([$($a),+])), Self($i32([$($b),+]))) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                Self($i32([$(unsafe { $m::sat_add($a, $b) }),+]))
            }
            #[inline(always)]
            fn saturating_sub(self, rhs: Self) -> Self {
                let (Self($i32([$($a),+])), Self($i32([$($b),+]))) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                Self($i32([$(unsafe { $m::sat_sub($a, $b) }),+]))
            }
            #[inline(always)]
            fn saturating_mul(self, rhs: Self) -> Self {
                let (Self($i32([$($a),+])), Self($i32([$($b),+]))) = (self, rhs);
                // SAFETY: the level's features are present (module safety section).
                Self($i32([$(unsafe { $m::fx_sat_mul($a, $b) }),+]))
            }
        }
    };
}

/// Leaf wrappers: each `unsafe fn` is exactly one intrinsic and requires its
/// level. Operations of several intrinsics are written out below them, one
/// `unsafe` block per intrinsic (AGENTS.md).
macro_rules! ops {
    ($($(#[$attr:meta])* fn $name:ident($($p:ident: $t:ty),*) $(-> $r:ty)? = $e:expr;)+) => {
        $(
            $(#[$attr])*
            /// # Safety
            /// The level's features are present, and any pointer is valid
            /// for one register ([module safety section](super)).
            #[inline(always)]
            pub(super) unsafe fn $name($($p: $t),*) $(-> $r)? {
                // SAFETY: this function's contract: one intrinsic of the level.
                unsafe { $e }
            }
        )+
    };
}

/// The operations SSE4.2 and AVX2 build from several leaves, written once
/// over each level's leaf names. Every block calls one leaf: its level's
/// features are present ([module safety section](super)).
#[cfg(target_arch = "x86_64")]
macro_rules! x86_composites {
    () => {
        /// `!a`: `a ^ -1`.
        ///
        /// # Safety
        /// The level's features are present.
        #[inline(always)]
        pub(super) unsafe fn not(a: R) -> R {
            // SAFETY: the level's features are present (module safety section).
            let ones = unsafe { splat(-1) };
            // SAFETY: as above.
            unsafe { xor(a, ones) }
        }
        /// The mask complement: `a ^ -1`.
        ///
        /// # Safety
        /// The level's features are present.
        #[inline(always)]
        pub(super) unsafe fn mask_not(a: M) -> M {
            // SAFETY: the level's features are present (module safety section).
            unsafe { not(a) }
        }
        /// Unsigned `a > b`: the signed compare after flipping both sign bits.
        ///
        /// # Safety
        /// The level's features are present.
        #[inline(always)]
        pub(super) unsafe fn gt_u(a: R, b: R) -> M {
            // SAFETY: the level's features are present (module safety section).
            let flip = unsafe { splat(i32::MIN) };
            // SAFETY: as above.
            let a = unsafe { xor(a, flip) };
            // SAFETY: as above.
            let b = unsafe { xor(b, flip) };
            // SAFETY: as above.
            unsafe { gt(a, b) }
        }
        /// The mask's lane bits, lane `i` in bit `i`.
        ///
        /// # Safety
        /// The level's features are present.
        #[inline(always)]
        pub(super) unsafe fn bits(a: M) -> u8 {
            // SAFETY: the level's features are present (module safety section).
            let floats = unsafe { as_ps(a) };
            // SAFETY: as above.
            let signs = unsafe { movemask(floats) };
            // One bit per 32-bit lane of one register: at most 8 lanes.
            debug_assert!((0..=0xff).contains(&signs), "movemask of {signs:#x}");
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a movemask of at most 8 lanes fits in u8"
            )]
            let out = signs as u8;
            out
        }
        /// A signed overflow of `s = a + b`: the sign bit of
        /// `(a ^ s) & (b ^ s)` (both operands differ in sign from the sum),
        /// spread over the lane.
        ///
        /// # Safety
        /// The level's features are present.
        #[inline(always)]
        pub(super) unsafe fn add_overflow(a: R, b: R, s: R) -> M {
            // SAFETY: the level's features are present (module safety section).
            let a_flips = unsafe { xor(a, s) };
            // SAFETY: as above.
            let b_flips = unsafe { xor(b, s) };
            // SAFETY: as above.
            let both = unsafe { and(a_flips, b_flips) };
            // SAFETY: as above.
            unsafe { sra31(both) }
        }
        /// `d = a - b` overflows when `a` and `b` differ in sign and `d`'s
        /// sign differs from `a`'s.
        ///
        /// # Safety
        /// The level's features are present.
        #[inline(always)]
        pub(super) unsafe fn sub_overflow(a: R, b: R, d: R) -> M {
            // SAFETY: the level's features are present (module safety section).
            let operands_differ = unsafe { xor(a, b) };
            // SAFETY: as above.
            let result_flips = unsafe { xor(a, d) };
            // SAFETY: as above.
            let both = unsafe { and(operands_differ, result_flips) };
            // SAFETY: as above.
            unsafe { sra31(both) }
        }
        /// The saturation bound for a result whose sign should be `sign`'s:
        /// `MIN` if `sign` is negative, else `MAX` (`(sign >> 31) ^ MAX`).
        ///
        /// # Safety
        /// The level's features are present.
        #[inline(always)]
        unsafe fn clamp_by_sign(sign: R) -> R {
            // SAFETY: the level's features are present (module safety section).
            let spread = unsafe { sra31(sign) };
            // SAFETY: as above.
            let max = unsafe { splat(i32::MAX) };
            // SAFETY: as above.
            unsafe { xor(spread, max) }
        }
        /// `Fx32::saturating_add`: on overflow, the bound of `a`'s sign.
        ///
        /// # Safety
        /// The level's features are present.
        #[inline(always)]
        pub(super) unsafe fn sat_add(a: R, b: R) -> R {
            // SAFETY: the level's features are present (module safety section).
            let s = unsafe { add(a, b) };
            // SAFETY: as above.
            let over = unsafe { add_overflow(a, b, s) };
            // SAFETY: as above.
            let clamp = unsafe { clamp_by_sign(a) };
            // SAFETY: as above.
            unsafe { select(over, clamp, s) }
        }
        /// `Fx32::saturating_sub`: on overflow, the bound of `a`'s sign.
        ///
        /// # Safety
        /// The level's features are present.
        #[inline(always)]
        pub(super) unsafe fn sat_sub(a: R, b: R) -> R {
            // SAFETY: the level's features are present (module safety section).
            let d = unsafe { sub(a, b) };
            // SAFETY: as above.
            let over = unsafe { sub_overflow(a, b, d) };
            // SAFETY: as above.
            let clamp = unsafe { clamp_by_sign(a) };
            // SAFETY: as above.
            unsafe { select(over, clamp, d) }
        }
        /// `Fx32::saturating_mul` (module documentation).
        ///
        /// # Safety
        /// The level's features are present.
        #[inline(always)]
        pub(super) unsafe fn fx_sat_mul(a: R, b: R) -> R {
            // Even lanes: the exact 64-bit products of dwords 0, 2, 4, 6.
            // SAFETY: the level's features are present (module safety section).
            let even = unsafe { mul_even(a, b) };
            // Odd lanes: dwords 1, 3, 5, 7 moved down, then multiplied.
            // SAFETY: as above.
            let a_odd = unsafe { srl64_32(a) };
            // SAFETY: as above.
            let b_odd = unsafe { srl64_32(b) };
            // SAFETY: as above.
            let odd = unsafe { mul_even(a_odd, b_odd) };
            // Bits [16, 48): into the low dword (even) and the high dword (odd).
            // SAFETY: as above.
            let even_mid = unsafe { srl64_16(even) };
            // SAFETY: as above.
            let odd_mid = unsafe { sll64_16(odd) };
            // SAFETY: as above.
            let wrapped = unsafe { blend_odd(even_mid, odd_mid) };
            // Bits [48, 64), sign-extended, aligned with each result lane.
            // SAFETY: as above.
            let even_top = unsafe { sra16(even) };
            // SAFETY: as above.
            let even_high = unsafe { srl64_32(even_top) };
            // SAFETY: as above.
            let odd_high = unsafe { sra16(odd) };
            // SAFETY: as above.
            let high = unsafe { blend_odd(even_high, odd_high) };
            // In range exactly when bits [47, 64) are all equal.
            // SAFETY: as above.
            let sign = unsafe { sra31(wrapped) };
            // SAFETY: as above.
            let fits = unsafe { eq(high, sign) };
            // Out of range: the product's sign (`a ^ b`) picks MIN or MAX.
            // SAFETY: as above.
            let product_sign = unsafe { xor(a, b) };
            // SAFETY: as above.
            let clamp = unsafe { clamp_by_sign(product_sign) };
            // SAFETY: as above.
            unsafe { select(fits, wrapped, clamp) }
        }
    };
}

/// AVX2: one `__m256i` per eight lanes.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    #[cfg_attr(
        not(test),
        expect(clippy::wildcard_imports, reason = "the intrinsics of one level")
    )]
    use core::arch::x86_64::*;
    pub(super) type R = __m256i;
    pub(super) type M = __m256i;
    ops! {
        fn splat(v: i32) -> R = _mm256_set1_epi32(v);
        fn load(p: *const i32) -> R = _mm256_loadu_si256(p.cast());
        fn store(p: *mut i32, a: R) = _mm256_storeu_si256(p.cast(), a);
        fn add(a: R, b: R) -> R = _mm256_add_epi32(a, b);
        fn sub(a: R, b: R) -> R = _mm256_sub_epi32(a, b);
        fn mul_lo(a: R, b: R) -> R = _mm256_mullo_epi32(a, b);
        fn and(a: R, b: R) -> R = _mm256_and_si256(a, b);
        fn or(a: R, b: R) -> R = _mm256_or_si256(a, b);
        fn xor(a: R, b: R) -> R = _mm256_xor_si256(a, b);
        fn eq(a: R, b: R) -> M = _mm256_cmpeq_epi32(a, b);
        fn gt(a: R, b: R) -> M = _mm256_cmpgt_epi32(a, b);
        fn min(a: R, b: R) -> R = _mm256_min_epi32(a, b);
        fn max(a: R, b: R) -> R = _mm256_max_epi32(a, b);
        fn min_u(a: R, b: R) -> R = _mm256_min_epu32(a, b);
        fn max_u(a: R, b: R) -> R = _mm256_max_epu32(a, b);
        /// `yes` where `m` is set, else `no`.
        fn select(m: M, yes: R, no: R) -> R = _mm256_blendv_epi8(no, yes, m);
        fn load_mask(p: *const i32) -> M = _mm256_loadu_si256(p.cast());
        fn mask_and(a: M, b: M) -> M = _mm256_and_si256(a, b);
        fn mask_or(a: M, b: M) -> M = _mm256_or_si256(a, b);
        /// The sign bit of each 32-bit lane (`vmovmskps`; the bit pattern is
        /// reinterpreted, not converted).
        fn as_ps(a: M) -> __m256 = _mm256_castsi256_ps(a);
        fn movemask(a: __m256) -> i32 = _mm256_movemask_ps(a);
        fn sra31(a: R) -> R = _mm256_srai_epi32::<31>(a);
        fn sra16(a: R) -> R = _mm256_srai_epi32::<16>(a);
        fn mul_even(a: R, b: R) -> R = _mm256_mul_epi32(a, b);
        fn srl64_32(a: R) -> R = _mm256_srli_epi64::<32>(a);
        fn srl64_16(a: R) -> R = _mm256_srli_epi64::<16>(a);
        fn sll64_16(a: R) -> R = _mm256_slli_epi64::<16>(a);
        /// Dwords 1, 3, 5, 7 from `b`, the rest from `a`.
        fn blend_odd(a: R, b: R) -> R = _mm256_blend_epi32::<0b1010_1010>(a, b);
    }
    x86_composites!();

    /// # Safety
    /// AVX2 is present.
    #[inline(always)]
    pub(super) unsafe fn shl<const N: i32>(a: R) -> R {
        // SAFETY: this function's contract: one AVX2 intrinsic.
        unsafe { _mm256_slli_epi32::<N>(a) }
    }
    /// # Safety
    /// AVX2 is present.
    #[inline(always)]
    pub(super) unsafe fn sra<const N: i32>(a: R) -> R {
        // SAFETY: this function's contract: one AVX2 intrinsic.
        unsafe { _mm256_srai_epi32::<N>(a) }
    }
    /// # Safety
    /// AVX2 is present.
    #[inline(always)]
    pub(super) unsafe fn srl<const N: i32>(a: R) -> R {
        // SAFETY: this function's contract: one AVX2 intrinsic.
        unsafe { _mm256_srli_epi32::<N>(a) }
    }
}

/// SSE4.2: two `__m128i` per eight lanes (the 32-bit `pmuldq`, `pmulld`,
/// `pminsd`, `pminud` and `pblendvb` are SSE4.1, implied by SSE4.2).
#[cfg(target_arch = "x86_64")]
mod sse42 {
    #[cfg_attr(
        not(test),
        expect(clippy::wildcard_imports, reason = "the intrinsics of one level")
    )]
    use core::arch::x86_64::*;
    pub(super) type R = __m128i;
    pub(super) type M = __m128i;
    ops! {
        fn splat(v: i32) -> R = _mm_set1_epi32(v);
        fn load(p: *const i32) -> R = _mm_loadu_si128(p.cast());
        fn store(p: *mut i32, a: R) = _mm_storeu_si128(p.cast(), a);
        fn add(a: R, b: R) -> R = _mm_add_epi32(a, b);
        fn sub(a: R, b: R) -> R = _mm_sub_epi32(a, b);
        fn mul_lo(a: R, b: R) -> R = _mm_mullo_epi32(a, b);
        fn and(a: R, b: R) -> R = _mm_and_si128(a, b);
        fn or(a: R, b: R) -> R = _mm_or_si128(a, b);
        fn xor(a: R, b: R) -> R = _mm_xor_si128(a, b);
        fn eq(a: R, b: R) -> M = _mm_cmpeq_epi32(a, b);
        fn gt(a: R, b: R) -> M = _mm_cmpgt_epi32(a, b);
        fn min(a: R, b: R) -> R = _mm_min_epi32(a, b);
        fn max(a: R, b: R) -> R = _mm_max_epi32(a, b);
        fn min_u(a: R, b: R) -> R = _mm_min_epu32(a, b);
        fn max_u(a: R, b: R) -> R = _mm_max_epu32(a, b);
        /// `yes` where `m` is set, else `no`.
        fn select(m: M, yes: R, no: R) -> R = _mm_blendv_epi8(no, yes, m);
        fn load_mask(p: *const i32) -> M = _mm_loadu_si128(p.cast());
        fn mask_and(a: M, b: M) -> M = _mm_and_si128(a, b);
        fn mask_or(a: M, b: M) -> M = _mm_or_si128(a, b);
        /// The sign bit of each 32-bit lane (`movmskps`; the bit pattern is
        /// reinterpreted, not converted).
        fn as_ps(a: M) -> __m128 = _mm_castsi128_ps(a);
        fn movemask(a: __m128) -> i32 = _mm_movemask_ps(a);
        fn sra31(a: R) -> R = _mm_srai_epi32::<31>(a);
        fn sra16(a: R) -> R = _mm_srai_epi32::<16>(a);
        fn mul_even(a: R, b: R) -> R = _mm_mul_epi32(a, b);
        fn srl64_32(a: R) -> R = _mm_srli_epi64::<32>(a);
        fn srl64_16(a: R) -> R = _mm_srli_epi64::<16>(a);
        fn sll64_16(a: R) -> R = _mm_slli_epi64::<16>(a);
        /// Dwords 1 and 3 (words 2, 3, 6, 7) from `b`, the rest from `a`.
        fn blend_odd(a: R, b: R) -> R = _mm_blend_epi16::<0xCC>(a, b);
    }
    x86_composites!();

    /// # Safety
    /// SSE4.2 is present.
    #[inline(always)]
    pub(super) unsafe fn shl<const N: i32>(a: R) -> R {
        // SAFETY: this function's contract: one SSE2 intrinsic.
        unsafe { _mm_slli_epi32::<N>(a) }
    }
    /// # Safety
    /// SSE4.2 is present.
    #[inline(always)]
    pub(super) unsafe fn sra<const N: i32>(a: R) -> R {
        // SAFETY: this function's contract: one SSE2 intrinsic.
        unsafe { _mm_srai_epi32::<N>(a) }
    }
    /// # Safety
    /// SSE4.2 is present.
    #[inline(always)]
    pub(super) unsafe fn srl<const N: i32>(a: R) -> R {
        // SAFETY: this function's contract: one SSE2 intrinsic.
        unsafe { _mm_srli_epi32::<N>(a) }
    }
}

/// NEON: two `int32x4_t` per eight lanes.
#[cfg(target_arch = "aarch64")]
mod neon {
    #[cfg_attr(
        not(test),
        expect(clippy::wildcard_imports, reason = "the intrinsics of one level")
    )]
    use core::arch::aarch64::*;
    pub(super) type R = int32x4_t;
    pub(super) type M = uint32x4_t;
    type U = uint32x4_t;
    ops! {
        fn splat(v: i32) -> R = vdupq_n_s32(v);
        fn load(p: *const i32) -> R = vld1q_s32(p);
        fn store(p: *mut i32, a: R) = vst1q_s32(p, a);
        fn add(a: R, b: R) -> R = vaddq_s32(a, b);
        fn sub(a: R, b: R) -> R = vsubq_s32(a, b);
        fn mul_lo(a: R, b: R) -> R = vmulq_s32(a, b);
        fn and(a: R, b: R) -> R = vandq_s32(a, b);
        fn or(a: R, b: R) -> R = vorrq_s32(a, b);
        fn xor(a: R, b: R) -> R = veorq_s32(a, b);
        fn not(a: R) -> R = vmvnq_s32(a);
        fn eq(a: R, b: R) -> M = vceqq_s32(a, b);
        fn gt(a: R, b: R) -> M = vcgtq_s32(a, b);
        fn min(a: R, b: R) -> R = vminq_s32(a, b);
        fn max(a: R, b: R) -> R = vmaxq_s32(a, b);
        /// `yes` where `m` is set, else `no`.
        fn select(m: M, yes: R, no: R) -> R = vbslq_s32(m, yes, no);
        fn mask_and(a: M, b: M) -> M = vandq_u32(a, b);
        fn mask_or(a: M, b: M) -> M = vorrq_u32(a, b);
        fn mask_not(a: M) -> M = vmvnq_u32(a);
        fn sat_add(a: R, b: R) -> R = vqaddq_s32(a, b);
        fn sat_sub(a: R, b: R) -> R = vqsubq_s32(a, b);
        fn as_u(a: R) -> U = vreinterpretq_u32_s32(a);
        fn as_s(a: U) -> R = vreinterpretq_s32_u32(a);
        fn gt_u32(a: U, b: U) -> M = vcgtq_u32(a, b);
        fn min_u32(a: U, b: U) -> U = vminq_u32(a, b);
        fn max_u32(a: U, b: U) -> U = vmaxq_u32(a, b);
        fn ltz(a: R) -> M = vcltzq_s32(a);
        fn sign_bits(a: M) -> U = vshrq_n_u32::<31>(a);
        fn lane0(a: U) -> u32 = vgetq_lane_u32::<0>(a);
        fn lane1(a: U) -> u32 = vgetq_lane_u32::<1>(a);
        fn lane2(a: U) -> u32 = vgetq_lane_u32::<2>(a);
        fn lane3(a: U) -> u32 = vgetq_lane_u32::<3>(a);
        fn low_half(a: R) -> int32x2_t = vget_low_s32(a);
        fn mull_low(a: int32x2_t, b: int32x2_t) -> int64x2_t = vmull_s32(a, b);
        fn mull_high(a: R, b: R) -> int64x2_t = vmull_high_s32(a, b);
        /// Arithmetic shift right by 16 of each 64-bit product (floor).
        fn sra64_16(a: int64x2_t) -> int64x2_t = vshrq_n_s64::<16>(a);
        /// The saturating narrow to `i32`.
        fn narrow_sat(a: int64x2_t) -> int32x2_t = vqmovn_s64(a);
        fn combine(low: int32x2_t, high: int32x2_t) -> R = vcombine_s32(low, high);
    }

    /// Unsigned `a > b`.
    ///
    /// # Safety
    /// NEON is present.
    #[inline(always)]
    pub(super) unsafe fn gt_u(a: R, b: R) -> M {
        // SAFETY: NEON is present (module safety section).
        let a = unsafe { as_u(a) };
        // SAFETY: as above.
        let b = unsafe { as_u(b) };
        // SAFETY: as above.
        unsafe { gt_u32(a, b) }
    }
    /// Unsigned minimum.
    ///
    /// # Safety
    /// NEON is present.
    #[inline(always)]
    pub(super) unsafe fn min_u(a: R, b: R) -> R {
        // SAFETY: NEON is present (module safety section).
        let a = unsafe { as_u(a) };
        // SAFETY: as above.
        let b = unsafe { as_u(b) };
        // SAFETY: as above.
        let m = unsafe { min_u32(a, b) };
        // SAFETY: as above.
        unsafe { as_s(m) }
    }
    /// Unsigned maximum.
    ///
    /// # Safety
    /// NEON is present.
    #[inline(always)]
    pub(super) unsafe fn max_u(a: R, b: R) -> R {
        // SAFETY: NEON is present (module safety section).
        let a = unsafe { as_u(a) };
        // SAFETY: as above.
        let b = unsafe { as_u(b) };
        // SAFETY: as above.
        let m = unsafe { max_u32(a, b) };
        // SAFETY: as above.
        unsafe { as_s(m) }
    }
    /// A mask register from four `0` or `-1` words.
    ///
    /// # Safety
    /// NEON is present, and `p` is valid for four `i32`.
    #[inline(always)]
    pub(super) unsafe fn load_mask(p: *const i32) -> M {
        // SAFETY: NEON is present and `p` is valid for one register (this
        // function's contract; module safety section).
        let words = unsafe { load(p) };
        // SAFETY: NEON is present (module safety section).
        unsafe { as_u(words) }
    }
    /// The mask's lane bits, lane `i` in bit `i`.
    ///
    /// # Safety
    /// NEON is present.
    #[inline(always)]
    pub(super) unsafe fn bits(a: M) -> u8 {
        // SAFETY: NEON is present (module safety section).
        let one = unsafe { sign_bits(a) };
        // SAFETY: as above.
        let b0 = unsafe { lane0(one) };
        // SAFETY: as above.
        let b1 = unsafe { lane1(one) };
        // SAFETY: as above.
        let b2 = unsafe { lane2(one) };
        // SAFETY: as above.
        let b3 = unsafe { lane3(one) };
        let packed = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3);
        // Four lanes of 0 or 1, one bit each: at most 0b1111.
        debug_assert!(packed <= 0b1111, "mask bits {packed:#x}");
        #[expect(
            clippy::cast_possible_truncation,
            reason = "four one-bit lanes fit in u8"
        )]
        let out = packed as u8;
        out
    }
    /// A signed overflow of `s = a + b` (as for the x86 levels).
    ///
    /// # Safety
    /// NEON is present.
    #[inline(always)]
    pub(super) unsafe fn add_overflow(a: R, b: R, s: R) -> M {
        // SAFETY: NEON is present (module safety section).
        let a_flips = unsafe { xor(a, s) };
        // SAFETY: as above.
        let b_flips = unsafe { xor(b, s) };
        // SAFETY: as above.
        let both = unsafe { and(a_flips, b_flips) };
        // SAFETY: as above.
        unsafe { ltz(both) }
    }
    /// A signed overflow of `d = a - b` (as for the x86 levels).
    ///
    /// # Safety
    /// NEON is present.
    #[inline(always)]
    pub(super) unsafe fn sub_overflow(a: R, b: R, d: R) -> M {
        // SAFETY: NEON is present (module safety section).
        let operands_differ = unsafe { xor(a, b) };
        // SAFETY: as above.
        let result_flips = unsafe { xor(a, d) };
        // SAFETY: as above.
        let both = unsafe { and(operands_differ, result_flips) };
        // SAFETY: as above.
        unsafe { ltz(both) }
    }
    /// `Fx32::saturating_mul`: exact 64-bit products, an arithmetic shift
    /// right by 16 (floor), and the saturating narrow.
    ///
    /// # Safety
    /// NEON is present.
    #[inline(always)]
    pub(super) unsafe fn fx_sat_mul(a: R, b: R) -> R {
        // SAFETY: NEON is present (module safety section).
        let a_low = unsafe { low_half(a) };
        // SAFETY: as above.
        let b_low = unsafe { low_half(b) };
        // SAFETY: as above.
        let low = unsafe { mull_low(a_low, b_low) };
        // SAFETY: as above.
        let high = unsafe { mull_high(a, b) };
        // SAFETY: as above.
        let low = unsafe { sra64_16(low) };
        // SAFETY: as above.
        let high = unsafe { sra64_16(high) };
        // SAFETY: as above.
        let low = unsafe { narrow_sat(low) };
        // SAFETY: as above.
        let high = unsafe { narrow_sat(high) };
        // SAFETY: as above.
        unsafe { combine(low, high) }
    }

    /// # Safety
    /// NEON is present.
    #[inline(always)]
    pub(super) unsafe fn shl<const N: i32>(a: R) -> R {
        // SAFETY: this function's contract: one NEON intrinsic.
        unsafe { vshlq_n_s32::<N>(a) }
    }
    /// # Safety
    /// NEON is present.
    #[inline(always)]
    pub(super) unsafe fn sra<const N: i32>(a: R) -> R {
        // SAFETY: this function's contract: one NEON intrinsic.
        unsafe { vshrq_n_s32::<N>(a) }
    }
    /// # Safety
    /// NEON is present.
    #[inline(always)]
    pub(super) unsafe fn srl<const N: i32>(a: R) -> R {
        // SAFETY: NEON is present (this function's contract).
        let bits = unsafe { as_u(a) };
        // SAFETY: as above.
        let shifted = unsafe { vshrq_n_u32::<N>(bits) };
        // SAFETY: as above.
        unsafe { as_s(shifted) }
    }
}

#[cfg(target_arch = "x86_64")]
register_int_lanes!(Avx2I32, Avx2Fx32, avx2, (a0 b0 c0 0));
#[cfg(target_arch = "x86_64")]
register_int_lanes!(Sse42I32, Sse42Fx32, sse42, (a0 b0 c0 0) (a1 b1 c1 4));
#[cfg(target_arch = "aarch64")]
register_int_lanes!(NeonI32, NeonFx32, neon, (a0 b0 c0 0) (a1 b1 c1 4));

/// One level's lane types: `f64`, `i32` and `Fx32` over the same registers.
///
/// Sealed: only this crate's levels implement it, so a value of any of its
/// types exists only inside [`LaneKernel::run`].
pub trait Lanes: sealed::Sealed {
    /// The level these types belong to.
    const LEVEL: crate::Level;
    /// The level's `f64` lanes.
    type F64: tessera_rl_num::math::lane::F64Lanes;
    /// The level's `i32` lanes.
    type I32: I32Lanes;
    /// The level's `Fx32` lanes, over [`Self::I32`].
    type Fx32: Fx32Lanes<Raw = Self::I32>;
}

/// A kernel generic over a level's lane family.
pub trait LaneKernel {
    /// The kernel's result.
    type Output;
    /// Runs the kernel. Implementations must be `#[inline(always)]`, or the
    /// lane operations leave the `target_feature` context.
    fn run<S: Lanes>(self) -> Self::Output;
}

mod sealed {
    pub trait Sealed {}
}

macro_rules! family {
    ($name:ident, $level:ident, $f64:ty, $i32:ty, $fx:ty) => {
        pub(crate) enum $name {}
        impl sealed::Sealed for $name {}
        impl Lanes for $name {
            const LEVEL: crate::Level = crate::Level::$level;
            type F64 = $f64;
            type I32 = $i32;
            type Fx32 = $fx;
        }
    };
}

family!(
    ScalarFamily,
    Scalar,
    crate::lanes::Scalar,
    ScalarI32,
    ScalarFx32
);
#[cfg(target_arch = "x86_64")]
family!(Avx2Family, Avx2, crate::lanes::Avx2Lanes, Avx2I32, Avx2Fx32);
#[cfg(target_arch = "x86_64")]
family!(
    Sse42Family,
    Sse42,
    crate::lanes::Sse42Lanes,
    Sse42I32,
    Sse42Fx32
);
#[cfg(target_arch = "aarch64")]
family!(NeonFamily, Neon, crate::lanes::NeonLanes, NeonI32, NeonFx32);

/// Runs `kernel` on plain arrays.
#[inline(never)]
pub(crate) fn run_scalar_lanes<K: LaneKernel>(kernel: K) -> K::Output {
    kernel.run::<ScalarFamily>()
}

/// # Safety
/// The CPU has AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
pub(crate) unsafe fn run_avx2_lanes<K: LaneKernel>(kernel: K) -> K::Output {
    kernel.run::<Avx2Family>()
}

/// # Safety
/// The CPU has SSE4.2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[inline(never)]
pub(crate) unsafe fn run_sse42_lanes<K: LaneKernel>(kernel: K) -> K::Output {
    kernel.run::<Sse42Family>()
}

/// # Safety
/// The CPU has NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline(never)]
pub(crate) unsafe fn run_neon_lanes<K: LaneKernel>(kernel: K) -> K::Output {
    kernel.run::<NeonFamily>()
}
