// SPDX-License-Identifier: MIT OR Apache-2.0

//! Eight `i32` and eight `Fx32` lanes, for safe integer and fixed-point lane
//! kernels (ADR-0090 decision 5, stage 2; ADR-0097 decision 3.3).
//!
//! Every operation is defined per lane as the named scalar operation of
//! `i32`, `u32` or [`Fx32`], bit for bit, at every level: lane `i` of the
//! result equals the scalar operation on lane `i` of the operands. The
//! operation set is ADR-0097 decision 3.3's closed set, plus
//! `overflowing_add` and `overflowing_sub`, which combine 3.3's wrapping add
//! and sub with a comparison-built overflow mask. Operations are named, with no
//! operator overloads, so no wrapping is silent; there is no `checked_*`
//! returning `Option`, because lane kernels are infallible: a kernel takes
//! the documented wrapping or saturating form, or the overflow mask and a
//! [`I32Lanes::select`].
//!
//! This module holds no intrinsics; `tessera-rl-simd` implements the traits
//! per instruction set, with the scalar implementation as the reference.

#![expect(
    clippy::inline_always,
    reason = "must inline into each target_feature kernel"
)]

use crate::Fx32;

/// Eight `i32` lanes (also `u32` columns, read as bits).
pub trait I32Lanes: Copy {
    /// One boolean per lane.
    type Mask: Copy;

    /// Every lane `v`.
    #[must_use]
    fn splat(v: i32) -> Self;
    /// The lanes of `v`, in order.
    #[must_use]
    fn from_array(v: [i32; 8]) -> Self;
    /// The lanes, in order.
    #[must_use]
    fn to_array(self) -> [i32; 8];

    /// Lane `I` (a compile-time index below 8).
    #[inline(always)]
    #[must_use]
    fn extract<const I: usize>(self) -> i32 {
        const { assert!(I < 8, "lane index out of range") };
        self.to_array().get(I).copied().unwrap_or(0)
    }
    /// The lanes with lane `I` replaced by `v`.
    #[inline(always)]
    #[must_use]
    fn replace<const I: usize>(self, v: i32) -> Self {
        const { assert!(I < 8, "lane index out of range") };
        let mut a = self.to_array();
        if let Some(slot) = a.get_mut(I) {
            *slot = v;
        }
        Self::from_array(a)
    }

    /// `i32::wrapping_add` per lane (also `u32::wrapping_add` on the bits).
    #[must_use]
    fn wrapping_add(self, rhs: Self) -> Self;
    /// `i32::wrapping_sub` per lane.
    #[must_use]
    fn wrapping_sub(self, rhs: Self) -> Self;
    /// `i32::wrapping_mul` per lane (the low 32 bits of the product).
    #[must_use]
    fn wrapping_mul(self, rhs: Self) -> Self;
    /// `i32::wrapping_add` per lane, and the lanes where
    /// `i32::checked_add` is `None`.
    ///
    /// A convenience over ADR-0097 3.3's ops: with `w = a.wrapping_add(b)`,
    /// the mask is the sign test `((a ^ w) & (b ^ w)) < 0`.
    #[must_use]
    fn overflowing_add(self, rhs: Self) -> (Self, Self::Mask);
    /// `i32::wrapping_sub` per lane, and the lanes where
    /// `i32::checked_sub` is `None`.
    ///
    /// A convenience over 3.3's ops: with `w = a.wrapping_sub(b)`, the mask
    /// is the sign test `((a ^ b) & (a ^ w)) < 0`.
    #[must_use]
    fn overflowing_sub(self, rhs: Self) -> (Self, Self::Mask);

    /// Bitwise and.
    #[must_use]
    fn and(self, rhs: Self) -> Self;
    /// Bitwise or.
    #[must_use]
    fn or(self, rhs: Self) -> Self;
    /// Bitwise exclusive or.
    #[must_use]
    fn xor(self, rhs: Self) -> Self;
    /// Bitwise not.
    #[must_use]
    fn not(self) -> Self;
    /// `x << N` per lane, `N` in `1..32`.
    #[must_use]
    fn shl<const N: i32>(self) -> Self;
    /// `x >> N` on `i32` per lane (arithmetic: rounds toward minus
    /// infinity), `N` in `1..32`. On an `Fx32`'s raw bits with `N = 16` it is
    /// the floor to an integer cell, `fx.to_raw() >> 16`.
    #[must_use]
    fn shr_arith<const N: i32>(self) -> Self;
    /// `x >> N` on the `u32` bits per lane (logical), `N` in `1..32`.
    #[must_use]
    fn shr_logical<const N: i32>(self) -> Self;

    /// `a == b` per lane.
    #[must_use]
    fn eq(self, rhs: Self) -> Self::Mask;
    /// `a != b` per lane.
    #[inline(always)]
    #[must_use]
    fn ne(self, rhs: Self) -> Self::Mask {
        Self::mask_not(self.eq(rhs))
    }
    /// `a < b` on `i32`, per lane.
    #[inline(always)]
    #[must_use]
    fn lt(self, rhs: Self) -> Self::Mask {
        rhs.gt(self)
    }
    /// `a <= b` on `i32`, per lane.
    #[inline(always)]
    #[must_use]
    fn le(self, rhs: Self) -> Self::Mask {
        Self::mask_not(self.gt(rhs))
    }
    /// `a > b` on `i32`, per lane.
    #[must_use]
    fn gt(self, rhs: Self) -> Self::Mask;
    /// `a >= b` on `i32`, per lane.
    #[inline(always)]
    #[must_use]
    fn ge(self, rhs: Self) -> Self::Mask {
        Self::mask_not(rhs.gt(self))
    }
    /// `a < b` on the `u32` bits, per lane.
    #[inline(always)]
    #[must_use]
    fn lt_u(self, rhs: Self) -> Self::Mask {
        rhs.gt_u(self)
    }
    /// `a <= b` on the `u32` bits, per lane.
    #[inline(always)]
    #[must_use]
    fn le_u(self, rhs: Self) -> Self::Mask {
        Self::mask_not(self.gt_u(rhs))
    }
    /// `a > b` on the `u32` bits, per lane.
    #[must_use]
    fn gt_u(self, rhs: Self) -> Self::Mask;
    /// `a >= b` on the `u32` bits, per lane.
    #[inline(always)]
    #[must_use]
    fn ge_u(self, rhs: Self) -> Self::Mask {
        Self::mask_not(rhs.gt_u(self))
    }

    /// `i32::min` per lane.
    #[must_use]
    fn min(self, rhs: Self) -> Self;
    /// `i32::max` per lane.
    #[must_use]
    fn max(self, rhs: Self) -> Self;
    /// `u32::min` on the bits, per lane.
    #[must_use]
    fn min_u(self, rhs: Self) -> Self;
    /// `u32::max` on the bits, per lane.
    #[must_use]
    fn max_u(self, rhs: Self) -> Self;

    /// `yes` where `mask`, else `no`, per lane.
    #[must_use]
    fn select(mask: Self::Mask, yes: Self, no: Self) -> Self;
    /// A mask from booleans.
    #[must_use]
    fn mask_from_array(v: [bool; 8]) -> Self::Mask;
    /// A mask from bits: lane `i` is bit `i`.
    #[must_use]
    fn mask_from_bits(bits: u8) -> Self::Mask;
    /// Bit `i` set for lane `i`.
    #[must_use]
    fn mask_bits(mask: Self::Mask) -> u8;
    /// Lanewise and.
    #[must_use]
    fn mask_and(a: Self::Mask, b: Self::Mask) -> Self::Mask;
    /// Lanewise or.
    #[must_use]
    fn mask_or(a: Self::Mask, b: Self::Mask) -> Self::Mask;
    /// Lanewise not.
    #[must_use]
    fn mask_not(a: Self::Mask) -> Self::Mask;

    // Narrow loads, stores and the byte gather (ADR-0100 decision 4). The
    // defaults are the reference: per lane, the named scalar conversion.
    // `tessera-rl-simd` may override a load with an extend instruction,
    // tested bit for bit against these.

    /// The lanes of `v`, each sign-extended (`i32::from`).
    #[inline(always)]
    #[must_use]
    fn from_i16(v: [i16; 8]) -> Self {
        Self::from_array(v.map(i32::from))
    }
    /// The lanes of `v`, each zero-extended (`i32::from`).
    #[inline(always)]
    #[must_use]
    fn from_u16(v: [u16; 8]) -> Self {
        Self::from_array(v.map(i32::from))
    }
    /// The lanes of `v`, each zero-extended (`i32::from`).
    #[inline(always)]
    #[must_use]
    fn from_u8(v: [u8; 8]) -> Self {
        Self::from_array(v.map(i32::from))
    }
    /// The low 16 bits of each lane, as `i16`.
    ///
    /// Exact when every lane is in `i16`'s range, which the caller
    /// guarantees (a game-rule clamp or a bounded counter); a lane outside
    /// it fails a `debug_assert!` and, in release, keeps its low bits.
    #[inline(always)]
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "exact by contract (ADR-0100 D4); debug_assert checks the range"
    )]
    fn to_i16(self) -> [i16; 8] {
        self.to_array().map(|v| {
            debug_assert!(i16::try_from(v).is_ok(), "lane {v} is outside i16");
            v as i16
        })
    }
    /// The low 16 bits of each lane, as `u16`; exact in `u16`'s range, as
    /// [`I32Lanes::to_i16`].
    #[inline(always)]
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "exact by contract (ADR-0100 D4); debug_assert checks the range"
    )]
    fn to_u16(self) -> [u16; 8] {
        self.to_array().map(|v| {
            debug_assert!(u16::try_from(v).is_ok(), "lane {v} is outside u16");
            v as u16
        })
    }
    /// The low 8 bits of each lane, as `u8`; exact in `u8`'s range, as
    /// [`I32Lanes::to_i16`].
    #[inline(always)]
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "exact by contract (ADR-0100 D4); debug_assert checks the range"
    )]
    fn to_u8(self) -> [u8; 8] {
        self.to_array().map(|v| {
            debug_assert!(u8::try_from(v).is_ok(), "lane {v} is outside u8");
            v as u8
        })
    }
    /// `base[idx[i]]` zero-extended, per lane: a scalar gather at every
    /// level (ADR-0100 decision 4).
    ///
    /// Infallible: the caller guarantees every index is in bounds; a lane
    /// whose index is negative or past the end fails a `debug_assert!` and,
    /// in release, reads 0. That 0 only keeps the gather infallible: it is
    /// not a semantics a kernel may rely on, and a kernel that passes an
    /// out-of-range index is a bug its differential must catch.
    #[inline(always)]
    #[must_use]
    fn gather_u8(base: &[u8], idx: [i32; 8]) -> Self {
        Self::from_array(idx.map(|i| {
            let got = usize::try_from(i).ok().and_then(|i| base.get(i)).copied();
            debug_assert!(
                got.is_some(),
                "gather index {i} is outside 0..{}",
                base.len()
            );
            got.map_or(0, i32::from)
        }))
    }
}

/// The mask of an [`Fx32Lanes`] type: its raw lanes' mask.
pub type Fx32Mask<L> = <<L as Fx32Lanes>::Raw as I32Lanes>::Mask;

/// Eight [`Fx32`] lanes over an [`I32Lanes`] of their raw bits.
///
/// Every operation equals the named [`Fx32`] operation per lane, bit for
/// bit. Order and equality are those of the raw `i32` (as for [`Fx32`]).
pub trait Fx32Lanes: Copy {
    /// The raw `i32` lanes.
    type Raw: I32Lanes;

    /// From raw bits, unchanged.
    #[must_use]
    fn from_raw(raw: Self::Raw) -> Self;
    /// The raw bits, unchanged.
    #[must_use]
    fn to_raw(self) -> Self::Raw;

    /// `Fx32::saturating_add` per lane.
    #[must_use]
    fn saturating_add(self, rhs: Self) -> Self;
    /// `Fx32::saturating_sub` per lane.
    #[must_use]
    fn saturating_sub(self, rhs: Self) -> Self;
    /// `Fx32::saturating_mul` per lane: the 64-bit product shifted right by
    /// 16 (rounding toward minus infinity), clamped to the `Fx32` range.
    /// The only multiplication (ADR-0097 decision 3.3).
    #[must_use]
    fn saturating_mul(self, rhs: Self) -> Self;

    /// Every lane `v`.
    #[inline(always)]
    #[must_use]
    fn splat(v: Fx32) -> Self {
        Self::from_raw(Self::Raw::splat(v.to_raw()))
    }
    /// The lanes of `v`, in order.
    #[inline(always)]
    #[must_use]
    fn from_array(v: [Fx32; 8]) -> Self {
        Self::from_raw(Self::Raw::from_array(v.map(Fx32::to_raw)))
    }
    /// The lanes, in order.
    #[inline(always)]
    #[must_use]
    fn to_array(self) -> [Fx32; 8] {
        self.to_raw().to_array().map(Fx32::from_raw)
    }
    /// Lane `I` (a compile-time index below 8).
    #[inline(always)]
    #[must_use]
    fn extract<const I: usize>(self) -> Fx32 {
        Fx32::from_raw(self.to_raw().extract::<I>())
    }
    /// The lanes with lane `I` replaced by `v`.
    #[inline(always)]
    #[must_use]
    fn replace<const I: usize>(self, v: Fx32) -> Self {
        Self::from_raw(self.to_raw().replace::<I>(v.to_raw()))
    }

    /// `Fx32::wrapping_add` per lane.
    #[inline(always)]
    #[must_use]
    fn wrapping_add(self, rhs: Self) -> Self {
        Self::from_raw(self.to_raw().wrapping_add(rhs.to_raw()))
    }
    /// `Fx32::wrapping_sub` per lane.
    #[inline(always)]
    #[must_use]
    fn wrapping_sub(self, rhs: Self) -> Self {
        Self::from_raw(self.to_raw().wrapping_sub(rhs.to_raw()))
    }
    /// `Fx32::wrapping_add` per lane, and the lanes where
    /// `Fx32::checked_add` is `None`: `select(mask, old, sum)` is
    /// `checked_add(..).unwrap_or(old)`.
    #[inline(always)]
    #[must_use]
    fn overflowing_add(self, rhs: Self) -> (Self, Fx32Mask<Self>) {
        let (sum, mask) = self.to_raw().overflowing_add(rhs.to_raw());
        (Self::from_raw(sum), mask)
    }
    /// `Fx32::wrapping_sub` per lane, and the lanes where
    /// `Fx32::checked_sub` is `None`.
    #[inline(always)]
    #[must_use]
    fn overflowing_sub(self, rhs: Self) -> (Self, Fx32Mask<Self>) {
        let (diff, mask) = self.to_raw().overflowing_sub(rhs.to_raw());
        (Self::from_raw(diff), mask)
    }

    /// `a == b` per lane.
    #[inline(always)]
    #[must_use]
    fn eq(self, rhs: Self) -> Fx32Mask<Self> {
        self.to_raw().eq(rhs.to_raw())
    }
    /// `a != b` per lane.
    #[inline(always)]
    #[must_use]
    fn ne(self, rhs: Self) -> Fx32Mask<Self> {
        self.to_raw().ne(rhs.to_raw())
    }
    /// `a < b` per lane.
    #[inline(always)]
    #[must_use]
    fn lt(self, rhs: Self) -> Fx32Mask<Self> {
        self.to_raw().lt(rhs.to_raw())
    }
    /// `a <= b` per lane.
    #[inline(always)]
    #[must_use]
    fn le(self, rhs: Self) -> Fx32Mask<Self> {
        self.to_raw().le(rhs.to_raw())
    }
    /// `a > b` per lane.
    #[inline(always)]
    #[must_use]
    fn gt(self, rhs: Self) -> Fx32Mask<Self> {
        self.to_raw().gt(rhs.to_raw())
    }
    /// `a >= b` per lane.
    #[inline(always)]
    #[must_use]
    fn ge(self, rhs: Self) -> Fx32Mask<Self> {
        self.to_raw().ge(rhs.to_raw())
    }
    /// `Fx32::min` per lane (`Ord`).
    #[inline(always)]
    #[must_use]
    fn min(self, rhs: Self) -> Self {
        Self::from_raw(self.to_raw().min(rhs.to_raw()))
    }
    /// `Fx32::max` per lane (`Ord`).
    #[inline(always)]
    #[must_use]
    fn max(self, rhs: Self) -> Self {
        Self::from_raw(self.to_raw().max(rhs.to_raw()))
    }
    /// `yes` where `mask`, else `no`, per lane.
    #[inline(always)]
    #[must_use]
    fn select(mask: Fx32Mask<Self>, yes: Self, no: Self) -> Self {
        Self::from_raw(Self::Raw::select(mask, yes.to_raw(), no.to_raw()))
    }
}
