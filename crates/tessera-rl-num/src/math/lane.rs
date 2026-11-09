// SPDX-License-Identifier: MIT OR Apache-2.0

//! Frozen sine and cosine over eight `f64` lanes, for a safe lane trait.
//!
//! The expression trees and coefficients are those of the scalar kernels
//! (`vendor::k_sin`, `vendor::k_cos`), and the domain is the scalar's own
//! high-word test. This module holds no intrinsics; `tessera-rl-simd`
//! implements [`F64Lanes`] per instruction set.
//!
//! [`sin`] and [`cos`] are defined only where [`in_polynomial_domain`] holds.
//! Any other lane must take the scalar [`crate::math::sin`] or
//! [`crate::math::cos`]: they reduce the argument, and these do not.

#![expect(
    clippy::inline_always,
    reason = "must inline into each target_feature kernel"
)]

use core::ops::{Add, Div, Mul, Sub};

use super::vendor::{k_cos as c, k_sin as s};

/// Eight `f64` lanes. Arithmetic is IEEE per element, with no contraction and
/// no estimate; comparisons are ordered (a NaN compares false).
pub trait F64Lanes:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    /// One boolean per lane.
    type Mask: Copy;

    /// Every lane equal to `v`.
    fn splat(v: f64) -> Self;
    /// Lanes from an array.
    fn from_array(v: [f64; 8]) -> Self;
    /// Lanes to an array, bits unchanged.
    fn to_array(self) -> [f64; 8];
    /// The conversion to `f32` per lane, rounding as the FP mode says.
    fn to_f32_array(self) -> [f32; 8];
    /// Clears each sign bit.
    #[must_use]
    fn abs(self) -> Self;
    /// `self <= rhs`.
    fn le(self, rhs: Self) -> Self::Mask;
    /// `self < rhs`.
    fn lt(self, rhs: Self) -> Self::Mask;
    /// `self > rhs`.
    fn gt(self, rhs: Self) -> Self::Mask;
    /// `yes` where `mask` holds, else `no`; bits unchanged.
    fn select(mask: Self::Mask, yes: Self, no: Self) -> Self;
    /// A mask from one boolean per lane.
    fn mask_from_array(v: [bool; 8]) -> Self::Mask;
    /// Lane-wise and.
    fn mask_and(a: Self::Mask, b: Self::Mask) -> Self::Mask;
    /// Lane-wise or.
    fn mask_or(a: Self::Mask, b: Self::Mask) -> Self::Mask;
    /// Bit `i` set if lane `i` holds.
    fn mask_bits(mask: Self::Mask) -> u8;
}

/// High word of `|x|` at the top of the polynomial path.
const POLYNOMIAL_WORD: u32 = 0x3fe9_21fb;
/// Below this high word, `sin(x)` is `x`.
const SIN_TINY_WORD: u32 = 0x3e50_0000;
/// Below this high word, `cos(x)` is 1.
const COS_TINY_WORD: u32 = 0x3e46_a09e;

/// `high word of |x| <= word`.
#[must_use]
#[inline(always)]
pub fn high_word_le<L: F64Lanes>(x: L, word: u32) -> L::Mask {
    x.abs().le(L::splat(f64::from_bits(
        (u64::from(word) << 32) | 0xffff_ffff,
    )))
}

#[inline(always)]
fn high_word_lt<L: F64Lanes>(x: L, word: u32) -> L::Mask {
    x.abs().lt(L::splat(f64::from_bits(u64::from(word) << 32)))
}

/// Lanes on the scalar's polynomial path, for both functions.
#[must_use]
#[inline(always)]
pub fn in_polynomial_domain<L: F64Lanes>(x: L) -> L::Mask {
    high_word_le(x, POLYNOMIAL_WORD)
}

/// Sine, as the scalar on its polynomial path.
#[must_use]
#[expect(
    clippy::many_single_char_names,
    reason = "the names of the scalar kernels"
)]
#[inline(always)]
pub fn sin<L: F64Lanes>(x: L) -> L {
    let k = L::splat;
    let z = x * x;
    let w = z * z;
    let r = k(s::S2) + z * (k(s::S3) + z * k(s::S4)) + z * w * (k(s::S5) + z * k(s::S6));
    let v = z * x;
    let polynomial = x + v * (k(s::S1) + z * r);
    L::select(high_word_lt(x, SIN_TINY_WORD), x, polynomial)
}

/// Cosine, as the scalar on its polynomial path.
#[must_use]
#[expect(
    clippy::many_single_char_names,
    reason = "the names of the scalar kernels"
)]
#[inline(always)]
pub fn cos<L: F64Lanes>(x: L) -> L {
    let k = L::splat;
    let y = k(0.0);
    let z = x * x;
    let w = z * z;
    let r = z * (k(c::C1) + z * (k(c::C2) + z * k(c::C3)))
        + w * w * (k(c::C4) + z * (k(c::C5) + z * k(c::C6)));
    let hz = k(0.5) * z;
    let w = k(1.0) - hz;
    let polynomial = w + (((k(1.0) - w) - hz) + (z * r - x * y));
    L::select(high_word_lt(x, COS_TINY_WORD), k(1.0), polynomial)
}
