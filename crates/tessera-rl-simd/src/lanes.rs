// SPDX-License-Identifier: MIT OR Apache-2.0

//! The lane types. Each is private: a value exists only inside
//! [`Kernel::run`](crate::Kernel::run), entered through a `run_*` function
//! after the level's features were detected, which is what makes the
//! intrinsic calls below sound.
//!
//! Operations are `#[inline(always)]` so they compile inside the kernel's
//! `target_feature` function. Each is one IEEE instruction per element;
//! no reciprocal estimate, no fused operation.

#![expect(
    unsafe_code,
    reason = "the audited simd module: intrinsics over checked levels"
)]
#![expect(
    clippy::inline_always,
    reason = "operations must inline into the target_feature kernel"
)]

use core::ops::{Add, Div, Mul, Sub};

use tessera_rl_num::math::lane::F64Lanes;

use crate::Kernel;

const fn bit_mask(v: bool) -> u64 {
    if v { u64::MAX } else { 0 }
}

/// Scalar lanes.
#[derive(Clone, Copy)]
struct Scalar([f64; 8]);

#[inline(always)]
fn zip(a: [f64; 8], b: [f64; 8], f: impl Fn(f64, f64) -> f64) -> [f64; 8] {
    let mut out = a;
    for (o, b) in out.iter_mut().zip(b) {
        *o = f(*o, b);
    }
    out
}

macro_rules! scalar_op {
    ($tr:ident, $f:ident, $op:tt) => {
        impl $tr for Scalar {
            type Output = Self;
            #[inline(always)]
            fn $f(self, rhs: Self) -> Self {
                Self(zip(self.0, rhs.0, |a, b| a $op b))
            }
        }
    };
}
scalar_op!(Add, add, +);
scalar_op!(Sub, sub, -);
scalar_op!(Mul, mul, *);
scalar_op!(Div, div, /);

#[inline(always)]
fn compare(a: [f64; 8], b: [f64; 8], f: impl Fn(f64, f64) -> bool) -> [bool; 8] {
    let mut out = [false; 8];
    for ((o, a), b) in out.iter_mut().zip(a).zip(b) {
        *o = f(a, b);
    }
    out
}

impl F64Lanes for Scalar {
    type Mask = [bool; 8];

    #[inline(always)]
    fn splat(v: f64) -> Self {
        Self([v; 8])
    }
    #[inline(always)]
    fn from_array(v: [f64; 8]) -> Self {
        Self(v)
    }
    #[inline(always)]
    fn to_array(self) -> [f64; 8] {
        self.0
    }
    #[inline(always)]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the f64 to f32 conversion under test"
    )]
    fn to_f32_array(self) -> [f32; 8] {
        self.0.map(|v| v as f32)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        Self(self.0.map(f64::abs))
    }
    #[inline(always)]
    fn le(self, rhs: Self) -> [bool; 8] {
        compare(self.0, rhs.0, |a, b| a <= b)
    }
    #[inline(always)]
    fn lt(self, rhs: Self) -> [bool; 8] {
        compare(self.0, rhs.0, |a, b| a < b)
    }
    #[inline(always)]
    fn gt(self, rhs: Self) -> [bool; 8] {
        compare(self.0, rhs.0, |a, b| a > b)
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
    fn mask_bits(mask: [bool; 8]) -> u8 {
        mask.iter()
            .enumerate()
            .fold(0, |bits, (i, &m)| bits | (u8::from(m) << i))
    }
}

/// Runs `kernel` on plain arrays.
#[inline(never)]
pub(crate) fn run_scalar<K: Kernel>(kernel: K) -> K::Output {
    kernel.run::<Scalar>()
}

/// A lane type of `$width`-element registers.
///
/// `$a`, `$b`, `$c`: identifiers, one set per register; `$i`: the register's offset
/// into the eight-element arrays, in elements.
macro_rules! register_lanes {
    ($name:ident, $m:ident, $(($a:ident $b:ident $c:ident $i:literal))+) => {
        #[derive(Clone, Copy)]
        struct $name([$m::F; { [$($i),+].len() }]);

        register_lanes!(@op $name, $m, Add, add, [$(($a $b))+]);
        register_lanes!(@op $name, $m, Sub, sub, [$(($a $b))+]);
        register_lanes!(@op $name, $m, Mul, mul, [$(($a $b))+]);
        register_lanes!(@op $name, $m, Div, div, [$(($a $b))+]);

        impl F64Lanes for $name {
            type Mask = [$m::M; { [$($i),+].len() }];

            #[inline(always)]
            fn splat(v: f64) -> Self {
                // SAFETY: see the module documentation.
                Self([$({ let _ = $i; unsafe { $m::splat(v) } }),+])
            }
            #[inline(always)]
            fn from_array(v: [f64; 8]) -> Self {
                let p = v.as_ptr();
                // SAFETY: `$i + WIDTH <= 8`, so each load is inside `v`;
                // the level is checked (module documentation).
                Self([$(unsafe { $m::load(p.add($i)) }),+])
            }
            #[inline(always)]
            fn to_array(self) -> [f64; 8] {
                let mut out = [0.0; 8];
                let p = out.as_mut_ptr();
                let Self([$($a),+]) = self;
                // SAFETY: `$i + WIDTH <= 8`, so each store is inside `out`;
                // the level is checked (module documentation).
                $(unsafe { $m::store(p.add($i), $a) };)+
                out
            }
            #[inline(always)]
            fn to_f32_array(self) -> [f32; 8] {
                let mut out = [0.0; 8];
                let p = out.as_mut_ptr();
                let Self([$($a),+]) = self;
                // SAFETY: `$i + WIDTH <= 8`, so each store is inside `out`;
                // the level is checked (module documentation).
                $(unsafe { $m::store_f32(p.add($i), $a) };)+
                out
            }
            #[inline(always)]
            fn abs(self) -> Self {
                let Self([$($a),+]) = self;
                // SAFETY: see the module documentation.
                Self([$(unsafe { $m::abs($a) }),+])
            }
            #[inline(always)]
            fn le(self, rhs: Self) -> Self::Mask {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: see the module documentation.
                [$(unsafe { $m::le($a, $b) }),+]
            }
            #[inline(always)]
            fn lt(self, rhs: Self) -> Self::Mask {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: see the module documentation.
                [$(unsafe { $m::lt($a, $b) }),+]
            }
            #[inline(always)]
            fn gt(self, rhs: Self) -> Self::Mask {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: see the module documentation.
                [$(unsafe { $m::gt($a, $b) }),+]
            }
            #[inline(always)]
            fn select(mask: Self::Mask, yes: Self, no: Self) -> Self {
                let ([$($c),+], Self([$($a),+]), Self([$($b),+])) = (mask, yes, no);
                // SAFETY: see the module documentation.
                Self([$(unsafe { $m::select($c, $a, $b) }),+])
            }
            #[inline(always)]
            fn mask_from_array(v: [bool; 8]) -> Self::Mask {
                let bits = v.map(bit_mask);
                let p = bits.as_ptr();
                // SAFETY: `$i + WIDTH <= 8`, so each load is inside `bits`;
                // the level is checked (module documentation).
                [$(unsafe { $m::load_mask(p.add($i)) }),+]
            }
            #[inline(always)]
            fn mask_and(a: Self::Mask, b: Self::Mask) -> Self::Mask {
                let ([$($a),+], [$($b),+]) = (a, b);
                // SAFETY: see the module documentation.
                [$(unsafe { $m::and($a, $b) }),+]
            }
            #[inline(always)]
            fn mask_or(a: Self::Mask, b: Self::Mask) -> Self::Mask {
                let ([$($a),+], [$($b),+]) = (a, b);
                // SAFETY: see the module documentation.
                [$(unsafe { $m::or($a, $b) }),+]
            }
            #[inline(always)]
            fn mask_bits(mask: Self::Mask) -> u8 {
                let [$($a),+] = mask;
                // SAFETY: see the module documentation.
                0 $(| (unsafe { $m::bits($a) } << $i))+
            }
        }
    };
    (@op $name:ident, $m:ident, $tr:ident, $f:ident, [$(($a:ident $b:ident))+]) => {
        impl $tr for $name {
            type Output = Self;
            #[inline(always)]
            fn $f(self, rhs: Self) -> Self {
                let (Self([$($a),+]), Self([$($b),+])) = (self, rhs);
                // SAFETY: see the module documentation.
                Self([$(unsafe { $m::$f($a, $b) }),+])
            }
        }
    };
}

/// Wrappers over the intrinsics: each `unsafe fn` requires its level.
macro_rules! ops {
    ($($(#[$attr:meta])* fn $name:ident($($p:ident: $t:ty),*) $(-> $r:ty)? = $e:expr;)+) => {
        $(
            $(#[$attr])*
            /// # Safety
            /// The level's features are present, and any pointer is valid
            /// for one register.
            #[inline(always)]
            pub(super) unsafe fn $name($($p: $t),*) $(-> $r)? {
                // SAFETY: this function's contract.
                unsafe { $e }
            }
        )+
    };
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    #[cfg_attr(
        not(test),
        expect(clippy::wildcard_imports, reason = "the intrinsics of one level")
    )]
    use core::arch::x86_64::*;
    pub(super) type F = __m256d;
    pub(super) type M = __m256d;
    ops! {
        fn splat(v: f64) -> F = _mm256_set1_pd(v);
        fn load(p: *const f64) -> F = _mm256_loadu_pd(p);
        fn store(p: *mut f64, a: F) = _mm256_storeu_pd(p, a);
        fn store_f32(p: *mut f32, a: F) = _mm_storeu_ps(p, _mm256_cvtpd_ps(a));
        fn add(a: F, b: F) -> F = _mm256_add_pd(a, b);
        fn sub(a: F, b: F) -> F = _mm256_sub_pd(a, b);
        fn mul(a: F, b: F) -> F = _mm256_mul_pd(a, b);
        fn div(a: F, b: F) -> F = _mm256_div_pd(a, b);
        fn abs(a: F) -> F = _mm256_andnot_pd(_mm256_set1_pd(-0.0), a);
        fn le(a: F, b: F) -> M = _mm256_cmp_pd::<_CMP_LE_OQ>(a, b);
        fn lt(a: F, b: F) -> M = _mm256_cmp_pd::<_CMP_LT_OQ>(a, b);
        fn gt(a: F, b: F) -> M = _mm256_cmp_pd::<_CMP_GT_OQ>(a, b);
        fn select(m: M, yes: F, no: F) -> F = _mm256_blendv_pd(no, yes, m);
        fn load_mask(p: *const u64) -> M = _mm256_loadu_pd(p.cast());
        fn and(a: M, b: M) -> M = _mm256_and_pd(a, b);
        fn or(a: M, b: M) -> M = _mm256_or_pd(a, b);
        fn bits(a: M) -> u8 = u8::try_from(_mm256_movemask_pd(a)).unwrap_or(0);
    }
}

#[cfg(target_arch = "x86_64")]
mod sse42 {
    #[cfg_attr(
        not(test),
        expect(clippy::wildcard_imports, reason = "the intrinsics of one level")
    )]
    use core::arch::x86_64::*;
    pub(super) type F = __m128d;
    pub(super) type M = __m128d;
    ops! {
        fn splat(v: f64) -> F = _mm_set1_pd(v);
        fn load(p: *const f64) -> F = _mm_loadu_pd(p);
        fn store(p: *mut f64, a: F) = _mm_storeu_pd(p, a);
        fn store_f32(p: *mut f32, a: F) = _mm_store_sd(p.cast(), _mm_castps_pd(_mm_cvtpd_ps(a)));
        fn add(a: F, b: F) -> F = _mm_add_pd(a, b);
        fn sub(a: F, b: F) -> F = _mm_sub_pd(a, b);
        fn mul(a: F, b: F) -> F = _mm_mul_pd(a, b);
        fn div(a: F, b: F) -> F = _mm_div_pd(a, b);
        fn abs(a: F) -> F = _mm_andnot_pd(_mm_set1_pd(-0.0), a);
        fn le(a: F, b: F) -> M = _mm_cmple_pd(a, b);
        fn lt(a: F, b: F) -> M = _mm_cmplt_pd(a, b);
        fn gt(a: F, b: F) -> M = _mm_cmpgt_pd(a, b);
        fn select(m: M, yes: F, no: F) -> F = _mm_blendv_pd(no, yes, m);
        fn load_mask(p: *const u64) -> M = _mm_loadu_pd(p.cast());
        fn and(a: M, b: M) -> M = _mm_and_pd(a, b);
        fn or(a: M, b: M) -> M = _mm_or_pd(a, b);
        fn bits(a: M) -> u8 = u8::try_from(_mm_movemask_pd(a)).unwrap_or(0);
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    #[cfg_attr(
        not(test),
        expect(clippy::wildcard_imports, reason = "the intrinsics of one level")
    )]
    use core::arch::aarch64::*;
    pub(super) type F = float64x2_t;
    pub(super) type M = uint64x2_t;
    ops! {
        fn splat(v: f64) -> F = vdupq_n_f64(v);
        fn load(p: *const f64) -> F = vld1q_f64(p);
        fn store(p: *mut f64, a: F) = vst1q_f64(p, a);
        fn store_f32(p: *mut f32, a: F) = vst1_f32(p, vcvt_f32_f64(a));
        fn add(a: F, b: F) -> F = vaddq_f64(a, b);
        fn sub(a: F, b: F) -> F = vsubq_f64(a, b);
        fn mul(a: F, b: F) -> F = vmulq_f64(a, b);
        fn div(a: F, b: F) -> F = vdivq_f64(a, b);
        fn abs(a: F) -> F = vabsq_f64(a);
        fn le(a: F, b: F) -> M = vcleq_f64(a, b);
        fn lt(a: F, b: F) -> M = vcltq_f64(a, b);
        fn gt(a: F, b: F) -> M = vcgtq_f64(a, b);
        fn select(m: M, yes: F, no: F) -> F = vbslq_f64(m, yes, no);
        fn load_mask(p: *const u64) -> M = vld1q_u64(p);
        fn and(a: M, b: M) -> M = vandq_u64(a, b);
        fn or(a: M, b: M) -> M = vorrq_u64(a, b);
        fn bits(a: M) -> u8 = u8::try_from((vgetq_lane_u64::<0>(a) & 1) | (vgetq_lane_u64::<1>(a) & 2)).unwrap_or(0);
    }
}

#[cfg(target_arch = "x86_64")]
register_lanes!(Avx2Lanes, avx2, (a0 b0 c0 0) (a1 b1 c1 4));
#[cfg(target_arch = "x86_64")]
register_lanes!(Sse42Lanes, sse42, (a0 b0 c0 0) (a1 b1 c1 2) (a2 b2 c2 4) (a3 b3 c3 6));
#[cfg(target_arch = "aarch64")]
register_lanes!(NeonLanes, neon, (a0 b0 c0 0) (a1 b1 c1 2) (a2 b2 c2 4) (a3 b3 c3 6));

/// # Safety
/// The CPU has AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
pub(crate) unsafe fn run_avx2<K: Kernel>(kernel: K) -> K::Output {
    kernel.run::<Avx2Lanes>()
}

/// # Safety
/// The CPU has SSE4.2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[inline(never)]
pub(crate) unsafe fn run_sse42<K: Kernel>(kernel: K) -> K::Output {
    kernel.run::<Sse42Lanes>()
}

/// # Safety
/// The CPU has NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline(never)]
pub(crate) unsafe fn run_neon<K: Kernel>(kernel: K) -> K::Output {
    kernel.run::<NeonLanes>()
}
