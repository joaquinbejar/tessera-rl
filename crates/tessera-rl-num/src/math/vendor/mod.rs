// SPDX-License-Identifier: MIT OR Apache-2.0

//! Vendored `libm` 0.2.16 sources: math version 1 (ADR-0008).
//!
//! The files in this module are kept verbatim so their results stay frozen.
//! It is the only module exempt from the indexing, cast and arithmetic rules
//! (global rules 9 and 10); the exemption is the `expect` below. Nothing
//! outside `crate::math` uses this module.
//!
//! The helpers defined here replace upstream support code without `unsafe`:
//! `force_eval!` uses `black_box` instead of a volatile read, `i!` uses plain
//! (bounds-checked) indexing instead of `get_unchecked`, and
//! `select_implementation!` expands to nothing because its only use selects
//! an x87 path for x86 without SSE, which Tessera does not target. `fabs`,
//! `floor` and `sqrt` are exact IEEE-754 operations, so the hardware
//! instructions give the same results as the upstream software versions.

#![expect(
    clippy::all,
    clippy::pedantic,
    clippy::indexing_slicing,
    clippy::unreachable,
    reason = "vendored libm 0.2.16, kept verbatim so results stay frozen (ADR-0008)"
)]

macro_rules! force_eval {
    ($e:expr) => {
        ::core::hint::black_box($e)
    };
}

macro_rules! i {
    ($array:expr, $index:expr) => {
        $array[$index]
    };
    ($array:expr, $index:expr, = , $rhs:expr) => {
        $array[$index] = $rhs;
    };
    ($array:expr, $index:expr, -= , $rhs:expr) => {
        $array[$index] -= $rhs;
    };
    ($array:expr, $index:expr, += , $rhs:expr) => {
        $array[$index] += $rhs;
    };
    ($array:expr, $index:expr, &= , $rhs:expr) => {
        $array[$index] &= $rhs;
    };
    ($array:expr, $index:expr, == , $rhs:expr) => {
        $array[$index] == $rhs
    };
}

macro_rules! div {
    ($a:expr, $b:expr) => {
        $a / $b
    };
}

macro_rules! select_implementation {
    ($($tt:tt)*) => {};
}

mod atan;
mod atan2;
mod cos;
mod exp;
mod expm1;
mod hypot;
mod k_cos;
mod k_sin;
mod k_tan;
mod log;
mod rem_pio2;
mod rem_pio2_large;
mod scalbn;
mod sin;
mod tan;
mod tanh;

pub(crate) use atan::atan;
pub(crate) use atan2::atan2;
pub(crate) use cos::cos;
pub(crate) use exp::exp;
pub(crate) use hypot::hypot;
pub(crate) use log::log;
pub(crate) use sin::sin;
pub(crate) use tan::tan;
pub(crate) use tanh::tanh;

use expm1::expm1;
use k_cos::k_cos;
use k_sin::k_sin;
use k_tan::k_tan;
use rem_pio2::rem_pio2;
use rem_pio2_large::rem_pio2_large;
use scalbn::scalbn;

#[inline]
fn fabs(x: f64) -> f64 {
    x.abs()
}

#[inline]
fn floor(x: f64) -> f64 {
    x.floor()
}

#[inline]
fn sqrt(x: f64) -> f64 {
    x.sqrt()
}
