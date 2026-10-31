// SPDX-License-Identifier: MIT OR Apache-2.0

//! `StateInit`: the initial value of state types (SPEC 5.1, ADR-0029).
//!
//! The engine initializes every record from it before the first reset, and
//! `Pool` fills inactive entries with it. It exists alongside `Default`
//! because `std` implements `Default` for arrays only up to 32 elements;
//! `StateInit` covers arrays of any length, recursively.

use tessera_rl_num::{Fx32, Fx64};

use crate::bool8::Bool8;
use crate::reserved::Reserved;

/// A valid initial value. `#[derive(EnvState)]` implements it field by
/// field, and `#[derive(StateEnum)]` returns the `#[default]` variant.
pub trait StateInit: Sized {
    /// Returns the initial value.
    fn state_default() -> Self;
}

macro_rules! state_init_zero {
    ($($ty:ty),*) => {$(
        impl StateInit for $ty {
            #[inline]
            fn state_default() -> Self {
                Self::default()
            }
        }
    )*};
}

state_init_zero!(
    u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Bool8, Fx32, Fx64
);

impl<const N: usize> StateInit for Reserved<N> {
    #[inline]
    fn state_default() -> Self {
        Self::ZERO
    }
}

impl<T: StateInit, const N: usize> StateInit for [T; N] {
    #[inline]
    fn state_default() -> Self {
        core::array::from_fn(|_| T::state_default())
    }
}

#[cfg(test)]
mod tests {
    use super::StateInit;

    #[test]
    fn test_state_init_nested_arrays_beyond_std_default() {
        let grid = <[[u8; 64]; 2]>::state_default();
        assert!(grid.iter().flatten().all(|&b| b == 0));
        let big = <[[f32; 40]; 40]>::state_default();
        assert!(big.iter().flatten().all(|&x| x.to_bits() == 0));
    }
}
