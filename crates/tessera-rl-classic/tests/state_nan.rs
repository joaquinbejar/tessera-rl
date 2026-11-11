// SPDX-License-Identifier: MIT OR Apache-2.0

//! ADR-0055: the `debug-checks` NaN trap scans every shipped state type.
//! A float-holding state finds a NaN written into each float element with
//! `first_nan`; every other state has no float field, which the decoder
//! confirms (no NaN window anywhere in its encoding is rejected as NaN).

#![expect(
    clippy::indexing_slicing,
    reason = "test helpers slice: a failure is the assertion"
)]

use tessera_rl_classic::{
    AcrobotState, BrickWallState, CartPoleState, GridState, KeyDoorState, MountainCarState,
    PaddleDuelState, PendulumState, SnakeState, TagState,
};
use tessera_rl_core::encoding::{Canonical, decode_from_slice, encode_to_vec};
use tessera_rl_core::{DecodeError, StateInit};

/// The number of positions in `T`'s default encoding where writing an
/// `f32` or `f64` NaN makes decoding fail with `NanInState`: zero exactly
/// when `T` has no float field outside `#[state(allow_nan)]`.
fn nan_windows<T: Canonical + StateInit>() -> usize {
    let base = encode_to_vec(&T::state_default());
    assert!(decode_from_slice::<T>(&base).is_ok(), "default must decode");
    let f32_nan = 0x7fc0_0000_u32.to_le_bytes();
    let f64_nan = 0x7ff8_0000_0000_0000_u64.to_le_bytes();
    let mut hits = 0;
    for pattern in [&f32_nan[..], &f64_nan[..]] {
        for k in 0..=base.len().saturating_sub(pattern.len()) {
            let mut bytes = base.clone();
            bytes[k..k + pattern.len()].copy_from_slice(pattern);
            if matches!(
                decode_from_slice::<T>(&bytes),
                Err(DecodeError::NanInState { .. })
            ) {
                hits += 1;
            }
        }
    }
    hits
}

/// Writes NaN into each of `n` float elements with `set` and checks that
/// `first_nan` names `field`; the clean default reports nothing.
fn each_element_found<T: Canonical + StateInit>(
    n: usize,
    field: &str,
    set: impl Fn(&mut T, usize),
) {
    assert!(nan_windows::<T>() > 0, "the decoder sees the float field");
    assert_eq!(T::state_default().first_nan(), None);
    for i in 0..n {
        let mut s = T::state_default();
        set(&mut s, i);
        assert_eq!(s.first_nan(), Some(field), "element {i}");
    }
}

#[test]
fn test_state_nan_float_states_find_a_nan_in_every_element() {
    each_element_found::<CartPoleState>(4, "s", |s, i| s.s[i] = f64::NAN);
    each_element_found::<AcrobotState>(4, "s", |s, i| s.s[i] = f64::NAN);
    each_element_found::<PendulumState>(2, "s", |s, i| s.s[i] = f64::NAN);
    each_element_found::<MountainCarState>(2, "s", |s, i| s.s[i] = f64::NAN);
}

#[test]
fn test_state_nan_other_classic_states_have_no_float_field() {
    assert_eq!(nan_windows::<BrickWallState>(), 0);
    assert_eq!(nan_windows::<GridState>(), 0);
    assert_eq!(nan_windows::<KeyDoorState>(), 0);
    assert_eq!(nan_windows::<PaddleDuelState>(), 0);
    assert_eq!(nan_windows::<SnakeState>(), 0);
    assert_eq!(nan_windows::<TagState>(), 0);
}
