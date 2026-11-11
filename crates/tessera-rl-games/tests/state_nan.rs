// SPDX-License-Identifier: MIT OR Apache-2.0

//! ADR-0055: the `debug-checks` NaN trap scans every shipped state type.
//! The board-game states have no float field, which the decoder confirms:
//! no NaN window anywhere in their encoding is rejected as NaN.

#![expect(
    clippy::indexing_slicing,
    reason = "test helpers slice: a failure is the assertion"
)]

use tessera_rl_core::encoding::{Canonical, decode_from_slice, encode_to_vec};
use tessera_rl_core::{DecodeError, StateInit};
use tessera_rl_games::{Connect4State, Go9State, LeducState, OthelloState};

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

#[test]
fn test_state_nan_game_states_have_no_float_field() {
    assert_eq!(nan_windows::<Connect4State>(), 0);
    assert_eq!(nan_windows::<Go9State>(), 0);
    assert_eq!(nan_windows::<LeducState>(), 0);
    assert_eq!(nan_windows::<OthelloState>(), 0);
}
