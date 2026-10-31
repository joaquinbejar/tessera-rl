// SPDX-License-Identifier: MIT OR Apache-2.0

// The example of SPEC section 10: 8 + 4 + 2 + 1 + 1 = 16 bytes, alignment 4.
use tessera_rl_core::Bool8;
use tessera_rl_macros::EnvState;
use tessera_rl_num::Fx32;

#[derive(EnvState, Clone, Copy, Default)]
#[repr(C)]
pub struct GridState {
    pos: [Fx32; 2],
    goal: [i16; 2],
    steps: u16,
    reached_goal: Bool8,
    out_of_bounds: Bool8,
}

fn main() {
    assert_eq!(<GridState as tessera_rl_core::Canonical>::ENCODED_LEN, 16);
}
