// SPDX-License-Identifier: MIT OR Apache-2.0

// Arrays longer than 32 elements, nested, have no std Default; StateInit
// covers them.
use tessera_rl_macros::EnvState;

#[derive(EnvState, Clone, Copy)]
#[repr(C)]
struct Maps {
    terrain: [[u8; 64]; 2],
    heat: [[f32; 40]; 40],
}

fn main() {
    let m = <Maps as tessera_rl_core::StateInit>::state_default();
    assert_eq!(m.terrain[1][63], 0);
}
