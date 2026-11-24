// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_macros::EnvState;

#[derive(EnvState, Clone, Copy, Default)]
#[repr(C)]
#[state(lanes)]
struct S {
    a: f64,
    b: [f64; 2],
}

fn main() {}
