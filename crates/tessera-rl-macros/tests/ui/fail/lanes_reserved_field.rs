// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::Reserved;
use tessera_rl_macros::EnvState;

#[derive(EnvState, Clone, Copy, Default)]
#[repr(C)]
#[state(lanes)]
struct S {
    a: i32,
    pad: Reserved<4>,
    b: f64,
}

fn main() {}
