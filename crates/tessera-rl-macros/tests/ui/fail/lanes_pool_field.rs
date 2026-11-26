// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::Pool;
use tessera_rl_macros::EnvState;

#[derive(EnvState, Clone, Copy, Default)]
#[repr(C)]
#[state(lanes)]
struct S {
    a: f64,
    items: Pool<i32, 4>,
}

fn main() {}
