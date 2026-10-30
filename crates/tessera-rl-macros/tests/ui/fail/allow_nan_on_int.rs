// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_macros::EnvState;

#[derive(EnvState, Clone, Copy, Default)]
#[repr(C)]
struct S {
    #[state(allow_nan)]
    count: i32,
}

fn main() {}
