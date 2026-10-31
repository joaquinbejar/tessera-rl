// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_macros::EnvState;

#[derive(EnvState, Clone, Copy, Default)]
#[repr(C, packed)]
struct S {
    a: u8,
    b: u32,
}

fn main() {}
