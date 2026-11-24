// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_macros::EnvState;

#[derive(EnvState, Clone, Copy, Default)]
#[repr(C)]
struct S {
    a: u32,
    b: u32,
    c: u8,
    d: u64,
}

fn main() {}
