// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_macros::EnvState;

#[derive(EnvState, Clone, Copy)]
#[repr(C)]
struct S<'a> {
    r: &'a u32,
}

fn main() {}
