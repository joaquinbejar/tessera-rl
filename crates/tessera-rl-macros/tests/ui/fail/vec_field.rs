// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_macros::EnvState;

#[derive(EnvState, Clone, Default)]
#[repr(C)]
struct S {
    items: Vec<u32>,
}

fn main() {}
