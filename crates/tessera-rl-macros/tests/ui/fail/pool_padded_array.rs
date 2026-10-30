// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::Pool;
use tessera_rl_macros::EnvState;

#[derive(EnvState, Clone, Copy)]
#[repr(C)]
struct S {
    pools: [Pool<u64, 2>; 2],
}

fn main() {}
