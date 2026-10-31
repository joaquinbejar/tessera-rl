// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::Pool;
use tessera_rl_macros::EnvState;

#[derive(EnvState, Clone, Copy)]
#[repr(C)]
struct Padded {
    items: Pool<u32, 3>,
}

fn main() {}
