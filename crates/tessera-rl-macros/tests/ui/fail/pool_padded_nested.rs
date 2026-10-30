// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::Pool;
use tessera_rl_macros::EnvState;

#[derive(EnvState, Clone, Copy)]
#[repr(C)]
struct Inner {
    items: Pool<u16, 3>,
    pad: u8,
}

#[derive(EnvState, Clone, Copy)]
#[repr(C)]
struct Outer {
    inner: [Inner; 2],
}

fn main() {}
