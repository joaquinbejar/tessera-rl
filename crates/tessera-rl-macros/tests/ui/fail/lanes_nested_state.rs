// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_macros::EnvState;

#[derive(EnvState, Clone, Copy, Default)]
#[repr(C)]
struct Inner {
    x: f64,
}

#[derive(EnvState, Clone, Copy, Default)]
#[repr(C)]
#[state(lanes)]
struct S {
    a: f64,
    b: Inner,
}

fn main() {}
