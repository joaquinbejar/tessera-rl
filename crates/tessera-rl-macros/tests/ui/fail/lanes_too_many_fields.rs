// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_macros::EnvState;

#[derive(EnvState, Clone, Copy, Default)]
#[repr(C)]
#[state(lanes)]
struct S {
    f0: u32,
    f1: u32,
    f2: u32,
    f3: u32,
    f4: u32,
    f5: u32,
    f6: u32,
    f7: u32,
    f8: u32,
    f9: u32,
    f10: u32,
    f11: u32,
    f12: u32,
}

fn main() {}
