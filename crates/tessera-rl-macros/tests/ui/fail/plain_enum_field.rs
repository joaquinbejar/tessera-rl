// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_macros::EnvState;

#[derive(Clone, Copy, Default)]
#[repr(u8)]
enum Phase {
    #[default]
    A,
    B,
}

#[derive(EnvState, Clone, Copy, Default)]
#[repr(C)]
struct S {
    phase: Phase,
}

fn main() {}
