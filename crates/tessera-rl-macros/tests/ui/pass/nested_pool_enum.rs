// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::{Bool8, Pool, Reserved};
use tessera_rl_macros::{EnvState, StateEnum};

// Valid discriminants are 1 and 2 only; the default is valid.
#[derive(StateEnum, Clone, Copy, Default, PartialEq, Debug)]
#[repr(u8)]
enum Kind {
    #[default]
    Scout = 1,
    Tank = 2,
}

#[derive(EnvState, Clone, Copy, Default, PartialEq)]
#[repr(C)]
struct Unit {
    hp: i16,
    kind: Kind,
    alive: Bool8,
}

#[derive(EnvState, Clone, Copy, Default)]
#[repr(C)]
struct World {
    tick: u32,
    units: Pool<Unit, 4>,
    #[state(allow_nan)]
    noise: f32,
}

#[derive(EnvState, Clone, Copy, Default)]
#[repr(C)]
struct Gap {
    a: u8,
    pad: Reserved<3>,
    b: u32,
}

fn main() {}
