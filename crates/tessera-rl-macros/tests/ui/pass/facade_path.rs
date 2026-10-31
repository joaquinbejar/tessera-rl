// SPDX-License-Identifier: MIT OR Apache-2.0

// Generated code reached through the facade with an explicit path.
use tessera_rl::core::Bool8;
use tessera_rl_macros::{EnvState, StateEnum};

#[derive(StateEnum, Clone, Copy, Default)]
#[tessera(crate = ::tessera_rl::core)]
#[repr(i16)]
enum Side {
    #[default]
    Left = -1,
    Right = 1,
}

#[derive(EnvState, Clone, Copy, Default)]
#[tessera(crate = ::tessera_rl::core)]
#[repr(C)]
struct Flag {
    side: Side,
    up: Bool8,
    down: Bool8,
}

fn main() {
    assert_eq!(<Flag as tessera_rl::core::Canonical>::ENCODED_LEN, 4);
}
