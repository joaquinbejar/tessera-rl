// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_macros::StateEnum;

#[derive(StateEnum, Clone, Copy, Default)]
#[repr(u8)]
#[repr(u16)]
enum E {
    #[default]
    A,
}

fn main() {}
