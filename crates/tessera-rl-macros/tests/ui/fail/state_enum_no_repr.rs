// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_macros::StateEnum;

#[derive(StateEnum, Clone, Copy, Default)]
enum E {
    #[default]
    A,
}

fn main() {}
