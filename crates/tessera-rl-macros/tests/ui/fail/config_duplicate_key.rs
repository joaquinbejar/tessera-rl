// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_macros::EnvConfig;

#[derive(EnvConfig)]
struct C {
    #[config(default = 1, default = 2)]
    size: u16,
}

fn main() {}
