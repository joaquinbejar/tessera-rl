// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_macros::EnvConfig;

#[derive(EnvConfig)]
#[config(validate = no_such_check)]
struct C {
    #[config(default = 8)]
    size: u16,
}

fn main() {}
