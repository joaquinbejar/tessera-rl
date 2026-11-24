// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_macros::EnvConfig;

#[derive(EnvConfig)]
struct Inner {
    #[config(default = 1, adaptive)]
    speed: u16,
}

#[derive(EnvConfig)]
struct Outer {
    inner: Inner,
}

fn main() {}
