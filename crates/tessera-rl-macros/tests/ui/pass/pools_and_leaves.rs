// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::{Bool8, ObsLeaf, Pool};
use tessera_rl_macros::{EnvState, Observation};

#[derive(EnvState, Clone, Copy)]
#[repr(C)]
struct S {
    items: Pool<u32, 4>,
    flags: Pool<Bool8, 3>,
    pad: [u8; 2],
}

#[derive(Observation)]
struct O {
    #[space(discrete(n = 3))]
    heading: ObsLeaf<i32>,
    #[space(multi_binary(shape = [4]))]
    bits: ObsLeaf<Bool8>,
}

fn main() {}
