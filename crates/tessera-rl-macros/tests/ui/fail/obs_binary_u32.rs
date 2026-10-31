// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::ObsLeaf;
use tessera_rl_macros::Observation;

#[derive(Observation)]
struct O {
    #[space(multi_binary(shape = [4]))]
    x: ObsLeaf<u32>,
}

fn main() {}
