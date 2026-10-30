// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::ObsLeaf;
use tessera_rl_macros::Observation;

#[derive(Observation)]
struct O {
    #[space(discrete(n = 3))]
    heading: ObsLeaf<u8>,
}

fn main() {}
