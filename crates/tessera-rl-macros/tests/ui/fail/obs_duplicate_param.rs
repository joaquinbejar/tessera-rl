// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::ObsLeaf;
use tessera_rl_macros::Observation;

#[derive(Observation)]
struct O {
    #[space(box(low = 0, low = 1, high = 5, shape = [1]))]
    x: ObsLeaf<u8>,
}

fn main() {}
