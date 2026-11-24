// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::ObsLeaf;
use tessera_rl_macros::Observation;

#[derive(Observation)]
struct O {
    #[space(box(low = 0, high = 9, shape = [2]))]
    pos: ObsLeaf<i32>,
    #[space(box(low = 0, high = 9, shape = [2]))]
    pos_item: ObsLeaf<i32>,
}

fn main() {}
