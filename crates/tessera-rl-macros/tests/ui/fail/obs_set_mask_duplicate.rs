// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::{ObsLeaf, ObsSet};
use tessera_rl_macros::Observation;

#[derive(Observation)]
struct O {
    #[space(set(max = 4, inner(box(low = 0, high = 9, shape = [2]))))]
    units: ObsSet<i32>,
    #[space(box(low = 0, high = 1, shape = [4]))]
    units_mask: ObsLeaf<u8>,
}

fn main() {}
