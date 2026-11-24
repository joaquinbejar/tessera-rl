// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::ObsSet;
use tessera_rl_macros::Observation;

#[derive(Observation)]
struct O {
    #[space(box(low = 0, high = 9, shape = [2]))]
    units: ObsSet<i32>,
}

fn main() {}
