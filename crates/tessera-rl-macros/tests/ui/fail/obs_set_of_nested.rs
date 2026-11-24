// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::{ObsSet, ObsSetOf};
use tessera_rl_macros::Observation;

#[derive(Observation)]
struct Group {
    #[space(set(max = 2, inner(box(low = 0, high = 9, shape = [2]))))]
    members: ObsSet<i32>,
}

#[derive(Observation)]
struct O {
    #[space(set(max = 4, items = Group))]
    groups: ObsSetOf<Group>,
}

fn main() {}
