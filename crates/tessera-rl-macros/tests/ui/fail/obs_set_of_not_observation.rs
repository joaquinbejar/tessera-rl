// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::ObsSetOf;
use tessera_rl_macros::Observation;

struct Plain;

#[derive(Observation)]
struct O {
    #[space(set(max = 4, items = Plain))]
    units: ObsSetOf<Plain>,
}

fn main() {}
