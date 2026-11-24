// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::io::ObsWriter;
use tessera_rl_core::{ObsLeaf, ObsSetOf};
use tessera_rl_macros::Observation;

#[derive(Observation)]
struct Unit {
    #[space(box(low = 0, high = 9, shape = [2]))]
    pos: ObsLeaf<i32>,
}

#[derive(Observation)]
struct Other {
    #[space(box(low = 0, high = 9, shape = [2]))]
    pos: ObsLeaf<i32>,
}

#[derive(Observation)]
struct O {
    #[space(set(max = 4, items = Unit))]
    units: ObsSetOf<Unit>,
}

fn write(out: &mut ObsWriter<'_, '_, O>) {
    if let Some(mut s) = out.set_of_mut(O::UNITS) {
        let _ = s.item(0, Other::POS_ITEM);
    }
}

fn main() {}
