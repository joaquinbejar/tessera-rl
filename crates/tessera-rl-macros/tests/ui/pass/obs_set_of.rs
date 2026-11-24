// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::space::Space;
use tessera_rl_core::{ItemId, LeafId, ObsLeaf, ObsSetOf, Observation, ObservationLayout, SetOf};
use tessera_rl_macros::Observation;

struct Cfg {
    max: u32,
}

/// Declared `hp` before `pos`: item ids and exported keys follow it.
#[derive(Observation)]
struct Unit {
    #[space(box(low = 0.0, high = 1.0, shape = [1]))]
    hp: ObsLeaf<f32>,
    #[space(box(low = 0, high = 9, shape = [2]))]
    pos: ObsLeaf<i32>,
}

#[derive(Observation)]
struct O {
    #[space(discrete(n = 3))]
    before: ObsLeaf<i32>,
    #[space(set(max = |c: &Cfg| c.max, items = Unit))]
    units: ObsSetOf<Unit>,
    #[space(discrete(n = 3))]
    after: ObsLeaf<i32>,
}

fn main() {
    const HP: ItemId<Unit, f32> = Unit::HP_ITEM;
    const POS: ItemId<Unit, i32> = Unit::POS_ITEM;
    const UNITS: SetOf<Unit> = O::UNITS;
    const _: LeafId<i32> = O::AFTER;
    const _: () = assert!(<Unit as ObservationLayout>::LEAVES == 2);
    const _: () = assert!(!<Unit as ObservationLayout>::CONTAINS_SET);
    const _: () = assert!(<O as ObservationLayout>::CONTAINS_SET);
    const _: () = assert!(<O as Observation<Cfg>>::CONTAINS_SET);
    const _: () = assert!(<O as Observation<Cfg>>::LEAVES == 5);
    assert_eq!((HP.index(), POS.index()), (0, 1));
    assert_eq!((UNITS.first(), UNITS.mask(), O::AFTER.index()), (1, 3, 4));
    let Ok(Space::Dict(entries)) = O::space(&Cfg { max: 5 }) else {
        panic!("a dict");
    };
    let Space::Set { inner, max: 5 } = &entries[1].1 else {
        panic!("a set of 5");
    };
    let Space::Dict(items) = &**inner else {
        panic!("a dict inner");
    };
    let keys: Vec<&str> = items.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["hp", "pos"]);
}
