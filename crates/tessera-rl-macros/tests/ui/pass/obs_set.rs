// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::space::Space;
use tessera_rl_core::{Bool8, LeafId, ObsLeaf, ObsSet, Observation, SetId};
use tessera_rl_macros::Observation;

struct Cfg {
    max: u32,
}

#[derive(Observation)]
struct O {
    #[space(box(low = 0, high = 9, shape = [2]))]
    agent: ObsLeaf<i32>,
    #[space(set(max = |c: &Cfg| c.max, inner(box(low = 0, high = 9, shape = [2]))))]
    items: ObsSet<i32>,
    #[space(discrete(n = 3))]
    after: ObsLeaf<i32>,
}

fn main() {
    const _: LeafId<i32> = O::AGENT;
    const ITEMS: SetId<i32> = O::ITEMS;
    const _: LeafId<i32> = O::AFTER;
    const _: () = assert!(<O as Observation<Cfg>>::LEAVES == 4);
    let _: LeafId<Bool8> = ITEMS.mask();
    assert_eq!((ITEMS.items().index(), ITEMS.mask().index(), O::AFTER.index()), (1, 2, 3));
    let Ok(Space::Dict(entries)) = O::space(&Cfg { max: 5 }) else {
        panic!("a dict");
    };
    assert!(matches!(&entries[1].1, Space::Set { max: 5, .. }));
}
