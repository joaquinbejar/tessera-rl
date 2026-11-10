// SPDX-License-Identifier: MIT OR Apache-2.0

//! Flow-field cost on generated maps: one build per goal class shape, and
//! a full descent over every cell. Local only, no claim (SPEC 16.1).

#![expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "bench setup")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_rts::flow::Field;
use tessera_rl_rts::mapgen::generate;
use tessera_rl_rts::world::{Grid, Team, Terrain};

fn draw(seed: u64) -> impl FnMut(NonZeroU32) -> u32 {
    let mut s = seed | 1;
    move |bound| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        u32::try_from(s % u64::from(bound.get())).unwrap()
    }
}

fn bench(c: &mut Criterion) {
    for size in [16u16, 32] {
        let g = Grid::new(size).unwrap();
        let map = generate(g, draw(7));
        let open = |cell: u16| map.terrain[usize::from(cell)] == Terrain::Open as u8;
        let ore: Vec<u16> = (0..g.cells())
            .filter(|&cell| map.terrain[usize::from(cell)] == Terrain::Ore as u8)
            .collect();
        let mut field = Box::new(Field::default());
        c.bench_function(&format!("rts_flow_build_core_{size}"), |b| {
            b.iter(|| field.build(g, black_box([map.cores[1]]), open));
        });
        c.bench_function(&format!("rts_flow_build_ore_{size}"), |b| {
            b.iter(|| field.build(g, black_box(ore.iter().copied()), open));
        });
        // The four goal classes of one team's step (section 5).
        c.bench_function(&format!("rts_flow_build_four_classes_{size}"), |b| {
            b.iter(|| {
                field.build(g, black_box([map.cores[1]]), open);
                field.build(g, black_box([map.cores[1], g.mirror(3)]), open);
                field.build(g, black_box(ore.iter().copied()), open);
                field.build(g, black_box([map.cores[0]]), open);
            });
        });
        field.build(g, [map.cores[1]], open);
        c.bench_function(&format!("rts_flow_descend_all_cells_{size}"), |b| {
            b.iter(|| {
                for cell in 0..g.cells() {
                    black_box(field.descend(g, black_box(cell), Team::Zero));
                }
            });
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
