// SPDX-License-Identifier: MIT OR Apache-2.0

//! The map generator: symmetry, connectivity, determinism and the fallback.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures"
)]

use core::num::NonZeroU32;

use tessera_rl_rts::mapgen::{
    MAP_RETRIES, Map, ORE_DRAWS, accepts, core_cells, fallback_terrain, generate, wall_draws,
};
use tessera_rl_rts::world::{Grid, Terrain};

/// A seeded draw for tests (xorshift), below the bound.
fn seeded(seed: u64) -> impl FnMut(NonZeroU32) -> u32 {
    let mut s = seed | 1;
    move |bound| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        u32::try_from(s % u64::from(bound.get())).unwrap()
    }
}

fn symmetric(g: Grid, m: &Map) -> bool {
    (0..g.cells()).all(|c| m.terrain[usize::from(c)] == m.terrain[usize::from(g.mirror(c))])
}

#[test]
fn test_mapgen_seed_sweep_is_symmetric_and_connected() {
    let mut fallbacks = 0;
    for size in [8, 9, 16, 23, 32] {
        let g = Grid::new(size).unwrap();
        for seed in 0..200 {
            let m = generate(g, seeded(seed));
            assert!(symmetric(g, &m), "size {size} seed {seed}");
            assert!(accepts(g, &m.terrain, m.cores), "size {size} seed {seed}");
            assert_eq!(m.cores[1], g.mirror(m.cores[0]));
            for c in m.cores {
                assert_eq!(m.terrain[usize::from(c)], Terrain::Open as u8);
            }
            fallbacks += u32::from(m.fallback);
        }
    }
    // The fallback is a rare path, not the usual one.
    assert!(fallbacks < 50, "{fallbacks}");
}

#[test]
fn test_mapgen_same_draws_give_the_same_map() {
    let g = Grid::new(16).unwrap();
    assert_eq!(generate(g, seeded(7)), generate(g, seeded(7)));
    assert_ne!(
        generate(g, seeded(7)).terrain,
        generate(g, seeded(8)).terrain
    );
}

#[test]
fn test_mapgen_every_attempt_draws_the_same_count() {
    let g = Grid::new(16).unwrap();
    let mut n = 0u32;
    let m = generate(g, |_| {
        n += 1;
        0
    });
    assert_eq!(n, m.attempts * (wall_draws(g) + ORE_DRAWS));
}

#[test]
fn test_mapgen_wall_across_the_middle_reaches_the_fallback_after_the_bound() {
    // Size 16: the first sixteen wall draws fill row 8, whose mirror is row
    // 7, outside both clear zones, so the halves are cut in every attempt.
    let g = Grid::new(16).unwrap();
    let per = wall_draws(g) + ORE_DRAWS;
    let mut k = 0u32;
    let m = generate(g, |_| {
        let pos = k % per;
        k += 1;
        if pos < 16 { 128 + pos } else { 0 }
    });
    assert!(m.fallback);
    assert_eq!(m.attempts, MAP_RETRIES);
    assert_eq!(k, MAP_RETRIES * per);
    assert!(symmetric(g, &m));
    assert!(accepts(g, &m.terrain, m.cores));
}

#[test]
fn test_mapgen_fallback_map_is_symmetric_connected_and_has_ore_at_every_size() {
    for size in 8..=32 {
        let g = Grid::new(size).unwrap();
        let cores = core_cells(g);
        let terrain = fallback_terrain(g, cores);
        let m = Map {
            terrain,
            cores,
            attempts: MAP_RETRIES,
            fallback: true,
        };
        assert!(symmetric(g, &m));
        assert!(accepts(g, &terrain, cores));
        let ore = (0..g.cells())
            .filter(|&c| terrain[usize::from(c)] == Terrain::Ore as u8)
            .count();
        assert_eq!(ore, 4, "size {size}");
    }
}
