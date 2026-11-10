// SPDX-License-Identifier: MIT OR Apache-2.0

//! Flow fields against a reference BFS, around walls, and under the mirror.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures"
)]

use core::num::NonZeroU32;
use std::collections::VecDeque;

use tessera_rl_rts::flow::{Field, UNREACHABLE};
use tessera_rl_rts::mapgen::generate;
use tessera_rl_rts::world::{Dir, Grid, MAX_CELLS, Team, Terrain};

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

fn open(terrain: &[u8; MAX_CELLS]) -> impl Fn(u16) -> bool + '_ {
    |c| terrain[usize::from(c)] == Terrain::Open as u8
}

/// The obvious BFS, with a heap queue and its own neighbour loop.
fn reference(g: Grid, goals: &[u16], passable: &dyn Fn(u16) -> bool) -> Vec<u16> {
    let cells = usize::from(g.cells());
    let mut dist = vec![UNREACHABLE; cells];
    let mut queue = VecDeque::new();
    for &s in goals {
        if dist[usize::from(s)] == UNREACHABLE {
            dist[usize::from(s)] = 0;
            queue.push_back(s);
        }
    }
    while let Some(c) = queue.pop_front() {
        let (col, row) = g.coords(c);
        let side = g.size();
        let mut ns = Vec::new();
        if row > 0 {
            ns.push(c - side);
        }
        if row + 1 < side {
            ns.push(c + side);
        }
        if col > 0 {
            ns.push(c - 1);
        }
        if col + 1 < side {
            ns.push(c + 1);
        }
        for m in ns {
            if dist[usize::from(m)] == UNREACHABLE && passable(m) {
                dist[usize::from(m)] = dist[usize::from(c)] + 1;
                queue.push_back(m);
            }
        }
    }
    dist
}

/// Goal sets of each class shape: one cell (a Core), a few cells (ore
/// nodes, impassable themselves), and a spread of open cells (units).
fn goal_sets(g: Grid, terrain: &[u8; MAX_CELLS], cores: [u16; 2], seed: u64) -> Vec<Vec<u16>> {
    let ore: Vec<u16> = (0..g.cells())
        .filter(|&c| terrain[usize::from(c)] == Terrain::Ore as u8)
        .collect();
    let mut draw = seeded(seed);
    let units: Vec<u16> = (0..6)
        .map(|_| u16::try_from(draw(NonZeroU32::new(u32::from(g.cells())).unwrap())).unwrap())
        .collect();
    vec![vec![cores[0]], vec![cores[1]], ore, units, cores.to_vec()]
}

#[test]
fn test_flow_field_matches_reference_bfs_on_generated_maps() {
    let mut field = Field::default();
    let mut compared = 0u32;
    for size in [8u16, 13, 16, 24, 32] {
        let g = Grid::new(size).unwrap();
        for seed in 1..=40u64 {
            let map = generate(g, seeded(seed * 7919 + u64::from(size)));
            for goals in goal_sets(g, &map.terrain, map.cores, seed) {
                field.build(g, goals.iter().copied(), open(&map.terrain));
                let want = reference(g, &goals, &open(&map.terrain));
                for c in 0..g.cells() {
                    assert_eq!(
                        field.distance(c),
                        want[usize::from(c)],
                        "size {size} seed {seed} cell {c}"
                    );
                }
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 5 * 40 * 5);
}

#[test]
fn test_flow_field_rebuild_on_a_smaller_grid_leaves_no_stale_distance() {
    let mut field = Field::default();
    let big = Grid::new(32).unwrap();
    let small = Grid::new(8).unwrap();
    let all_open = [Terrain::Open as u8; MAX_CELLS];
    field.build(big, [0u16], open(&all_open));
    field.build(small, [63u16], open(&all_open));
    let want = reference(small, &[63], &open(&all_open));
    for c in 0..small.cells() {
        assert_eq!(field.distance(c), want[usize::from(c)]);
    }
}

#[test]
fn test_flow_field_walled_in_goal_is_unreachable_and_never_approached() {
    let g = Grid::new(8).unwrap();
    let mut terrain = [Terrain::Open as u8; MAX_CELLS];
    // A ring of walls around the goal at (5, 5).
    let goal = g.index(5, 5).unwrap();
    for (x, y) in [
        (4, 4),
        (5, 4),
        (6, 4),
        (4, 5),
        (6, 5),
        (4, 6),
        (5, 6),
        (6, 6),
    ] {
        terrain[usize::from(g.index(x, y).unwrap())] = Terrain::Wall as u8;
    }
    let mut field = Field::default();
    field.build(g, [goal], open(&terrain));
    assert_eq!(field.distance(goal), 0);
    for c in 0..g.cells() {
        if c != goal {
            assert_eq!(field.distance(c), UNREACHABLE, "cell {c}");
            assert_eq!(field.descend(g, c, Team::Zero), None);
            assert_eq!(field.descend(g, c, Team::One), None);
        }
    }
}

#[test]
fn test_flow_field_descend_steps_strictly_closer_and_stops_next_to_the_goal() {
    let g = Grid::new(16).unwrap();
    for seed in 1..=20u64 {
        let map = generate(g, seeded(seed));
        let mut field = Field::default();
        field.build(g, [map.cores[1]], open(&map.terrain));
        for team in [Team::Zero, Team::One] {
            for c in 0..g.cells() {
                let d = field.distance(c);
                if d == 1 {
                    // Next to the goal: the goal is a source, never entered.
                    assert_eq!(field.descend(g, c, team), None, "cell {c}");
                }
                match field.descend(g, c, team) {
                    Some(dir) => {
                        let n = g.step(c, dir).unwrap();
                        assert_eq!(field.distance(n), d - 1);
                        assert!(open(&map.terrain)(n));
                    }
                    None => assert!(d == UNREACHABLE || d <= 1, "cell {c} at {d} did not move"),
                }
            }
        }
    }
}

#[test]
fn test_flow_field_ties_follow_the_team_order_and_mirror() {
    // Open grid, goal two cells diagonal from the unit: two first steps tie.
    let g = Grid::new(8).unwrap();
    let all_open = [Terrain::Open as u8; MAX_CELLS];
    let unit = g.index(3, 3).unwrap();
    let goal = g.index(5, 5).unwrap();
    let mut field = Field::default();
    field.build(g, [goal], open(&all_open));
    // Team 0 orders N, E, S, W: East before South.
    assert_eq!(field.descend(g, unit, Team::Zero), Some(Dir::East));
    // Team 1 at the mirrored cell toward the mirrored goal takes the mirrored step.
    let mut mirrored = Field::default();
    mirrored.build(g, [g.mirror(goal)], open(&all_open));
    assert_eq!(
        mirrored.descend(g, g.mirror(unit), Team::One),
        Some(Dir::West)
    );
}

#[test]
fn test_flow_field_choices_commute_with_the_mirror_on_generated_maps() {
    let g = Grid::new(16).unwrap();
    for seed in 1..=20u64 {
        let map = generate(g, seeded(seed * 31));
        // Both Cores as goals: a goal set the mirror maps onto itself.
        let mut field = Field::default();
        field.build(g, map.cores, open(&map.terrain));
        for c in 0..g.cells() {
            assert_eq!(field.distance(c), field.distance(g.mirror(c)));
            let zero = field.descend(g, c, Team::Zero);
            let one = field.descend(g, g.mirror(c), Team::One);
            assert_eq!(zero.map(Dir::mirrored), one, "seed {seed} cell {c}");
        }
    }
}

#[test]
fn test_flow_field_goals_outside_the_grid_are_ignored() {
    let g = Grid::new(8).unwrap();
    let map = generate(g, seeded(11));
    let mut with = Field::default();
    let mut without = Field::default();
    // 64 is one past the last cell of size 8; 1023 the last of the largest grid.
    with.build(g, [map.cores[1], 64, 1023], open(&map.terrain));
    without.build(g, [map.cores[1]], open(&map.terrain));
    for c in 0..g.cells() {
        assert_eq!(with.distance(c), without.distance(c), "cell {c}");
    }
}

#[test]
fn test_flow_field_reads_past_the_last_grid_are_unreachable() {
    let all_open = [Terrain::Open as u8; MAX_CELLS];
    let mut field = Field::default();
    field.build(Grid::new(32).unwrap(), [0u16], open(&all_open));
    field.build(Grid::new(8).unwrap(), [0u16], open(&all_open));
    assert_eq!(field.distance(63), 14);
    assert_eq!(field.distance(64), UNREACHABLE);
    assert_eq!(field.distance(1023), UNREACHABLE);
}

#[test]
fn test_flow_field_a_mover_on_a_cell_left_out_of_passable_stays() {
    // The documented contract: `passable` must include the movers' cells.
    let g = Grid::new(8).unwrap();
    let all_open = [Terrain::Open as u8; MAX_CELLS];
    let unit = g.index(3, 3).unwrap();
    let goal = g.index(6, 3).unwrap();
    let mut field = Field::default();
    field.build(g, [goal], |c| c != unit && open(&all_open)(c));
    assert_eq!(field.distance(unit), UNREACHABLE);
    assert_eq!(field.descend(g, unit, Team::Zero), None);
    field.build(g, [goal], open(&all_open));
    assert_eq!(field.descend(g, unit, Team::Zero), Some(Dir::East));
}
