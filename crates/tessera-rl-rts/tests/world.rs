// SPDX-License-Identifier: MIT OR Apache-2.0

//! The grid, its mirror, neighbour orders, distance, the unit table and the
//! movement counter.

use tessera_rl_rts::units::{Kind, Unit, advance_counter};
use tessera_rl_rts::world::{Dir, Grid, MAX_SIZE, MIN_SIZE, Team, mirror_window};

#[test]
fn test_grid_bounds_accept_8_and_32_refuse_7_and_33() {
    assert!(Grid::new(MIN_SIZE).is_some());
    assert!(Grid::new(MAX_SIZE).is_some());
    assert!(Grid::new(MIN_SIZE - 1).is_none());
    assert!(Grid::new(MAX_SIZE + 1).is_none());
}

#[test]
fn test_mirror_is_the_180_degree_rotation_and_an_involution() {
    for size in [8, 9, 16, 31, 32] {
        let g = Grid::new(size).unwrap();
        for cell in 0..g.cells() {
            let (x, y) = g.coords(cell);
            let m = g.mirror(cell);
            assert_eq!(g.coords(m), (size - 1 - x, size - 1 - y));
            assert_eq!(g.mirror(m), cell);
        }
    }
}

#[test]
fn test_team_orders_are_mirror_images_cell_by_cell() {
    let g = Grid::new(11).unwrap();
    for cell in 0..g.cells() {
        let zero = g.neighbours(cell, Team::Zero);
        let one = g.neighbours(g.mirror(cell), Team::One);
        assert_eq!(zero.map(|n| n.map(|c| g.mirror(c))), one);
    }
    for (d, m) in Dir::order(Team::Zero).iter().zip(Dir::order(Team::One)) {
        assert_eq!(d.mirrored(), m);
    }
}

#[test]
fn test_chebyshev_range_1_includes_diagonals() {
    let g = Grid::new(8).unwrap();
    let c = g.index(3, 3).unwrap();
    for (x, y) in [(2, 2), (4, 4), (2, 4), (4, 2), (3, 2), (4, 3)] {
        assert_eq!(g.chebyshev(c, g.index(x, y).unwrap()), 1);
    }
    assert_eq!(g.chebyshev(c, g.index(5, 4).unwrap()), 2);
    assert!(g.adjacent(c, g.index(3, 4).unwrap()));
    assert!(!g.adjacent(c, g.index(4, 4).unwrap()));
}

#[test]
fn test_mirror_window_maps_offset_to_its_negation() {
    let w = 9;
    for i in 0..w * w {
        let (dx, dy) = (i32::from(i % w) - 4, i32::from(i / w) - 4);
        let m = mirror_window(i, w);
        assert_eq!((i32::from(m % w) - 4, i32::from(m / w) - 4), (-dx, -dy));
    }
}

#[test]
fn test_counter_speed_4_and_6_step_on_the_right_ticks() {
    for speed in [4u8, 6] {
        let (mut c, mut steps) = (0u8, Vec::new());
        for tick in 1..=24 {
            let (n, s) = advance_counter(c, speed);
            c = n;
            if s {
                steps.push(tick);
            }
        }
        let expect: Vec<u8> = (1..=24 / speed).map(|k| k * speed).collect();
        assert_eq!(steps, expect);
    }
    // A Lancer (speed 6) at 4 ticks per decision: 2 cells per 3 decisions.
    let (mut c, mut cells) = (0u8, 0);
    for _ in 0..3 * 4 {
        let (n, s) = advance_counter(c, 6);
        c = n;
        cells += u8::from(s);
    }
    assert_eq!(cells, 2);
    assert_eq!(advance_counter(0, 0), (0, false));
}

#[test]
fn test_unit_is_16_bytes_and_zero_is_empty() {
    assert_eq!(size_of::<Unit>(), 16);
    assert!(!Unit::default().alive.get());
    for k in Kind::ALL {
        assert_eq!(Kind::from_byte(k as u8), Some(k));
        assert_eq!(k.is_structure(), k.stats().speed == 0);
    }
    assert_eq!(Kind::from_byte(0), None);
    assert_eq!(Kind::from_byte(6), None);
}
