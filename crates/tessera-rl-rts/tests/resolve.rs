// SPDX-License-Identifier: MIT OR Apache-2.0

//! Move resolution (rule 1): contests, swaps, cycles, chains and order
//! independence, on hand-built claims.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures"
)]

use tessera_rl_rts::resolve::{Claims, MAX_UNITS, NONE};

/// Units at `cells` (one per unit index), claiming `targets` (`None`: no
/// move). Returns which moved.
fn run(cells: &[u16], targets: &[Option<u16>]) -> Vec<bool> {
    let mut c = Claims::default();
    for (u, &cell) in cells.iter().enumerate() {
        c.holder[usize::from(cell)] = u16::try_from(u).unwrap();
    }
    for (u, t) in targets.iter().enumerate() {
        c.target[u] = t.unwrap_or(NONE);
    }
    c.resolve()[..cells.len()].to_vec()
}

#[test]
fn test_resolve_free_move_and_vacated_cell_are_allowed() {
    assert_eq!(run(&[0], &[Some(1)]), [true]);
    // 0 -> 1 while the unit on 1 moves to 2.
    assert_eq!(run(&[0, 1], &[Some(1), Some(2)]), [true, true]);
}

#[test]
fn test_resolve_swap_is_refused_and_leaves_both() {
    assert_eq!(run(&[0, 1], &[Some(1), Some(0)]), [false, false]);
}

#[test]
fn test_resolve_cycles_of_three_and_four_are_refused_whole() {
    assert_eq!(run(&[0, 1, 2], &[Some(1), Some(2), Some(0)]), [false; 3]);
    assert_eq!(
        run(&[0, 1, 2, 3], &[Some(1), Some(2), Some(3), Some(0)]),
        [false; 4]
    );
}

#[test]
fn test_resolve_three_way_contest_refuses_all() {
    assert_eq!(run(&[0, 2, 9], &[Some(1), Some(1), Some(1)]), [false; 3]);
}

#[test]
fn test_resolve_own_cell_is_not_a_move() {
    assert_eq!(run(&[0], &[Some(0)]), [false]);
}

#[test]
fn test_resolve_chains_behind_a_stayer_are_refused_at_every_length() {
    // Unit 0 stays on cell 10; units 1..=len queue behind it.
    for len in 1..=4u16 {
        let cells: Vec<u16> = (0..=len).map(|k| 10 + k).collect();
        let mut targets = vec![None];
        targets.extend((1..=len).map(|k| Some(10 + k - 1)));
        assert_eq!(run(&cells, &targets), vec![false; usize::from(len) + 1]);
        // The same chain with the stayer as the highest index, so that a
        // single pass in index order would let the tail through.
        let rev_cells: Vec<u16> = cells.iter().rev().copied().collect();
        let rev_targets: Vec<Option<u16>> = targets.iter().rev().copied().collect();
        assert_eq!(
            run(&rev_cells, &rev_targets),
            vec![false; usize::from(len) + 1]
        );
    }
}

#[test]
fn test_resolve_cycle_feeding_a_chain_refuses_the_chain() {
    // 0,1 swap; 2 -> 1's cell; 3 -> 2's cell.
    let r = run(&[0, 1, 2, 3], &[Some(1), Some(0), Some(1), Some(2)]);
    assert_eq!(r, [false; 4]);
    // Cycle of three, and 3 -> 0's cell.
    let r = run(&[0, 1, 2, 3], &[Some(1), Some(2), Some(0), Some(0)]);
    assert_eq!(r, [false; 4]);
}

#[test]
fn test_resolve_contest_feeding_a_chain_refuses_the_chain() {
    // 0 and 1 contest cell 5; 2 queues on 0's cell, 3 on 2's cell.
    let r = run(&[0, 1, 2, 3], &[Some(5), Some(5), Some(0), Some(2)]);
    assert_eq!(r, [false; 4]);
}

#[test]
fn test_resolve_structure_blocks_and_others_move() {
    // 0 is immobile on 4; 1 -> 4 refused; 2 -> 7 free.
    assert_eq!(
        run(&[4, 3, 6], &[None, Some(4), Some(7)]),
        [false, false, true]
    );
}

#[test]
fn test_resolve_result_does_not_depend_on_unit_order() {
    let mut s = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = |m: u64| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s % m
    };
    for _ in 0..2000 {
        let n = usize::try_from(next(12) + 2).unwrap();
        // Distinct cells in 0..20 for n units.
        let mut cells: Vec<u16> = (0..20).collect();
        for i in (1..cells.len()).rev() {
            cells.swap(
                i,
                usize::try_from(next(u64::try_from(i).unwrap() + 1)).unwrap(),
            );
        }
        cells.truncate(n);
        let targets: Vec<Option<u16>> = (0..n)
            .map(|_| (next(4) != 0).then(|| u16::try_from(next(20)).unwrap()))
            .collect();
        let base = run(&cells, &targets);
        // A random permutation of unit indices.
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            perm.swap(
                i,
                usize::try_from(next(u64::try_from(i).unwrap() + 1)).unwrap(),
            );
        }
        let pc: Vec<u16> = perm.iter().map(|&p| cells[p]).collect();
        let pt: Vec<Option<u16>> = perm.iter().map(|&p| targets[p]).collect();
        let got = run(&pc, &pt);
        for (k, &p) in perm.iter().enumerate() {
            assert_eq!(got[k], base[p]);
        }
    }
    assert_eq!(MAX_UNITS, 256);
}

#[test]
fn test_resolve_allowed_moves_never_collide() {
    let mut s = 0x9e37_79b9_u64;
    let mut next = |m: u64| {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (s >> 33) % m
    };
    for _ in 0..2000 {
        let n = 10usize;
        let cells: Vec<u16> = (0..10).map(|k| k * 2).collect();
        let targets: Vec<Option<u16>> = (0..n)
            .map(|_| (next(3) != 0).then(|| u16::try_from(next(20)).unwrap()))
            .collect();
        let r = run(&cells, &targets);
        let mut end: Vec<u16> = (0..n)
            .map(|u| if r[u] { targets[u].unwrap() } else { cells[u] })
            .collect();
        end.sort_unstable();
        end.dedup();
        assert_eq!(end.len(), n);
    }
}
