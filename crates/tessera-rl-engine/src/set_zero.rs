// SPDX-License-Identifier: MIT OR Apache-2.0

//! Zeroing of absent `Set` slots after `observe` (ADR-0082 point 3).

use crate::layout::SetSpans;

/// Zeroes, in every item leaf of each set, the row of every slot whose mask
/// byte is zero: one pass over the `MAX` mask bytes per item leaf that
/// writes only absent rows, with no allocation. `leaves` is one agent's leaf rows, in leaf order.
///
/// With `debug-checks` (and debug assertions), a nonzero row in an absent
/// slot, which only the environment can have written, fails an assertion
/// naming the set's mask leaf and the slot.
#[inline]
pub(crate) fn zero_absent_rows(leaves: &mut [&mut [u8]], sets: &SetSpans) {
    let rows = sets.rows();
    for set in sets.as_slice() {
        let Some((head, tail)) = leaves.split_at_mut_checked(set.mask) else {
            continue;
        };
        let Some(mask) = tail.first() else {
            continue;
        };
        // Common case: every slot present, nothing to clear.
        if !mask.contains(&0) {
            continue;
        }
        let items = head.get_mut(set.first..).unwrap_or(&mut []);
        let strides = rows.get(set.first..set.mask).unwrap_or(&[]);
        for (leaf, &row) in items.iter_mut().zip(strides) {
            #[cfg(feature = "debug-checks")]
            report_absent_writes(leaf, row, mask, set);
            // Only absent rows are written; present rows are left as
            // `observe` wrote them.
            if row == 0 {
                continue;
            }
            let mut start = 0usize;
            for &present in mask.iter() {
                let Some(end) = start.checked_add(row) else {
                    break;
                };
                if present == 0
                    && let Some(bytes) = leaf.get_mut(start..end)
                {
                    bytes.fill(0);
                }
                start = end;
            }
        }
    }
}

/// `debug-checks`: an assertion naming the set and the first absent slot
/// `observe` wrote into.
#[cfg(feature = "debug-checks")]
fn report_absent_writes(leaf: &[u8], row: usize, mask: &[u8], set: &crate::layout::SetSpan) {
    for (slot, (bytes, &present)) in leaf.chunks_exact(row).zip(mask).enumerate() {
        debug_assert!(
            present != 0 || bytes.iter().all(|&b| b == 0),
            "observe wrote into absent slot {slot} of the set whose mask is leaf {}",
            set.mask
        );
    }
}

#[cfg(test)]
mod tests {
    use super::zero_absent_rows;
    use crate::layout::{SetSpan, SetSpans};

    #[cfg(not(all(feature = "debug-checks", debug_assertions)))]
    #[test]
    fn test_zero_absent_rows_clears_only_absent_slots_of_set_leaves() {
        // Leaf 0 is outside the set; leaves 1 and 2 are items of three
        // slots (2 and 1 bytes per row); leaf 3 is the mask.
        let mut outside = [9u8; 2];
        let mut a = [1u8, 2, 3, 4, 5, 6];
        let mut b = [7u8, 8, 9];
        let mut mask = [1u8, 0, 1];
        let mut leaves = [&mut outside[..], &mut a[..], &mut b[..], &mut mask[..]];
        let set = SetSpan {
            first: 1,
            mask: 3,
            max: 3,
        };
        let sets = SetSpans::from_parts(&[set], &[(1, 2), (2, 1)]);
        zero_absent_rows(&mut leaves, &sets);
        assert_eq!(outside, [9, 9]);
        assert_eq!(a, [1, 2, 0, 0, 5, 6]);
        assert_eq!(b, [7, 0, 9]);
        assert_eq!(mask, [1, 0, 1]);
    }

    #[cfg(all(feature = "debug-checks", debug_assertions))]
    #[test]
    #[should_panic(expected = "observe wrote into absent slot 1 of the set whose mask is leaf 1")]
    fn test_zero_absent_rows_write_into_absent_slot_is_reported_under_debug_checks() {
        let mut a = [1u8, 2];
        let mut mask = [1u8, 0];
        let mut leaves = [&mut a[..], &mut mask[..]];
        let set = SetSpan {
            first: 0,
            mask: 1,
            max: 2,
        };
        let sets = SetSpans::from_parts(&[set], &[(0, 1)]);
        zero_absent_rows(&mut leaves, &sets);
    }

    #[test]
    fn test_zero_absent_rows_without_sets_changes_nothing() {
        let mut a = [1u8, 2];
        let mut leaves = [&mut a[..]];
        zero_absent_rows(&mut leaves, &SetSpans::default());
        assert_eq!(a, [1, 2]);
    }
}
