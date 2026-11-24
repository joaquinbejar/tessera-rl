// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed `Set` observation leaves (SPEC 11.1, ADR-0082): item leaves of
//! shape `[MAX, ...]` and a `bool` mask `[MAX]` right after them.

use core::marker::PhantomData;

use crate::bool8::Bool8;
use crate::leaf::{LeafDType, LeafId, LeafMut};

/// Marker field type of an `#[derive(Observation)]` struct: a `Set` of
/// single-leaf items of element type `T`, exported as the item leaf and
/// its mask (ADR-0082). It holds no data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ObsSet<T>(PhantomData<fn() -> T>);

/// Identifies a `Set` of single-leaf items: its item leaf and its mask
/// leaf. Generated as an associated constant by `#[derive(Observation)]`.
#[derive(Debug, PartialEq, Eq)]
pub struct SetId<T> {
    items: LeafId<T>,
    mask: LeafId<Bool8>,
}

impl<T> Clone for SetId<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SetId<T> {}

impl<T> SetId<T> {
    /// The set whose item leaf is at `items` and mask leaf at `mask`.
    #[must_use]
    #[inline]
    pub const fn new(items: usize, mask: usize) -> Self {
        Self {
            items: LeafId::new(items),
            mask: LeafId::new(mask),
        }
    }

    /// The item leaf, `[MAX, ...]`.
    #[must_use]
    #[inline]
    pub const fn items(self) -> LeafId<T> {
        self.items
    }

    /// The mask leaf, `[MAX]`.
    #[must_use]
    #[inline]
    pub const fn mask(self) -> LeafId<Bool8> {
        self.mask
    }
}

/// Typed, bounds-checked writes into one agent's `Set`: the presence mask
/// and one row of the item leaf per slot. Slots left absent are zeroed by
/// the engine after `observe` (ADR-0082 point 3).
#[derive(Debug)]
pub struct SetLeaf<'a, T> {
    items: &'a mut [u8],
    mask: &'a mut [u8],
    _dtype: PhantomData<fn() -> T>,
}

impl<'a, T: LeafDType> SetLeaf<'a, T> {
    #[inline]
    pub(crate) fn new(items: &'a mut [u8], mask: &'a mut [u8]) -> Self {
        Self {
            items,
            mask,
            _dtype: PhantomData,
        }
    }

    /// The capacity `MAX`.
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.mask.len()
    }

    /// Marks slot `i` present or absent; returns `false` if it does not
    /// exist.
    #[inline]
    pub fn set(&mut self, i: usize, present: bool) -> bool {
        self.mask
            .get_mut(i)
            .map(|m| *m = Bool8::new(present).to_byte())
            .is_some()
    }

    /// The item row of slot `i`, or `None` if it does not exist.
    #[must_use]
    #[inline]
    pub fn item(&mut self, i: usize) -> Option<LeafMut<'_, T>> {
        let row = self.items.len().checked_div(self.mask.len())?;
        let start = i.checked_mul(row)?;
        let end = start.checked_add(row)?;
        if i >= self.mask.len() {
            return None;
        }
        self.items.get_mut(start..end).map(LeafMut::new)
    }
}

#[cfg(test)]
mod tests {
    use super::{SetId, SetLeaf};

    #[test]
    fn test_set_leaf_writes_mask_and_rows_with_bounds() {
        let mut items = [0u8; 12];
        let mut mask = [0u8; 3];
        let mut s = SetLeaf::<i16>::new(&mut items, &mut mask);
        assert_eq!(s.capacity(), 3);
        assert!(s.set(1, true));
        assert!(!s.set(3, true));
        let mut row = s.item(1).unwrap();
        assert_eq!(row.len(), 2);
        assert!(row.set(1, 0x0102));
        assert!(s.item(3).is_none());
        assert_eq!(mask, [0, 1, 0]);
        assert_eq!(items, [0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0]);
        let id = SetId::<i16>::new(4, 5);
        assert_eq!((id.items().index(), id.mask().index()), (4, 5));
    }

    #[test]
    fn test_set_leaf_of_no_slots_has_no_rows() {
        let mut items = [0u8; 0];
        let mut mask = [0u8; 0];
        let mut s = SetLeaf::<u8>::new(&mut items, &mut mask);
        assert!(s.item(0).is_none());
        assert!(!s.set(0, true));
    }
}
