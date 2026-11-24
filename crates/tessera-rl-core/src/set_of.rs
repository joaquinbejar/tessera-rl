// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed `Set` observation leaves whose items are an `#[derive(Observation)]`
//! struct (SPEC 11.1, ADR-0082, #429): one item leaf `p/<field>` of shape
//! `[MAX, ...]` per item field, in the item struct's declaration order, then
//! the `bool` mask `[MAX]`.

use core::fmt;
use core::marker::PhantomData;

use crate::bool8::Bool8;
use crate::leaf::{LeafDType, LeafMut};

/// Marker field type of an `#[derive(Observation)]` struct: a `Set` whose
/// items are the observation struct `I`, declared with
/// `#[space(set(max = .., items = I))]`. It holds no data.
pub struct ObsSetOf<I>(PhantomData<fn() -> I>);

impl<I> fmt::Debug for ObsSetOf<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ObsSetOf")
    }
}

impl<I> Clone for ObsSetOf<I> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<I> Copy for ObsSetOf<I> {}

impl<I> Default for ObsSetOf<I> {
    #[inline]
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Identifies leaf `index` of the item struct `I`, with its element type
/// `T`, inside a `Set` of `I`. `#[derive(Observation)]` generates one per
/// leaf field of every struct, named `<FIELD>_ITEM`; the index is the
/// field's position among `I`'s leaves, which is the order the set exports
/// its item leaves in. Bound to `I`, so an id of another struct does not
/// type-check in [`SetOfLeaf::item`].
pub struct ItemId<I, T> {
    index: usize,
    _owner: PhantomData<fn() -> (I, T)>,
}

impl<I, T> fmt::Debug for ItemId<I, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ItemId")
            .field("index", &self.index)
            .finish()
    }
}

impl<I, T> PartialEq for ItemId<I, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<I, T> Eq for ItemId<I, T> {}

impl<I, T> Clone for ItemId<I, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<I, T> Copy for ItemId<I, T> {}

impl<I, T> ItemId<I, T> {
    /// Leaf `index` of the item struct `I`.
    #[must_use]
    #[inline]
    pub const fn new(index: usize) -> Self {
        Self {
            index,
            _owner: PhantomData,
        }
    }

    /// The leaf position within the item struct.
    #[must_use]
    #[inline]
    pub const fn index(self) -> usize {
        self.index
    }
}

/// Identifies a `Set` of item struct `I`: its first item leaf and its mask
/// leaf, right after the last item leaf. Generated as an associated
/// constant by `#[derive(Observation)]`.
pub struct SetOf<I> {
    first: usize,
    mask: usize,
    _items: PhantomData<fn() -> I>,
}

impl<I> fmt::Debug for SetOf<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SetOf")
            .field("first", &self.first)
            .field("mask", &self.mask)
            .finish()
    }
}

impl<I> PartialEq for SetOf<I> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        (self.first, self.mask) == (other.first, other.mask)
    }
}

impl<I> Eq for SetOf<I> {}

impl<I> Clone for SetOf<I> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<I> Copy for SetOf<I> {}

impl<I> SetOf<I> {
    /// The set whose item leaves are `first..mask` and mask leaf `mask`.
    #[must_use]
    #[inline]
    pub const fn new(first: usize, mask: usize) -> Self {
        Self {
            first,
            mask,
            _items: PhantomData,
        }
    }

    /// The first item leaf.
    #[must_use]
    #[inline]
    pub const fn first(self) -> usize {
        self.first
    }

    /// The mask leaf, `[MAX]`.
    #[must_use]
    #[inline]
    pub const fn mask(self) -> usize {
        self.mask
    }
}

/// Typed, bounds-checked writes into one agent's `Set` of item struct `I`:
/// the presence mask and, per slot, one row of each item leaf. Slots left
/// absent are zeroed by the engine after `observe` (ADR-0082 point 3).
pub struct SetOfLeaf<'a, 'b, I> {
    items: &'a mut [&'b mut [u8]],
    mask: &'a mut [u8],
    _items: PhantomData<fn() -> I>,
}

impl<I> fmt::Debug for SetOfLeaf<'_, '_, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SetOfLeaf")
            .field("leaves", &self.items.len())
            .field("capacity", &self.mask.len())
            .finish()
    }
}

impl<'a, 'b, I> SetOfLeaf<'a, 'b, I> {
    #[inline]
    pub(crate) fn new(items: &'a mut [&'b mut [u8]], mask: &'a mut [u8]) -> Self {
        Self {
            items,
            mask,
            _items: PhantomData,
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

    /// Slot `i`'s row of the item leaf `id`, or `None` if the slot or the
    /// leaf does not exist.
    #[must_use]
    #[inline]
    pub fn item<T: LeafDType>(&mut self, i: usize, id: ItemId<I, T>) -> Option<LeafMut<'_, T>> {
        let capacity = self.mask.len();
        if i >= capacity {
            return None;
        }
        let leaf = self.items.get_mut(id.index())?;
        let row = leaf.len().checked_div(capacity)?;
        let start = i.checked_mul(row)?;
        let end = start.checked_add(row)?;
        leaf.get_mut(start..end).map(LeafMut::new)
    }
}

#[cfg(test)]
mod tests {
    use super::{ItemId, SetOf, SetOfLeaf};

    struct Unit;

    #[test]
    fn test_set_of_leaf_item_writes_the_named_leaf_row_with_bounds() {
        let mut a = [0u8; 6];
        let mut b = [0u8; 3];
        let mut mask = [0u8; 3];
        let mut items = [&mut a[..], &mut b[..]];
        let mut s = SetOfLeaf::<Unit>::new(&mut items, &mut mask);
        assert_eq!(s.capacity(), 3);
        assert!(s.set(2, true));
        assert!(!s.set(3, true));
        assert!(
            s.item(1, ItemId::<Unit, u16>::new(0))
                .unwrap()
                .set(0, 0x0102)
        );
        assert!(s.item(2, ItemId::<Unit, u8>::new(1)).unwrap().set(0, 9));
        assert!(s.item(3, ItemId::<Unit, u8>::new(1)).is_none());
        assert!(s.item(0, ItemId::<Unit, u8>::new(2)).is_none());
        assert_eq!(s.item(0, ItemId::<Unit, u16>::new(0)).unwrap().len(), 1);
        assert_eq!(mask, [0, 0, 1]);
        assert_eq!(a, [0, 0, 2, 1, 0, 0]);
        assert_eq!(b, [0, 0, 9]);
        let id = SetOf::<Unit>::new(4, 6);
        assert_eq!((id.first(), id.mask()), (4, 6));
        assert_eq!(ItemId::<Unit, u8>::new(5).index(), 5);
    }

    #[test]
    fn test_set_of_leaf_of_no_slots_has_no_rows() {
        let mut a = [0u8; 0];
        let mut mask = [0u8; 0];
        let mut items = [&mut a[..]];
        let mut s = SetOfLeaf::<Unit>::new(&mut items, &mut mask);
        assert!(s.item(0, ItemId::<Unit, u8>::new(0)).is_none());
        assert!(!s.set(0, true));
    }
}
