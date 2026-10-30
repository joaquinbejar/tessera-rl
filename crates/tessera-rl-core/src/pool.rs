// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Pool<T, CAP>`: fixed-capacity variable-size state (SPEC 5.1, ADR-0029).

use crate::bool8::Bool8;
use crate::encoding::{Canonical, Decoder, Encoder, Sink};
use crate::error::DecodeError;

/// Up to `CAP` entries with an occupancy array. Inactive entries always
/// hold `T::default()`, so equal pools encode identically.
///
/// The layout is `items` then `occupied`. A containing state struct must
/// still be padding-free: choose `CAP` (or add `Reserved` fields) so the
/// occupancy bytes end on the alignment of the next field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Pool<T, const CAP: usize> {
    items: [T; CAP],
    occupied: [Bool8; CAP],
}

impl<T: Copy + Default, const CAP: usize> Default for Pool<T, CAP> {
    #[inline]
    fn default() -> Self {
        Self {
            items: [T::default(); CAP],
            occupied: [Bool8::FALSE; CAP],
        }
    }
}

impl<T: Copy + Default, const CAP: usize> Pool<T, CAP> {
    /// The compile-time capacity.
    pub const CAPACITY: usize = CAP;

    /// Returns the number of occupied entries.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.occupied.iter().filter(|b| b.get()).count()
    }

    /// Returns whether no entry is occupied.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        !self.occupied.iter().any(|b| b.get())
    }

    /// Returns whether entry `index` is occupied; `false` out of range.
    #[must_use]
    #[inline]
    pub fn is_occupied(&self, index: usize) -> bool {
        self.occupied.get(index).is_some_and(|b| b.get())
    }

    /// Returns the occupied entry `index`.
    #[must_use]
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if self.is_occupied(index) {
            self.items.get(index)
        } else {
            None
        }
    }

    /// Returns the occupied entry `index` mutably.
    #[must_use]
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if self.is_occupied(index) {
            self.items.get_mut(index)
        } else {
            None
        }
    }

    /// Stores `value` in the first free entry and returns its index, or
    /// `None` if the pool is full. Lowest index first, so insertion order
    /// is deterministic.
    #[inline]
    pub fn insert(&mut self, value: T) -> Option<usize> {
        let index = self.occupied.iter().position(|b| !b.get())?;
        let (slot, flag) = (self.items.get_mut(index)?, self.occupied.get_mut(index)?);
        *slot = value;
        *flag = Bool8::TRUE;
        Some(index)
    }

    /// Removes and returns entry `index`, resetting it to `T::default()`.
    #[inline]
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if !self.is_occupied(index) {
            return None;
        }
        let (slot, flag) = (self.items.get_mut(index)?, self.occupied.get_mut(index)?);
        *flag = Bool8::FALSE;
        Some(core::mem::take(slot))
    }

    /// Iterates over occupied entries in index order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.items
            .iter()
            .zip(&self.occupied)
            .enumerate()
            .filter_map(|(i, (item, flag))| flag.get().then_some((i, item)))
    }
}

impl<T, const CAP: usize> Canonical for Pool<T, CAP>
where
    T: Canonical + Copy + Default + PartialEq,
{
    const ENCODED_LEN: usize = <[T; CAP]>::ENCODED_LEN + CAP;

    #[inline]
    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        self.items.encode(out);
        self.occupied.encode(out);
    }

    /// Rejects inactive entries that are not `T::default()`.
    #[inline]
    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        let items = <[T; CAP]>::decode(input)?;
        let occupied = <[Bool8; CAP]>::decode(input)?;
        let default = T::default();
        let bad = items
            .iter()
            .zip(&occupied)
            .position(|(item, flag)| !flag.get() && *item != default);
        match bad {
            Some(index) => Err(DecodeError::InactivePoolEntry { index }),
            None => Ok(Self { items, occupied }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Pool;
    use crate::encoding::{decode_from_slice, encode_to_vec};
    use crate::error::DecodeError;

    #[test]
    fn test_pool_insert_fills_lowest_free_index() {
        let mut p = Pool::<u32, 3>::default();
        assert_eq!(p.insert(10), Some(0));
        assert_eq!(p.insert(11), Some(1));
        assert_eq!(p.remove(0), Some(10));
        assert_eq!(p.insert(12), Some(0));
        assert_eq!(p.insert(13), Some(2));
        assert_eq!(p.insert(14), None);
        assert_eq!(p.len(), 3);
    }

    #[test]
    fn test_pool_remove_resets_to_default() {
        let mut p = Pool::<u32, 2>::default();
        p.insert(7);
        p.remove(0);
        assert_eq!(p, Pool::default());
        assert_eq!(p.remove(0), None);
        assert_eq!(p.get(5), None);
    }

    #[test]
    fn test_pool_iter_yields_occupied_in_order() {
        let mut p = Pool::<u8, 4>::default();
        p.insert(1);
        p.insert(2);
        p.insert(3);
        p.remove(1);
        let got: Vec<_> = p.iter().map(|(i, v)| (i, *v)).collect();
        assert_eq!(got, [(0, 1), (2, 3)]);
    }

    #[test]
    fn test_pool_decode_rejects_dirty_inactive_entry() {
        let mut p = Pool::<u8, 2>::default();
        p.insert(5);
        let mut bytes = encode_to_vec(&p);
        assert_eq!(bytes, [5, 0, 1, 0]);
        assert_eq!(decode_from_slice::<Pool<u8, 2>>(&bytes), Ok(p));
        bytes[1] = 9;
        assert_eq!(
            decode_from_slice::<Pool<u8, 2>>(&bytes),
            Err(DecodeError::InactivePoolEntry { index: 1 })
        );
    }
}
