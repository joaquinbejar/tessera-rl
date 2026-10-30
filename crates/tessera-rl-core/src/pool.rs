// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Pool<T, CAP>`: fixed-capacity variable-size state (SPEC 5.1, ADR-0029).

use crate::bool8::Bool8;
use crate::encoding::{Canonical, Decoder, Encoder, Sink, canonical_eq};
use crate::error::DecodeError;
use crate::state_init::StateInit;

/// Up to `CAP` entries with an occupancy array. Inactive entries always
/// hold `T::state_default()` (canonically: the same bytes), so equal pools
/// encode identically.
///
/// `insert`, `len` and `is_empty` scan the occupancy array, so they are
/// O(CAP). `insert` reuses the lowest free index; indices are therefore not
/// stable handles across removals.
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

impl<T: StateInit, const CAP: usize> StateInit for Pool<T, CAP> {
    #[inline]
    fn state_default() -> Self {
        Self {
            items: <[T; CAP]>::state_default(),
            occupied: [Bool8::FALSE; CAP],
        }
    }
}

impl<T: StateInit, const CAP: usize> Default for Pool<T, CAP> {
    #[inline]
    fn default() -> Self {
        Self::state_default()
    }
}

impl<T: StateInit, const CAP: usize> Pool<T, CAP> {
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

    /// Removes and returns entry `index`, resetting it to
    /// `T::state_default()`.
    #[inline]
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if !self.is_occupied(index) {
            return None;
        }
        let (slot, flag) = (self.items.get_mut(index)?, self.occupied.get_mut(index)?);
        *flag = Bool8::FALSE;
        Some(core::mem::replace(slot, T::state_default()))
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

    /// Iterates mutably over occupied entries in index order.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut T)> {
        self.items
            .iter_mut()
            .zip(&self.occupied)
            .enumerate()
            .filter_map(|(i, (item, flag))| flag.get().then_some((i, item)))
    }
}

impl<T, const CAP: usize> Canonical for Pool<T, CAP>
where
    T: Canonical + StateInit,
{
    const ENCODED_LEN: usize = <[T; CAP]>::ENCODED_LEN + CAP;

    #[inline]
    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        self.items.encode(out);
        self.occupied.encode(out);
    }

    /// Rejects inactive entries whose canonical bytes differ from those of
    /// `T::state_default()`; `PartialEq` is not used, so `-0.0` or a lax
    /// equality cannot hide a difference.
    #[inline]
    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        let items = <[T; CAP]>::decode(input)?;
        let occupied = <[Bool8; CAP]>::decode(input)?;
        let default = T::state_default();
        let bad = items
            .iter()
            .zip(&occupied)
            .position(|(item, flag)| !flag.get() && !canonical_eq(item, &default));
        match bad {
            Some(index) => Err(DecodeError::InactivePoolEntry { index }),
            None => Ok(Self { items, occupied }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Pool;
    use crate::encoding::{Canonical, Decoder, Encoder, Sink, decode_from_slice, encode_to_vec};
    use crate::error::DecodeError;
    use crate::state_init::StateInit;

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
    fn test_pool_iter_mut_updates_occupied_only() {
        let mut p = Pool::<u8, 3>::default();
        p.insert(1);
        p.insert(2);
        p.remove(0);
        for (_, v) in p.iter_mut() {
            *v += 10;
        }
        assert_eq!(p.get(1), Some(&12));
        assert_eq!(p.get(0), None);
    }

    #[test]
    fn test_pool_decode_rejects_inactive_negative_zero() {
        let mut p = Pool::<f64, 2>::default();
        p.insert(1.5);
        let mut bytes = encode_to_vec(&p);
        bytes[8..16].copy_from_slice(&(-0.0f64).to_bits().to_le_bytes());
        assert_eq!(
            decode_from_slice::<Pool<f64, 2>>(&bytes),
            Err(DecodeError::InactivePoolEntry { index: 1 })
        );
    }

    #[test]
    fn test_pool_decode_rejects_inactive_nan_payload() {
        let p = Pool::<f64, 1>::default();
        let mut bytes = encode_to_vec(&p);
        bytes[..8].copy_from_slice(&0x7ff8_0000_0000_0001u64.to_le_bytes());
        assert!(decode_from_slice::<Pool<f64, 1>>(&bytes).is_err());
    }

    /// Equality that ignores the value: decoding must not rely on it.
    #[derive(Debug, Clone, Copy)]
    struct Lax(u8);

    impl PartialEq for Lax {
        fn eq(&self, _: &Self) -> bool {
            true
        }
    }

    impl StateInit for Lax {
        fn state_default() -> Self {
            Self(0)
        }
    }

    impl Canonical for Lax {
        const ENCODED_LEN: usize = 1;
        fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
            self.0.encode(out);
        }
        fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
            u8::decode(input).map(Self)
        }
    }

    #[test]
    fn test_pool_decode_ignores_lax_partial_eq() {
        let bytes = [5u8, 0];
        assert_eq!(
            decode_from_slice::<Pool<Lax, 1>>(&bytes),
            Err(DecodeError::InactivePoolEntry { index: 0 })
        );
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
