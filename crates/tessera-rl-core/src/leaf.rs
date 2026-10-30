// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed observation leaves (SPEC 11, ADR-0014).

use core::marker::PhantomData;

use crate::bool8::Bool8;
use crate::error::SpaceError;
use crate::space::{BoxBounds, DType, Space};

/// An element type that observation leaves can hold.
pub trait LeafDType: Copy + 'static {
    /// The dtype of the leaf buffer.
    const DTYPE: DType;

    /// Writes `self` little-endian into `dst`, which has `DTYPE.size()`
    /// bytes.
    fn write_le(self, dst: &mut [u8]);

    /// Box bounds `low..=high` of this element type, typed for its dtype.
    fn box_bounds(low: Self, high: Self) -> BoxBounds;
}

macro_rules! leaf_dtype {
    ($($ty:ty => $dtype:ident, $variant:ident, $wide:ty);* $(;)?) => {$(
        impl LeafDType for $ty {
            const DTYPE: DType = DType::$dtype;

            #[inline]
            fn write_le(self, dst: &mut [u8]) {
                dst.copy_from_slice(&self.to_le_bytes());
            }

            #[inline]
            fn box_bounds(low: Self, high: Self) -> BoxBounds {
                BoxBounds::$variant {
                    low: vec![<$wide>::from(low)],
                    high: vec![<$wide>::from(high)],
                }
            }
        }
    )*};
}

leaf_dtype! {
    u8 => U8, UInt, u64; u16 => U16, UInt, u64; u32 => U32, UInt, u64;
    i8 => I8, Int, i64; i16 => I16, Int, i64; i32 => I32, Int, i64; i64 => I64, Int, i64;
    f32 => F32, Float, f64; f64 => F64, Float, f64;
}

impl LeafDType for Bool8 {
    const DTYPE: DType = DType::Bool;

    #[inline]
    fn write_le(self, dst: &mut [u8]) {
        dst.copy_from_slice(&[self.to_byte()]);
    }

    #[inline]
    fn box_bounds(low: Self, high: Self) -> BoxBounds {
        BoxBounds::Int {
            low: vec![i64::from(low.to_byte())],
            high: vec![i64::from(high.to_byte())],
        }
    }
}

/// Marker field type of an `#[derive(Observation)]` struct: one leaf of
/// element type `T`. It holds no data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ObsLeaf<T>(PhantomData<fn() -> T>);

/// Identifies leaf `index` of an observation, with its element type.
/// Generated as associated constants by `#[derive(Observation)]`.
#[derive(Debug, PartialEq, Eq)]
pub struct LeafId<T> {
    index: usize,
    _dtype: PhantomData<fn() -> T>,
}

impl<T> Clone for LeafId<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for LeafId<T> {}

impl<T> LeafId<T> {
    /// The leaf at position `index` in declaration order.
    #[must_use]
    #[inline]
    pub const fn new(index: usize) -> Self {
        Self {
            index,
            _dtype: PhantomData,
        }
    }

    /// The leaf position.
    #[must_use]
    #[inline]
    pub const fn index(self) -> usize {
        self.index
    }
}

/// Typed, bounds-checked writes into one leaf of one agent.
#[derive(Debug)]
pub struct LeafMut<'a, T> {
    bytes: &'a mut [u8],
    _dtype: PhantomData<fn() -> T>,
}

impl<'a, T: LeafDType> LeafMut<'a, T> {
    #[inline]
    pub(crate) fn new(bytes: &'a mut [u8]) -> Self {
        Self {
            bytes,
            _dtype: PhantomData,
        }
    }

    /// Number of elements.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.bytes.len() / T::DTYPE.size()
    }

    /// Returns whether the leaf has no elements.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Writes element `index`; returns `false` if it is out of range.
    #[inline]
    pub fn set(&mut self, index: usize, value: T) -> bool {
        let size = T::DTYPE.size();
        let Some(start) = index.checked_mul(size) else {
            return false;
        };
        match start
            .checked_add(size)
            .and_then(|end| self.bytes.get_mut(start..end))
        {
            Some(dst) => {
                value.write_le(dst);
                true
            }
            None => false,
        }
    }

    /// Writes every element.
    #[inline]
    pub fn fill(&mut self, value: T) {
        for chunk in self.bytes.chunks_exact_mut(T::DTYPE.size()) {
            value.write_le(chunk);
        }
    }
}

/// An observation type: its space for a config, as leaves in declaration
/// order. Implemented by `#[derive(Observation)]`.
pub trait Observation<C>: 'static {
    /// Number of leaves.
    const LEAVES: usize;

    /// The observation space for `cfg`.
    ///
    /// # Errors
    ///
    /// Returns a [`SpaceError`] if `cfg` yields an invalid space.
    fn space(cfg: &C) -> Result<Space, SpaceError>;
}

#[cfg(test)]
mod tests {
    use super::{LeafId, LeafMut};

    #[test]
    fn test_leaf_mut_writes_little_endian_with_bounds() {
        let mut bytes = [0u8; 4];
        let mut leaf = LeafMut::<u16>::new(&mut bytes);
        assert_eq!(leaf.len(), 2);
        assert!(leaf.set(1, 0x0102));
        assert!(!leaf.set(2, 7));
        assert_eq!(bytes, [0, 0, 2, 1]);
        let mut floats = [0u8; 8];
        LeafMut::<f32>::new(&mut floats).fill(-0.0);
        assert_eq!(floats, [0, 0, 0, 0x80, 0, 0, 0, 0x80]);
        assert_eq!(LeafId::<u8>::new(3).index(), 3);
    }
}
