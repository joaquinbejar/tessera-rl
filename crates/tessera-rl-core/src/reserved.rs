// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Reserved<N>`: explicit padding in state (ADR-0029).

/// `N` bytes that are always zero. State types use it to fill alignment
/// gaps so that they have no compiler padding; decoding rejects non-zero
/// bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Reserved<const N: usize>([u8; N]);

impl<const N: usize> Reserved<N> {
    /// The only value: all zero.
    pub const ZERO: Self = Self([0; N]);
}

impl<const N: usize> Default for Reserved<N> {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::Reserved;

    #[test]
    fn test_reserved_default_is_zero_and_sized() {
        assert_eq!(Reserved::<3>::default(), Reserved::<3>::ZERO);
        assert_eq!(size_of::<Reserved<3>>(), 3);
    }
}
