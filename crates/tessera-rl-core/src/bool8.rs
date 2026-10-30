// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Bool8`: the one-byte boolean used in state (ADR-0029).

/// A boolean stored as one byte holding 0 or 1.
///
/// State types use `Bool8` instead of `bool` so every byte pattern of a
/// record is either valid or rejected by decoding, never undefined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Bool8(u8);

impl Bool8 {
    /// False (0).
    pub const FALSE: Self = Self(0);
    /// True (1).
    pub const TRUE: Self = Self(1);

    /// Converts from `bool`.
    #[must_use]
    #[inline]
    pub const fn new(value: bool) -> Self {
        if value { Self::TRUE } else { Self::FALSE }
    }

    /// Returns the value as `bool`.
    #[must_use]
    #[inline]
    pub const fn get(self) -> bool {
        self.0 != 0
    }

    /// Returns the raw byte, 0 or 1.
    #[must_use]
    #[inline]
    pub const fn to_byte(self) -> u8 {
        self.0
    }

    /// Validates a raw byte: `Some` for 0 or 1, `None` otherwise.
    #[must_use]
    #[inline]
    pub const fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::FALSE),
            1 => Some(Self::TRUE),
            _ => None,
        }
    }
}

impl From<bool> for Bool8 {
    #[inline]
    fn from(value: bool) -> Self {
        Self::new(value)
    }
}

impl From<Bool8> for bool {
    #[inline]
    fn from(value: Bool8) -> Self {
        value.get()
    }
}

#[cfg(test)]
mod tests {
    use super::Bool8;

    #[test]
    fn test_bool8_from_byte_rejects_values_above_one() {
        assert_eq!(Bool8::from_byte(0), Some(Bool8::FALSE));
        assert_eq!(Bool8::from_byte(1), Some(Bool8::TRUE));
        assert_eq!(Bool8::from_byte(2), None);
        assert_eq!(Bool8::from_byte(255), None);
    }

    #[test]
    fn test_bool8_round_trips_bool() {
        assert!(bool::from(Bool8::from(true)));
        assert!(!Bool8::default().get());
        assert_eq!(size_of::<Bool8>(), 1);
    }
}
