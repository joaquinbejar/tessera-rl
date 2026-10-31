// SPDX-License-Identifier: MIT OR Apache-2.0

//! Supported element types (ADR-0014).

/// The element type of a box space or leaf buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// One-byte boolean.
    Bool,
    /// `u8`.
    U8,
    /// `i8`.
    I8,
    /// `u16`.
    U16,
    /// `i16`.
    I16,
    /// `u32`.
    U32,
    /// `i32`.
    I32,
    /// `i64`.
    I64,
    /// `f32`.
    F32,
    /// `f64`.
    F64,
}

impl DType {
    /// Size of one element in bytes.
    #[must_use]
    #[inline]
    pub const fn size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    /// Returns whether the type is a float.
    #[must_use]
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }
}

#[cfg(test)]
mod tests {
    use super::DType;

    #[test]
    fn test_dtype_sizes_match_rust_types() {
        assert_eq!(DType::Bool.size(), 1);
        assert_eq!(DType::I16.size(), size_of::<i16>());
        assert_eq!(DType::F32.size(), size_of::<f32>());
        assert_eq!(DType::I64.size(), size_of::<i64>());
        assert!(DType::F64.is_float() && !DType::U32.is_float());
    }
}
