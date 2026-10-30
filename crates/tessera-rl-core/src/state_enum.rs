// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `StateEnum` trait for enums stored in state (ADR-0029).
//!
//! `#[derive(StateEnum)]` (milestone 4) implements it for enums with an
//! integer representation and a `#[default]` variant.

/// An enum stored in state through its integer representation.
///
/// Decoding validates the representation with [`StateEnum::from_repr`], so
/// invalid discriminants are rejected instead of producing an invalid enum.
pub trait StateEnum: Copy + Default + 'static {
    /// The integer representation (`#[repr(u8)]` and so on).
    type Repr: Copy + Into<i64>;

    /// The enum's name, used in decode errors.
    const NAME: &'static str;

    /// Returns the integer representation.
    fn to_repr(self) -> Self::Repr;

    /// Returns the variant for `repr`, or `None` if no variant has it.
    fn from_repr(repr: Self::Repr) -> Option<Self>;
}
