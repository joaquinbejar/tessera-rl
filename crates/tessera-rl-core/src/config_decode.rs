// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decoding a config from a front end's values (ADR-0065 item 3): the
//! host factory and `export_host_env!` build a [`ConfigSource`] (Rust code,
//! or Python kwargs) and decode the environment's config from it with one
//! shared decoder, so both front ends give the same config.
//!
//! The rules, for the decoder `#[derive(EnvConfig)]` generates:
//!
//! - decoding starts from the generated `Default` and overwrites the
//!   fields whose keys are given, so a partial source is a valid source;
//! - unknown and repeated keys are errors, and so are the keys of adaptive
//!   fields (ADR-0030), which keep their declared default;
//! - integers come from [`ConfigItem::Int`] only and are range-checked,
//!   never wrapped or saturated;
//! - floats come from [`ConfigItem::Float`], or from an `Int` exactly
//!   representable in the field's type; an `f64` into an `f32` field is
//!   rounded to the nearest `f32`, ties to even (beyond `f32`'s range it
//!   becomes an infinity), and float bits are otherwise kept as given;
//! - `Fx32` and `Fx64` come from [`ConfigItem::Str`] only, a decimal parsed
//!   exactly (`Fx32::from_decimal`);
//! - `bool` comes from [`ConfigItem::Bool`] only;
//! - a nested config comes from a [`ConfigItem::Table`], at most
//!   [`MAX_CONFIG_DEPTH`] tables deep;
//! - `EnvConfig::validate` runs last; its error is
//!   [`ConfigDecodeError::Invalid`].

use tessera_rl_num::{DecimalError, Fx32, Fx64};
use thiserror::Error;

use crate::env::EnvConfig;
use crate::error::ConfigError;

/// Most nested tables a source may have, the top level included.
pub const MAX_CONFIG_DEPTH: u32 = 8;

/// One value of a config source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConfigItem<'a> {
    /// An integer (wide enough for every integer field type).
    Int(i128),
    /// A float.
    Float(f64),
    /// A boolean.
    Bool(bool),
    /// A string: the exact decimal of a fixed-point field.
    Str(&'a str),
    /// A nested config.
    Table(ConfigSource<'a>),
}

/// A config as a front end gives it: an ordered list of `(key, item)`
/// pairs. It owns nothing; the caller keeps the pairs alive.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ConfigSource<'a> {
    entries: &'a [(&'a str, ConfigItem<'a>)],
}

impl<'a> ConfigSource<'a> {
    /// A source over `entries`.
    #[must_use]
    pub const fn new(entries: &'a [(&'a str, ConfigItem<'a>)]) -> Self {
        Self { entries }
    }

    /// A source with no entries: the config's default.
    #[must_use]
    pub const fn empty() -> Self {
        Self { entries: &[] }
    }

    /// The entries, in order.
    #[must_use]
    pub const fn entries(&self) -> &'a [(&'a str, ConfigItem<'a>)] {
        self.entries
    }
}

/// Why a source does not decode into a config.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ConfigDecodeError {
    /// A key that names no field (or an adaptive field).
    #[error("unknown config key `{key}`")]
    UnknownKey {
        /// The key.
        key: String,
    },
    /// A key given more than once.
    #[error("config key `{key}` given more than once")]
    DuplicateKey {
        /// The key.
        key: String,
    },
    /// A required key is missing (hand-written decoders only: a derived
    /// decoder starts from the default).
    #[error("config key `{field}` is required")]
    Missing {
        /// The field.
        field: &'static str,
    },
    /// An item of the wrong kind for the field.
    #[error("config key `{field}` expects {expected}")]
    WrongType {
        /// The field.
        field: &'static str,
        /// What it accepts.
        expected: &'static str,
    },
    /// A value outside the field type's range.
    #[error("config key `{field}` is out of range")]
    OutOfRange {
        /// The field.
        field: &'static str,
    },
    /// A value the field's type cannot represent exactly.
    #[error("config key `{field}` cannot be represented exactly")]
    Inexact {
        /// The field.
        field: &'static str,
    },
    /// Nested tables deeper than [`MAX_CONFIG_DEPTH`].
    #[error("config tables nested deeper than {MAX_CONFIG_DEPTH}")]
    TooDeep,
    /// The decoded config failed `EnvConfig::validate`.
    #[error(transparent)]
    Invalid(#[from] ConfigError),
}

/// A config decoded from a [`ConfigSource`]. `#[derive(EnvConfig)]`
/// generates it.
///
/// A hand-written decoder implements [`ConfigDecode::decode_at`] (not
/// `decode`, which calls it at depth 1). It should call [`check_keys`]
/// first, which refuses unknown and repeated keys and enforces the depth
/// bound ([`MAX_CONFIG_DEPTH`]); decode each field with [`ConfigField`],
/// and a nested config with [`decode_nested`], which passes the depth on;
/// and run `EnvConfig::validate` last. These helpers are public API: the
/// generated decoder is written with them.
pub trait ConfigDecode: EnvConfig {
    /// Decodes a config from `src` and validates it.
    ///
    /// # Errors
    ///
    /// A [`ConfigDecodeError`] naming the first key or field that fails.
    fn decode(src: &ConfigSource<'_>) -> Result<Self, ConfigDecodeError> {
        Self::decode_at(src, 1)
    }

    /// Decodes `src` as a table at nesting `depth` (1 for the top level).
    ///
    /// # Errors
    ///
    /// As [`ConfigDecode::decode`], and [`ConfigDecodeError::TooDeep`] past
    /// [`MAX_CONFIG_DEPTH`].
    fn decode_at(src: &ConfigSource<'_>, depth: u32) -> Result<Self, ConfigDecodeError>;
}

/// One config field's type, decoded from one item. Used by generated code;
/// implemented for every field type `#[derive(EnvConfig)]` accepts.
pub trait ConfigField: Sized {
    /// Decodes `item` for the field `field` at table `depth`.
    ///
    /// # Errors
    ///
    /// A [`ConfigDecodeError`] naming `field`.
    fn from_item(
        item: &ConfigItem<'_>,
        field: &'static str,
        depth: u32,
    ) -> Result<Self, ConfigDecodeError>;
}

/// Checks that every key of `src` is one of `known` and appears once.
/// Used by generated code, before any field is decoded.
///
/// # Errors
///
/// [`ConfigDecodeError::UnknownKey`] or [`ConfigDecodeError::DuplicateKey`]
/// for the first offending key, and [`ConfigDecodeError::TooDeep`] past
/// [`MAX_CONFIG_DEPTH`].
pub fn check_keys(
    src: &ConfigSource<'_>,
    known: &[&str],
    depth: u32,
) -> Result<(), ConfigDecodeError> {
    if depth > MAX_CONFIG_DEPTH {
        return Err(ConfigDecodeError::TooDeep);
    }
    for (n, (key, _)) in src.entries().iter().enumerate() {
        if !known.contains(key) {
            return Err(ConfigDecodeError::UnknownKey {
                key: (*key).to_owned(),
            });
        }
        if src.entries().iter().take(n).any(|(k, _)| k == key) {
            return Err(ConfigDecodeError::DuplicateKey {
                key: (*key).to_owned(),
            });
        }
    }
    Ok(())
}

/// Decodes a nested config field from a table one level deeper. Used by
/// generated code.
///
/// # Errors
///
/// [`ConfigDecodeError::WrongType`] for an item that is not a table, and
/// the nested config's own errors.
pub fn decode_nested<C: ConfigDecode>(
    item: &ConfigItem<'_>,
    field: &'static str,
    depth: u32,
) -> Result<C, ConfigDecodeError> {
    match item {
        ConfigItem::Table(src) => {
            C::decode_at(src, depth.checked_add(1).ok_or(ConfigDecodeError::TooDeep)?)
        }
        _ => Err(wrong(field, "a table")),
    }
}

fn wrong(field: &'static str, expected: &'static str) -> ConfigDecodeError {
    ConfigDecodeError::WrongType { field, expected }
}

macro_rules! int_field {
    ($($ty:ty),*) => {$(
        impl ConfigField for $ty {
            fn from_item(
                item: &ConfigItem<'_>,
                field: &'static str,
                _depth: u32,
            ) -> Result<Self, ConfigDecodeError> {
                match item {
                    ConfigItem::Int(v) => {
                        <$ty>::try_from(*v).map_err(|_| ConfigDecodeError::OutOfRange { field })
                    }
                    _ => Err(wrong(field, "an integer")),
                }
            }
        }
    )*};
}

int_field!(u8, u16, u32, u64, i8, i16, i32, i64);

/// 2^127, exact in both float types: `i128` values lie in `[-2^127, 2^127)`.
const I128_BOUND: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0;

impl ConfigField for f64 {
    fn from_item(
        item: &ConfigItem<'_>,
        field: &'static str,
        _depth: u32,
    ) -> Result<Self, ConfigDecodeError> {
        match item {
            ConfigItem::Float(v) => Ok(*v),
            ConfigItem::Int(v) => {
                #[expect(clippy::cast_precision_loss, reason = "exactness is checked below")]
                let f = *v as f64;
                // `as i128` saturates, so the round trip is exact only for
                // values below 2^127 in magnitude.
                #[expect(clippy::cast_possible_truncation, reason = "the round-trip check")]
                let back = f as i128;
                if (-I128_BOUND..I128_BOUND).contains(&f) && back == *v {
                    Ok(f)
                } else {
                    Err(ConfigDecodeError::Inexact { field })
                }
            }
            _ => Err(wrong(field, "a float or an exact integer")),
        }
    }
}

impl ConfigField for f32 {
    fn from_item(
        item: &ConfigItem<'_>,
        field: &'static str,
        _depth: u32,
    ) -> Result<Self, ConfigDecodeError> {
        match item {
            // Round to nearest, ties to even (the owner's decision,
            // ADR-0065 item 3); beyond f32's range an infinity. A NaN stays
            // a NaN, but Rust does not guarantee its sign or payload through
            // the cast; a derived config's `validate` refuses NaN anyway.
            #[expect(clippy::cast_possible_truncation, reason = "round to nearest f32")]
            ConfigItem::Float(v) => Ok(*v as f32),
            ConfigItem::Int(v) => {
                #[expect(clippy::cast_precision_loss, reason = "exactness is checked below")]
                let f = *v as f32;
                let exact = (-I128_BOUND..I128_BOUND).contains(&f64::from(f)) && {
                    #[expect(clippy::cast_possible_truncation, reason = "the round-trip check")]
                    let back = f as i128;
                    back == *v
                };
                if exact {
                    Ok(f)
                } else {
                    Err(ConfigDecodeError::Inexact { field })
                }
            }
            _ => Err(wrong(field, "a float or an exact integer")),
        }
    }
}

impl ConfigField for bool {
    fn from_item(
        item: &ConfigItem<'_>,
        field: &'static str,
        _depth: u32,
    ) -> Result<Self, ConfigDecodeError> {
        match item {
            ConfigItem::Bool(b) => Ok(*b),
            _ => Err(wrong(field, "a bool")),
        }
    }
}

fn decimal_error(field: &'static str, e: DecimalError) -> ConfigDecodeError {
    match e {
        DecimalError::Syntax => wrong(field, "an exact decimal string"),
        DecimalError::OutOfRange => ConfigDecodeError::OutOfRange { field },
        DecimalError::Inexact => ConfigDecodeError::Inexact { field },
    }
}

impl ConfigField for Fx32 {
    fn from_item(
        item: &ConfigItem<'_>,
        field: &'static str,
        _depth: u32,
    ) -> Result<Self, ConfigDecodeError> {
        match item {
            ConfigItem::Str(s) => Self::from_decimal(s).map_err(|e| decimal_error(field, e)),
            _ => Err(wrong(field, "an exact decimal string")),
        }
    }
}

impl ConfigField for Fx64 {
    fn from_item(
        item: &ConfigItem<'_>,
        field: &'static str,
        _depth: u32,
    ) -> Result<Self, ConfigDecodeError> {
        match item {
            ConfigItem::Str(s) => Self::from_decimal(s).map_err(|e| decimal_error(field, e)),
            _ => Err(wrong(field, "an exact decimal string")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ConfigDecodeError, ConfigField, ConfigItem, ConfigSource, check_keys};
    use tessera_rl_num::{Fx32, Fx64};

    fn get<T: ConfigField>(item: ConfigItem<'_>) -> Result<T, ConfigDecodeError> {
        T::from_item(&item, "f", 1)
    }

    fn range(field: &'static str) -> ConfigDecodeError {
        ConfigDecodeError::OutOfRange { field }
    }

    #[test]
    fn test_config_field_integers_at_their_bounds_and_one_past() {
        macro_rules! bounds {
            ($($ty:ty),*) => {$(
                let (lo, hi) = (i128::from(<$ty>::MIN), i128::from(<$ty>::MAX));
                assert_eq!(get::<$ty>(ConfigItem::Int(lo)), Ok(<$ty>::MIN));
                assert_eq!(get::<$ty>(ConfigItem::Int(hi)), Ok(<$ty>::MAX));
                assert_eq!(get::<$ty>(ConfigItem::Int(lo - 1)), Err(range("f")));
                assert_eq!(get::<$ty>(ConfigItem::Int(hi + 1)), Err(range("f")));
                assert!(matches!(get::<$ty>(ConfigItem::Float(1.0)), Err(ConfigDecodeError::WrongType { .. })));
                assert!(matches!(get::<$ty>(ConfigItem::Bool(true)), Err(ConfigDecodeError::WrongType { .. })));
            )*};
        }
        bounds!(u8, u16, u32, u64, i8, i16, i32, i64);
    }

    #[test]
    fn test_config_field_integer_into_float_exact_or_refused() {
        assert_eq!(
            get::<f64>(ConfigItem::Int(1 << 53)),
            Ok(9_007_199_254_740_992.0)
        );
        assert_eq!(
            get::<f64>(ConfigItem::Int((1 << 53) + 1)),
            Err(ConfigDecodeError::Inexact { field: "f" })
        );
        assert_eq!(get::<f32>(ConfigItem::Int(1 << 24)), Ok(16_777_216.0));
        assert_eq!(
            get::<f32>(ConfigItem::Int((1 << 24) + 1)),
            Err(ConfigDecodeError::Inexact { field: "f" })
        );
        assert_eq!(
            get::<f64>(ConfigItem::Int(i128::MAX)),
            Err(ConfigDecodeError::Inexact { field: "f" })
        );
        assert_eq!(
            get::<f64>(ConfigItem::Int(i128::MIN)),
            Ok(-1.701_411_834_604_692_3e38)
        );
    }

    #[test]
    fn test_config_field_f64_into_f32_rounds_to_nearest() {
        assert_eq!(
            get::<f32>(ConfigItem::Float(0.1)).map(f32::to_bits),
            Ok(0.1f32.to_bits())
        );
        // 1 + 2^-24 is halfway between 1 and the next f32: ties to even.
        let tie = 1.0 + f64::from(f32::EPSILON) / 2.0;
        assert_eq!(get::<f32>(ConfigItem::Float(tie)), Ok(1.0));
        let tie_up = 1.0 + 3.0 * f64::from(f32::EPSILON) / 2.0;
        assert_eq!(
            get::<f32>(ConfigItem::Float(tie_up)),
            Ok(1.0 + 2.0 * f32::EPSILON)
        );
        assert_eq!(get::<f32>(ConfigItem::Float(1e300)), Ok(f32::INFINITY));
        assert_eq!(get::<f32>(ConfigItem::Float(-1e300)), Ok(f32::NEG_INFINITY));
        let neg_zero = get::<f32>(ConfigItem::Float(-0.0)).map(f32::to_bits);
        assert_eq!(neg_zero, Ok((-0.0f32).to_bits()));
        assert_eq!(
            get::<f64>(ConfigItem::Float(-0.0)).map(f64::to_bits),
            Ok((-0.0f64).to_bits())
        );
        let nan = f64::from_bits(0xfff8_0000_0000_0001);
        assert_eq!(
            get::<f64>(ConfigItem::Float(nan)).map(f64::to_bits),
            Ok(nan.to_bits())
        );
        assert!(matches!(
            get::<f32>(ConfigItem::Str("1")),
            Err(ConfigDecodeError::WrongType { .. })
        ));
    }

    #[test]
    fn test_config_field_fixed_point_from_exact_decimal_strings_only() {
        assert_eq!(
            get::<Fx32>(ConfigItem::Str("1.5")),
            Fx32::from_decimal("1.5").map_err(|_| range("x"))
        );
        assert_eq!(
            get::<Fx64>(ConfigItem::Str("-0.25")),
            Fx64::from_decimal("-0.25").map_err(|_| range("x"))
        );
        assert_eq!(
            get::<Fx32>(ConfigItem::Str("0.1")),
            Err(ConfigDecodeError::Inexact { field: "f" })
        );
        assert_eq!(get::<Fx32>(ConfigItem::Str("40000")), Err(range("f")));
        for item in [
            ConfigItem::Float(1.5),
            ConfigItem::Int(1),
            ConfigItem::Str("1e3"),
        ] {
            assert!(
                matches!(get::<Fx32>(item), Err(ConfigDecodeError::WrongType { .. })),
                "{item:?}"
            );
            assert!(
                matches!(get::<Fx64>(item), Err(ConfigDecodeError::WrongType { .. })),
                "{item:?}"
            );
        }
    }

    #[test]
    fn test_config_field_bool_from_bool_only() {
        assert_eq!(get::<bool>(ConfigItem::Bool(true)), Ok(true));
        assert!(matches!(
            get::<bool>(ConfigItem::Int(1)),
            Err(ConfigDecodeError::WrongType { .. })
        ));
    }

    #[test]
    fn test_config_check_keys_unknown_duplicate_and_depth() {
        let entries = [("a", ConfigItem::Int(1)), ("b", ConfigItem::Int(2))];
        let src = ConfigSource::new(&entries);
        assert_eq!(check_keys(&src, &["a", "b"], 1), Ok(()));
        assert_eq!(
            check_keys(&src, &["a"], 1),
            Err(ConfigDecodeError::UnknownKey {
                key: "b".to_owned()
            })
        );
        let dup = [("a", ConfigItem::Int(1)), ("a", ConfigItem::Int(2))];
        assert_eq!(
            check_keys(&ConfigSource::new(&dup), &["a"], 1),
            Err(ConfigDecodeError::DuplicateKey {
                key: "a".to_owned()
            })
        );
        assert_eq!(check_keys(&ConfigSource::empty(), &[], 8), Ok(()));
        assert_eq!(
            check_keys(&ConfigSource::empty(), &[], 9),
            Err(ConfigDecodeError::TooDeep)
        );
    }
}
