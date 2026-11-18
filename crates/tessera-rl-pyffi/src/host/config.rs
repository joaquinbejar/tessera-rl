// SPDX-License-Identifier: MIT OR Apache-2.0

//! Python keyword arguments as a config source (ADR-0065 item 3): the
//! Python front end of the shared config decoder.
//!
//! Each value maps to one [`ConfigItem`]: `bool` to `Bool` (checked before
//! `int`, of which it is a subclass), an exact `int` to `Int`, an exact
//! `float` to `Float` (its bits as given), an exact `str` to `Str`, and an
//! exact `dict` to `Table`, recursively. Anything else, and an `int` beyond
//! 64 bits, is refused before decoding ([`KwargsError`]). Tables deeper than
//! the decoder's bound are kept empty: the decoder refuses them
//! (`TooDeep`) at the same point as a Rust source.
//!
//! The decoded config is the decoder's: the Rust and Python front ends give
//! the same value, bit for bit, from the same source.

use tessera_rl_core::config_decode::MAX_CONFIG_DEPTH;
use tessera_rl_core::{ConfigDecode, ConfigDecodeError, ConfigItem, ConfigSource};

use crate::errors;
use crate::ffi::capi::{self, CResult, Gil, Kind, Mine, Ref};

/// One owned config value, as read from Python.
#[derive(Debug, Clone, PartialEq)]
pub enum OwnedItem {
    /// An integer of at most 64 bits.
    Int(i128),
    /// A float, bits as given.
    Float(f64),
    /// A boolean.
    Bool(bool),
    /// A string.
    Str(String),
    /// A nested table.
    Table(Vec<(String, OwnedItem)>),
}

/// Why a keyword argument is not a config value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KwargsError {
    /// A value of a Python type no config field takes.
    Type {
        /// The key, dotted for nested tables.
        key: String,
    },
    /// An `int` beyond 64 bits.
    Int {
        /// The key, dotted for nested tables.
        key: String,
    },
}

impl core::fmt::Display for KwargsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Type { key } => write!(
                f,
                "config key `{key}`: a value must be a bool, int, float, str or dict"
            ),
            Self::Int { key } => write!(f, "config key `{key}`: an int beyond 64 bits"),
        }
    }
}

/// The result of reading keyword arguments: the owned entries, or a
/// [`KwargsError`] (a Python exception may also be pending: `CResult`).
pub type Read = Result<Vec<(String, OwnedItem)>, KwargsError>;

/// Reads `entries` (a dict's items) at table `depth` (1 for the keyword
/// arguments themselves).
///
/// # Errors
///
/// A Python exception (`Err` of the outer `CResult`) when reading a value
/// raised; a [`KwargsError`] for a value no config field takes.
pub fn read<'py>(g: Gil<'py>, entries: Vec<(String, Mine<'py>)>, depth: u32) -> CResult<Read> {
    let mut out = Vec::with_capacity(entries.len());
    for (key, value) in entries {
        let item = match item(g, &key, value.as_ref(), depth)? {
            Ok(item) => item,
            Err(e) => return Ok(Err(e)),
        };
        out.push((key, item));
    }
    Ok(Ok(out))
}

fn item(g: Gil<'_>, key: &str, v: Ref<'_>, depth: u32) -> CResult<Result<OwnedItem, KwargsError>> {
    Ok(Ok(match capi::kind(v) {
        Kind::Bool => OwnedItem::Bool(capi::to_bool(g, v, key)?),
        Kind::Int => {
            if let Ok(x) = capi::to_i64(g, v, key) {
                OwnedItem::Int(i128::from(x))
            } else {
                // Beyond i64: an unsigned 64-bit value, or too wide.
                capi::clear_pending(g);
                if let Ok(x) = capi::to_u64(g, v, key) {
                    OwnedItem::Int(i128::from(x))
                } else {
                    capi::clear_pending(g);
                    return Ok(Err(KwargsError::Int {
                        key: key.to_owned(),
                    }));
                }
            }
        }
        Kind::Float => OwnedItem::Float(capi::to_f64(g, v, key)?),
        Kind::Str => OwnedItem::Str(capi::to_string(g, v, key)?),
        Kind::Dict => {
            if depth >= MAX_CONFIG_DEPTH {
                // One table past the bound is kept, empty: the decoder
                // refuses it with `TooDeep` before reading its keys.
                OwnedItem::Table(Vec::new())
            } else {
                let entries = capi::dict_items(g, v, key)?;
                match read(g, entries, depth.saturating_add(1))? {
                    Ok(t) => OwnedItem::Table(t),
                    Err(e) => return Ok(Err(e)),
                }
            }
        }
        Kind::None | Kind::Other => {
            return Ok(Err(KwargsError::Type {
                key: key.to_owned(),
            }));
        }
    }))
}

/// Number of `(key, item)` pairs in `entries`, nested tables included.
fn count(entries: &[(String, OwnedItem)]) -> usize {
    entries
        .iter()
        .map(|(_, item)| {
            1 + match item {
                OwnedItem::Table(t) => count(t),
                _ => 0,
            }
        })
        .sum()
}

/// Lays `entries` out in `buf` (exactly `count(entries)` slots): this
/// table's pairs first, then each nested table's after them, so a table's
/// slice is fully written before its parent borrows it.
fn place<'a>(
    buf: &'a mut [(&'a str, ConfigItem<'a>)],
    entries: &'a [(String, OwnedItem)],
) -> &'a [(&'a str, ConfigItem<'a>)] {
    let (mine, mut rest) = buf.split_at_mut(entries.len().min(buf.len()));
    for (slot, (key, owned)) in mine.iter_mut().zip(entries) {
        let item = match owned {
            OwnedItem::Int(v) => ConfigItem::Int(*v),
            OwnedItem::Float(v) => ConfigItem::Float(*v),
            OwnedItem::Bool(v) => ConfigItem::Bool(*v),
            OwnedItem::Str(s) => ConfigItem::Str(s),
            OwnedItem::Table(t) => {
                let n = count(t).min(rest.len());
                let (child, tail) = core::mem::take(&mut rest).split_at_mut(n);
                rest = tail;
                ConfigItem::Table(ConfigSource::new(place(child, t)))
            }
        };
        *slot = (key.as_str(), item);
    }
    mine
}

/// Decodes `C` from owned `entries` with the shared decoder.
///
/// # Errors
///
/// The decoder's [`ConfigDecodeError`].
pub fn decode<C: ConfigDecode>(entries: &[(String, OwnedItem)]) -> Result<C, ConfigDecodeError> {
    let mut buf: Vec<(&str, ConfigItem<'_>)> = vec![("", ConfigItem::Bool(false)); count(entries)];
    let top = place(&mut buf, entries);
    C::decode(&ConfigSource::new(top))
}

/// Raises the Python error of a config that does not decode: an unknown
/// key is an `ArgumentError` (the factory refuses unknown keywords, SPEC
/// 14.4.1 item 4); every other decoding error is a `ConfigError`.
#[must_use]
pub fn raise_decode(g: Gil<'_>, e: &ConfigDecodeError) -> capi::Raised {
    let kind = match e {
        ConfigDecodeError::UnknownKey { .. } => errors::Kind::Argument,
        _ => errors::Kind::Config,
    };
    errors::error(g, kind, e.to_string())
}

/// Raises the `ConfigError` of a keyword argument no config field takes.
#[must_use]
pub fn raise_kwargs(g: Gil<'_>, e: &KwargsError) -> capi::Raised {
    errors::error(g, errors::Kind::Config, e.to_string())
}

#[cfg(test)]
mod tests {
    use super::{OwnedItem, count, decode};
    use tessera_rl_core::{ConfigDecode, ConfigDecodeError, ConfigItem, ConfigSource, EnvConfig};

    /// A hand-written decoder that echoes the source's shape, to check the
    /// layout without the derive.
    #[derive(Debug, PartialEq)]
    struct Echo(String);

    impl EnvConfig for Echo {}

    fn show(src: &ConfigSource<'_>) -> String {
        src.entries()
            .iter()
            .map(|(k, item)| match item {
                ConfigItem::Table(t) => format!("{k}={{{}}}", show(t)),
                other => format!("{k}={other:?}"),
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    impl ConfigDecode for Echo {
        fn decode_at(src: &ConfigSource<'_>, _: u32) -> Result<Self, ConfigDecodeError> {
            Ok(Self(show(src)))
        }
    }

    fn s(k: &str) -> String {
        k.to_owned()
    }

    #[test]
    fn test_host_config_layout_keeps_order_and_nesting() {
        let entries = vec![
            (s("a"), OwnedItem::Int(-3)),
            (
                s("t"),
                OwnedItem::Table(vec![
                    (s("x"), OwnedItem::Float(-0.0)),
                    (
                        s("u"),
                        OwnedItem::Table(vec![(s("z"), OwnedItem::Bool(true))]),
                    ),
                ]),
            ),
            (s("b"), OwnedItem::Str(s("0.5"))),
            (s("e"), OwnedItem::Table(Vec::new())),
        ];
        assert_eq!(count(&entries), 7);
        assert_eq!(
            decode::<Echo>(&entries),
            Ok(Echo(
                "a=Int(-3),t={x=Float(-0.0),u={z=Bool(true)}},b=Str(\"0.5\"),e={}".to_owned()
            ))
        );
        assert_eq!(decode::<Echo>(&[]), Ok(Echo(String::new())));
    }
}
