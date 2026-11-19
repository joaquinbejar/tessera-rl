// SPDX-License-Identifier: MIT OR Apache-2.0

//! Python keyword arguments as a config source (ADR-0065 item 3): the
//! Python front end of the shared config decoder.
//!
//! Each value maps to one [`ConfigItem`]: `bool` to `Bool` (checked before
//! `int`, of which it is a subclass), an exact `int` to `Int`, an exact
//! `float` to `Float` (its bits as given), an exact `str` to `Str`, and an
//! exact `dict` to `Table`, recursively. Only exact built-in types are
//! taken: no subclasses and no NumPy scalars. Anything else, an `int`
//! beyond 64 bits, and more than [`MAX_ENTRIES`] pairs are refused
//! ([`KwargsError`]), after the key names are checked: an unknown key is
//! reported first ([`decode_kwargs`]). Tables deeper than
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

/// Most `(key, value)` pairs the keyword arguments may hold, nested tables
/// included: a bound on what one `make` call reads and allocates.
pub const MAX_ENTRIES: usize = 4096;

/// Why keyword arguments are not a config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KwargsError {
    /// A value of a Python type no config field takes (only exact `bool`,
    /// `int`, `float`, `str` and `dict` are taken: no subclasses, no NumPy
    /// scalars).
    Type {
        /// The key, dotted for nested tables (`physics.gravity`).
        key: String,
    },
    /// An `int` beyond 64 bits.
    Int {
        /// The key, dotted for nested tables.
        key: String,
    },
    /// More than [`MAX_ENTRIES`] pairs.
    TooMany,
}

impl core::fmt::Display for KwargsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Type { key } => write!(
                f,
                "config key `{key}`: a value must be a bool, int, float, str or dict"
            ),
            Self::Int { key } => write!(f, "config key `{key}`: an int beyond 64 bits"),
            Self::TooMany => write!(f, "more than {MAX_ENTRIES} config entries"),
        }
    }
}

impl std::error::Error for KwargsError {}

/// Keyword arguments read for decoding. A value no field takes is kept as a
/// placeholder and recorded in `refused`, so the decoder still checks every
/// key name first: an unknown key is reported before a value's type
/// (SPEC 14.4.1 item 4).
#[derive(Debug, Clone, PartialEq)]
pub struct Kwargs {
    /// The pairs, in order.
    pub entries: Vec<(String, OwnedItem)>,
    /// The first value no field takes, if any.
    pub refused: Option<KwargsError>,
}

/// Reads keyword arguments (a dict's items).
///
/// # Errors
///
/// A Python exception (`Err` of the outer `CResult`) when reading a value
/// raised; [`KwargsError::TooMany`] past [`MAX_ENTRIES`].
pub fn read<'py>(
    g: Gil<'py>,
    entries: Vec<(String, Mine<'py>)>,
) -> CResult<Result<Kwargs, KwargsError>> {
    let mut reader = Reader {
        refused: None,
        budget: MAX_ENTRIES,
    };
    let entries = match reader.table(g, entries, "", 1)? {
        Ok(entries) => entries,
        Err(e) => return Ok(Err(e)),
    };
    Ok(Ok(Kwargs {
        entries,
        refused: reader.refused,
    }))
}

struct Reader {
    refused: Option<KwargsError>,
    budget: usize,
}

impl Reader {
    fn table<'py>(
        &mut self,
        g: Gil<'py>,
        entries: Vec<(String, Mine<'py>)>,
        prefix: &str,
        depth: u32,
    ) -> CResult<Result<Vec<(String, OwnedItem)>, KwargsError>> {
        let mut out = Vec::with_capacity(entries.len());
        for (key, value) in entries {
            let Some(left) = self.budget.checked_sub(1) else {
                return Ok(Err(KwargsError::TooMany));
            };
            self.budget = left;
            let dotted = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{prefix}.{key}")
            };
            let item = match self.item(g, &dotted, value.as_ref(), depth)? {
                Ok(item) => item,
                Err(e) => return Ok(Err(e)),
            };
            out.push((key, item));
        }
        Ok(Ok(out))
    }

    /// A placeholder for a refused value: any item does, as decoding fails
    /// with the recorded error whenever every key name is known.
    fn refuse(&mut self, e: KwargsError) -> OwnedItem {
        self.refused.get_or_insert(e);
        OwnedItem::Bool(false)
    }

    fn item(
        &mut self,
        g: Gil<'_>,
        key: &str,
        v: Ref<'_>,
        depth: u32,
    ) -> CResult<Result<OwnedItem, KwargsError>> {
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
                        self.refuse(KwargsError::Int {
                            key: key.to_owned(),
                        })
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
                    match self.table(g, entries, key, depth.saturating_add(1))? {
                        Ok(t) => OwnedItem::Table(t),
                        Err(e) => return Ok(Err(e)),
                    }
                }
            }
            Kind::None | Kind::Other => self.refuse(KwargsError::Type {
                key: key.to_owned(),
            }),
        }))
    }
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

/// Why keyword arguments do not decode into a config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The decoder's error.
    Decode(ConfigDecodeError),
    /// A value no field takes, or too many entries.
    Kwargs(KwargsError),
}

/// Decodes `C` from keyword arguments read by [`read`]: key names first,
/// then values. An unknown top-level key is reported before a refused
/// value anywhere, and a refused value before any other decoding error.
/// An unknown key inside a nested table is found only when the decoder
/// reaches that table, so a refused value in an earlier field is reported
/// first; it is named without its table's path (the decoder's error
/// carries the key only).
///
/// # Errors
///
/// The [`Refusal`], by that precedence.
pub fn decode_kwargs<C: ConfigDecode>(kwargs: &Kwargs) -> Result<C, Refusal> {
    match (decode::<C>(&kwargs.entries), &kwargs.refused) {
        (Err(e @ ConfigDecodeError::UnknownKey { .. }), _) => Err(Refusal::Decode(e)),
        (_, Some(refused)) => Err(Refusal::Kwargs(refused.clone())),
        (result, None) => result.map_err(Refusal::Decode),
    }
}

/// Raises the Python error of a [`Refusal`].
#[must_use]
pub fn raise_refusal(g: Gil<'_>, r: &Refusal) -> capi::Raised {
    match r {
        Refusal::Decode(e) => raise_decode(g, e),
        Refusal::Kwargs(e) => raise_kwargs(g, e),
    }
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
