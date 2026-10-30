// SPDX-License-Identifier: MIT OR Apache-2.0

//! Canonical form and hash of configurations (ADR-0030).
//!
//! The canonical form is the format version byte followed by the static
//! (non-adaptive) entries of the config, sorted by key. Each entry is
//! `[key length: u16][key: UTF-8][type tag: u8][value]`, values
//! little-endian; floats are their exact bits, booleans 0 or 1, fixed point
//! its raw integer, and a nested config is `[tag][entry count: u16]` then
//! its own sorted entries. `#[derive(EnvConfig)]` generates the
//! implementation, sorting keys at compile time.
//!
//! The config hash is XXH3-64 of the format version, the environment id
//! (`[length: u16][UTF-8]`), the environment version (`u32`) and the
//! entries. Adaptive fields are excluded.

use tessera_rl_num::{Fx32, Fx64};

use crate::encoding::{Canonical, Encoder, HashSink, Sink};

/// Version of the canonical config format. A change to the format bumps
/// it, so old and new forms never collide silently.
pub const CONFIG_FORMAT_VERSION: u8 = 1;

/// A config field value with a canonical encoding.
pub trait ConfigValue {
    /// Type tag written before the value.
    const TAG: u8;

    /// Writes the value (without its tag).
    fn write_value<S: Sink>(&self, out: &mut Encoder<S>);
}

macro_rules! config_value {
    ($($ty:ty => $tag:expr),* $(,)?) => {$(
        impl ConfigValue for $ty {
            const TAG: u8 = $tag;

            #[inline]
            fn write_value<S: Sink>(&self, out: &mut Encoder<S>) {
                self.encode(out);
            }
        }
    )*};
}

config_value! {
    u8 => 1, u16 => 2, u32 => 3, u64 => 4,
    i8 => 5, i16 => 6, i32 => 7, i64 => 8,
    f32 => 9, f64 => 10,
    Fx32 => 12, Fx64 => 13,
}

impl ConfigValue for bool {
    const TAG: u8 = 11;

    #[inline]
    fn write_value<S: Sink>(&self, out: &mut Encoder<S>) {
        out.write_bytes(&[u8::from(*self)]);
    }
}

/// Tag of a nested config value.
pub const NESTED_TAG: u8 = 14;

/// A configuration with a canonical form.
pub trait CanonicalConfig {
    /// Names of the fields marked `#[config(adaptive)]`, excluded from the
    /// canonical form.
    const ADAPTIVE_FIELDS: &'static [&'static str];

    /// Number of static entries.
    const STATIC_ENTRIES: u16;

    /// Writes the static entries, sorted by key.
    fn write_entries<S: Sink>(&self, out: &mut Encoder<S>);
}

/// Writes one entry: key, tag and value. Used by generated code.
#[inline]
pub fn write_entry<S: Sink, V: ConfigValue>(out: &mut Encoder<S>, key: &str, value: &V) {
    write_key(out, key);
    out.write_bytes(&[V::TAG]);
    value.write_value(out);
}

/// Writes one nested-config entry. Used by generated code.
#[inline]
pub fn write_nested_entry<S: Sink, C: CanonicalConfig>(out: &mut Encoder<S>, key: &str, value: &C) {
    write_key(out, key);
    out.write_bytes(&[NESTED_TAG]);
    C::STATIC_ENTRIES.encode(out);
    value.write_entries(out);
}

#[inline]
fn write_key<S: Sink>(out: &mut Encoder<S>, key: &str) {
    // Keys are field names; the derive rejects names longer than u16::MAX.
    let len = u16::try_from(key.len()).unwrap_or(u16::MAX);
    len.encode(out);
    out.write_bytes(key.as_bytes());
}

/// Writes the canonical form of `cfg`: the format version, then its static
/// entries.
#[inline]
pub fn write_canonical<S: Sink, C: CanonicalConfig>(out: &mut Encoder<S>, cfg: &C) {
    out.write_bytes(&[CONFIG_FORMAT_VERSION]);
    cfg.write_entries(out);
}

/// The canonical form of `cfg` as bytes. For persistence and replays.
#[must_use]
pub fn canonical_bytes<C: CanonicalConfig>(cfg: &C) -> Vec<u8> {
    let mut out = Encoder::new(Vec::new());
    write_canonical(&mut out, cfg);
    out.into_sink()
}

/// The config hash: XXH3-64 of the format version, environment id and
/// version, and the static entries of `cfg`. Never allocates.
#[must_use]
pub fn config_hash<C: CanonicalConfig>(env_id: &str, env_version: u32, cfg: &C) -> u64 {
    let mut out = Encoder::new(HashSink::new());
    out.write_bytes(&[CONFIG_FORMAT_VERSION]);
    write_key(&mut out, env_id);
    env_version.encode(&mut out);
    cfg.write_entries(&mut out);
    out.sink().digest()
}
