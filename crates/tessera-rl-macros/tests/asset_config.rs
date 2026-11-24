// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Asset<T>` config fields (ADR-0077): canonical form, asset list, names,
//! bundled defaults and decoding from a name or bytes.

use std::borrow::Cow;

use tessera_rl_core::config::{asset_list, canonical_bytes, config_hash};
use tessera_rl_core::{
    Asset, AssetData, AssetError, ConfigDecode, ConfigDecodeError, ConfigItem, ConfigSource,
};
use tessera_rl_macros::EnvConfig;

/// A list of bytes; text input is comma-separated decimals.
#[derive(Debug, Default, PartialEq)]
struct Table(Vec<u8>);

impl AssetData for Table {
    const MAX_DECODED_BYTES: usize = 16;

    fn decode(bytes: &[u8]) -> Result<Self, AssetError> {
        if let Some((&0xff, rest)) = bytes.split_first() {
            return Ok(Self(rest.to_vec()));
        }
        let text =
            core::str::from_utf8(bytes).map_err(|_| AssetError::Invalid { reason: "utf-8" })?;
        text.split(',')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(|p| p.parse().map_err(|_| AssetError::Invalid { reason: "u8" }))
            .collect::<Result<_, _>>()
            .map(Self)
    }

    fn canonical_bytes(&self) -> Cow<'_, [u8]> {
        Cow::Owned([&[0xff][..], &self.0].concat())
    }

    fn bundled(name: &str) -> Option<&'static [u8]> {
        (name == "small").then_some(b"1,2")
    }
}

#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct Inner {
    #[config(asset = "inner.units", default = "small")]
    units: Asset<Table>,
}

#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct Outer {
    #[config(default = 3)]
    size: u16,
    map: Asset<Table>,
    #[config(asset = "tiles")]
    tile_set: Asset<Table>,
    inner: Inner,
}

#[test]
fn test_asset_config_defaults_bundled_and_type_default() {
    let c = Outer::default();
    assert_eq!(c.inner.units.get(), &Table(vec![1, 2]));
    assert_eq!(c.map.get(), &Table::default());
}

#[test]
fn test_asset_config_list_uses_names_sorted_with_nested() {
    let c = Outer::default();
    let names: Vec<_> = asset_list(&c).into_iter().map(|e| e.name).collect();
    assert_eq!(names, ["inner.units", "map", "tiles"]);
    let list = asset_list(&c);
    assert_eq!(list[1].hash, c.map.hash());
}

#[test]
fn test_asset_config_hash_follows_the_asset_not_its_spelling() {
    let base = Outer::default();
    let same = Outer {
        map: Asset::from_bytes(b" ").unwrap(),
        ..Outer::default()
    };
    assert_eq!(canonical_bytes(&base), canonical_bytes(&same));
    let other = Outer {
        map: Asset::from_bytes(b"9").unwrap(),
        ..Outer::default()
    };
    assert_ne!(
        config_hash("t/X-v1", 1, &base).unwrap(),
        config_hash("t/X-v1", 1, &other).unwrap()
    );
    let canonical = canonical_bytes(&other);
    let entry = [&[15u8][..], &other.map.hash().0].concat();
    assert!(canonical.windows(17).any(|w| w == entry.as_slice()));
}

#[test]
fn test_asset_config_decodes_bundled_name_and_bytes() {
    let src = [("map", ConfigItem::Bytes(b"5,6"))];
    let c = Outer::decode(&ConfigSource::new(&src)).unwrap();
    assert_eq!(c.map.get(), &Table(vec![5, 6]));
    // The source key is the field name; `asset = ...` names it in headers
    // only.
    let src = [("tiles", ConfigItem::Bytes(&[0xff, 7]))];
    assert!(matches!(
        Outer::decode(&ConfigSource::new(&src)),
        Err(ConfigDecodeError::UnknownKey { .. })
    ));
    let src = [("tile_set", ConfigItem::Bytes(&[0xff, 7]))];
    let c = Outer::decode(&ConfigSource::new(&src)).unwrap();
    assert_eq!(c.tile_set.get(), &Table(vec![7]));
    let src = [("map", ConfigItem::Str("small"))];
    assert_eq!(
        Outer::decode(&ConfigSource::new(&src)).unwrap().map.get(),
        &Table(vec![1, 2])
    );
}

#[test]
fn test_asset_config_decode_refusals_name_the_field() {
    let src = [("map", ConfigItem::Int(3))];
    assert!(matches!(
        Outer::decode(&ConfigSource::new(&src)),
        Err(ConfigDecodeError::WrongType { field: "map", .. })
    ));
    let src = [("map", ConfigItem::Str("nope"))];
    assert!(matches!(
        Outer::decode(&ConfigSource::new(&src)),
        Err(ConfigDecodeError::Asset {
            field: "map",
            source: AssetError::UnknownBundled { .. }
        })
    ));
    let src = [("map", ConfigItem::Bytes(b"x"))];
    assert!(matches!(
        Outer::decode(&ConfigSource::new(&src)),
        Err(ConfigDecodeError::Asset {
            field: "map",
            source: AssetError::Invalid { .. }
        })
    ));
}

#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct OnlyMap {
    map: Asset<Table>,
}

#[test]
fn test_asset_config_canonical_decode_rebuilds_only_the_empty_asset() {
    use tessera_rl_core::config_canonical::{CanonicalDecodeError, decode_canonical};
    let plain = OnlyMap::default();
    assert_eq!(
        decode_canonical::<OnlyMap>(&canonical_bytes(&plain)),
        Ok(plain)
    );
    let filled = OnlyMap {
        map: Asset::from_bytes(b"4").unwrap(),
    };
    assert_eq!(
        decode_canonical::<OnlyMap>(&canonical_bytes(&filled)),
        Err(CanonicalDecodeError::Asset { key: "map" })
    );
    // A nested bundled default is not the empty asset either.
    assert!(matches!(
        decode_canonical::<Outer>(&canonical_bytes(&Outer::default())),
        Err(CanonicalDecodeError::Asset { key: "units" })
    ));
}
