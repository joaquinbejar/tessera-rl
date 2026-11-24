// SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;

use super::{
    Asset, AssetData, AssetError, AssetHash, MAX_ASSET_INPUT, bundled_or_default,
    check_asset_fixpoint,
};

/// A list of small integers: input is decimal numbers separated by commas
/// and optional spaces; the canonical form is `u8` bytes.
#[derive(Debug, Default, PartialEq)]
struct Toy(Vec<u8>);

impl AssetData for Toy {
    const MAX_DECODED_BYTES: usize = 8;

    fn decode(bytes: &[u8]) -> Result<Self, AssetError> {
        if bytes.first() == Some(&0xff) {
            // Canonical form: a marker byte, then the values.
            let values = bytes.get(1..).unwrap_or_default();
            if values.len() > Self::MAX_DECODED_BYTES {
                return Err(AssetError::DecodedTooLarge {
                    limit: Self::MAX_DECODED_BYTES,
                });
            }
            return Ok(Self(values.to_vec()));
        }
        let text = core::str::from_utf8(bytes).map_err(|_| AssetError::Invalid {
            reason: "not UTF-8",
        })?;
        let mut out = Vec::new();
        for part in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            if out.len() == Self::MAX_DECODED_BYTES {
                return Err(AssetError::DecodedTooLarge {
                    limit: Self::MAX_DECODED_BYTES,
                });
            }
            out.push(
                part.parse()
                    .map_err(|_| AssetError::Invalid { reason: "not a u8" })?,
            );
        }
        Ok(Self(out))
    }

    fn canonical_bytes(&self) -> Cow<'_, [u8]> {
        let mut v = vec![0xff];
        v.extend_from_slice(&self.0);
        Cow::Owned(v)
    }

    fn bundled(name: &str) -> Option<&'static [u8]> {
        match name {
            "three" => Some(b"1,2,3"),
            "broken" => Some(b"x"),
            _ => None,
        }
    }
}

#[test]
fn test_asset_non_canonical_inputs_hash_equal() {
    let a = Asset::<Toy>::from_bytes(b"1,2,3").unwrap();
    let b = Asset::<Toy>::from_bytes(b" 1, 2 ,3 ").unwrap();
    let c = Asset::<Toy>::from_bytes(&[0xff, 1, 2, 3]).unwrap();
    assert_eq!(a.hash(), b.hash());
    assert_eq!(a.hash(), c.hash());
    assert_eq!(a, b);
    assert_eq!(a.get(), &Toy(vec![1, 2, 3]));
}

#[test]
fn test_asset_hash_is_xxh3_128_of_canonical_bytes() {
    let a = Asset::new(Toy(vec![7]));
    assert_eq!(a.hash(), AssetHash::of(&[0xff, 7]));
    assert_eq!(
        a.hash().0,
        xxhash_rust::xxh3::xxh3_128(&[0xff, 7]).to_le_bytes()
    );
    assert_ne!(a.hash(), Asset::new(Toy(vec![8])).hash());
}

#[test]
fn test_asset_input_over_cap_refused() {
    let big = vec![b' '; MAX_ASSET_INPUT + 1];
    assert_eq!(
        Asset::<Toy>::from_bytes(&big).unwrap_err(),
        AssetError::InputTooLarge {
            len: MAX_ASSET_INPUT + 1
        }
    );
    // At the cap the input reaches decode.
    let at_cap = vec![b' '; MAX_ASSET_INPUT];
    assert!(Asset::<Toy>::from_bytes(&at_cap).is_ok());
}

#[test]
fn test_asset_decoded_over_limit_refused() {
    assert_eq!(
        Asset::<Toy>::from_bytes(b"1,2,3,4,5,6,7,8,9").unwrap_err(),
        AssetError::DecodedTooLarge { limit: 8 }
    );
}

#[test]
fn test_asset_bundled_known_unknown_and_default() {
    assert_eq!(
        Asset::<Toy>::bundled("three").unwrap().get(),
        &Toy(vec![1, 2, 3])
    );
    assert_eq!(
        Asset::<Toy>::bundled("nope").unwrap_err(),
        AssetError::UnknownBundled {
            name: "nope".to_owned()
        }
    );
    assert_eq!(
        bundled_or_default::<Toy>("three").get(),
        &Toy(vec![1, 2, 3])
    );
    assert_eq!(bundled_or_default::<Toy>("broken").get(), &Toy::default());
}

#[test]
fn test_asset_fixpoint_helper_accepts_and_rejects() {
    assert_eq!(check_asset_fixpoint::<Toy>(b"4, 5"), Ok(()));
    assert!(check_asset_fixpoint::<Toy>(b"x").is_err());
}

#[test]
fn test_asset_clone_shares_value() {
    let a = Asset::new(Toy(vec![1]));
    let b = a.clone();
    assert!(core::ptr::eq(a.get(), b.get()));
}
