// SPDX-License-Identifier: MIT OR Apache-2.0

//! The canonical config encoding, version 1 (`CONFIG_FORMAT_VERSION`),
//! against its committed fixture `tests/fixtures/formats/config-v1.bin`: the
//! canonical bytes of a config with every value type, an asset and a nested
//! config, and that config's `config_hash`. The bytes decode back to the
//! config, today's encoder writes the same bytes and hash, and a newer or
//! another-major version byte is refused with the typed error naming it.
//! A changed encoding is resolved by a new version that keeps this file and
//! its read-back test, or by a fix; the fixture is regenerated only by
//! `test_format_config_v1_write_fixture` with `TESSERA_BLESS_FORMATS=1`.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use std::borrow::Cow;

use tessera_rl_core::config::{CONFIG_FORMAT_VERSION, canonical_bytes, config_hash};
use tessera_rl_core::{Asset, AssetData, AssetError, CanonicalDecodeError, decode_canonical};
use tessera_rl_macros::EnvConfig;
use tessera_rl_num::{Fx32, Fx64};

const FIXTURE: &[u8] = include_bytes!("fixtures/formats/config-v1.bin");
/// The `config_hash` of the fixture's config for `ENV_ID` and `ENV_VERSION`,
/// as a decimal line.
const FIXTURE_HASH: &str = include_str!("fixtures/formats/config-v1.hash");
const ENV_ID: &str = "test/Formats-v1";
const ENV_VERSION: u32 = 3;

/// An asset whose canonical bytes are its contents. The canonical form
/// holds only an asset's hash, and decodes back only the asset of empty
/// input (ADR-0077), so the fixture's asset is the empty one.
#[derive(Debug, Default, PartialEq)]
struct Blob(Vec<u8>);

impl AssetData for Blob {
    const MAX_DECODED_BYTES: usize = 64;

    fn decode(bytes: &[u8]) -> Result<Self, AssetError> {
        Ok(Self(bytes.to_vec()))
    }

    fn canonical_bytes(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.0)
    }

    fn bundled(_: &str) -> Option<&'static [u8]> {
        None
    }
}

#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct Physics {
    #[config(default = 9.8, range = 0.0..=100.0)]
    gravity: f64,
    #[config(default = -3)]
    floor: i16,
}

/// Every value type of the canonical form: the unsigned and signed
/// integers, both floats, bool, both fixed-point types, an asset and a
/// nested config.
#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct Formats {
    tiny: u8,
    #[config(default = 16, range = 4..=64)]
    size: u16,
    word: u32,
    seed: u64,
    small: i8,
    #[config(default = -2)]
    short: i16,
    signed: i32,
    big: i64,
    #[config(default = 0.5)]
    noise: f32,
    wide_float: f64,
    #[config(default = true)]
    hints: bool,
    step: Fx32,
    wide: Fx64,
    map: Asset<Blob>,
    physics: Physics,
}

fn config() -> Formats {
    Formats {
        tiny: 255,
        size: 33,
        word: 7,
        seed: u64::MAX,
        small: -128,
        short: -300,
        signed: -7,
        big: i64::MIN,
        noise: -0.0,
        wide_float: 1.0e-300,
        hints: false,
        step: Fx32::from_decimal("0.25").unwrap_or_default(),
        wide: Fx64::from_decimal("-1.5").unwrap_or_default(),
        map: Asset::<Blob>::from_bytes(b"").unwrap(),
        physics: Physics {
            gravity: 1.25,
            floor: 4,
        },
    }
}

#[test]
#[ignore = "writes the fixture; run with TESSERA_BLESS_FORMATS=1"]
fn test_format_config_v1_write_fixture() {
    if std::env::var("TESSERA_BLESS_FORMATS").as_deref() != Ok("1") {
        return;
    }
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/formats");
    std::fs::write(format!("{dir}/config-v1.bin"), canonical_bytes(&config())).unwrap();
    let hash = config_hash(ENV_ID, ENV_VERSION, &config()).unwrap();
    std::fs::write(format!("{dir}/config-v1.hash"), format!("{hash}\n")).unwrap();
}

#[test]
fn test_format_config_v1_fixture_reads_back() {
    assert_eq!(FIXTURE[0], 1, "the version byte");
    let decoded: Formats = decode_canonical(FIXTURE).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(decoded, config());
    assert!(decoded.noise.is_sign_negative());
}

#[test]
fn test_format_config_v1_encoder_writes_the_fixture_and_hash() {
    assert_eq!(CONFIG_FORMAT_VERSION, 1);
    assert_eq!(canonical_bytes(&config()), FIXTURE);
    let hash = config_hash(ENV_ID, ENV_VERSION, &config()).unwrap();
    assert_eq!(FIXTURE_HASH.trim(), hash.to_string());
}

#[test]
fn test_format_config_newer_and_other_major_versions_are_refused() {
    for found in [2u8, 0x80, 0xff] {
        let mut bytes = FIXTURE.to_vec();
        bytes[0] = found;
        assert_eq!(
            decode_canonical::<Formats>(&bytes),
            Err(CanonicalDecodeError::Version { found })
        );
    }
    let message = CanonicalDecodeError::Version { found: 2 }.to_string();
    assert!(
        message.contains('2') && message.contains("version"),
        "{message}"
    );
}
