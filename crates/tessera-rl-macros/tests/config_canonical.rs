// SPDX-License-Identifier: MIT OR Apache-2.0

//! The canonical decoder `#[derive(EnvConfig)]` generates (#377): the
//! inverse of the canonical form (ADR-0030), for replays that rebuild their
//! pool from the header (ADR-0072).

use tessera_rl_core::config::canonical_bytes;
use tessera_rl_core::{CanonicalDecodeError, ConfigError, decode_canonical};
use tessera_rl_macros::EnvConfig;
use tessera_rl_num::{Fx32, Fx64};

#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct Physics {
    #[config(default = 9.8, range = 0.0..=100.0)]
    gravity: f64,
    #[config(default = -3)]
    floor: i16,
}

#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct Game {
    #[config(default = 16, range = 4..=64)]
    size: u16,
    #[config(default = 0.5)]
    noise: f32,
    seed: u64,
    #[config(default = true)]
    hints: bool,
    step: Fx32,
    wide: Fx64,
    tiny: u8,
    small: i8,
    word: u32,
    signed: i32,
    big: i64,
    #[config(default = 1, adaptive)]
    level: u8,
    physics: Physics,
}

#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct Empty {}

fn game() -> Game {
    Game {
        size: 33,
        noise: -0.0,
        seed: u64::MAX,
        hints: false,
        step: Fx32::from_decimal("0.25").unwrap_or_default(),
        wide: Fx64::from_decimal("-1.5").unwrap_or_default(),
        tiny: 255,
        small: -128,
        word: 7,
        signed: -7,
        big: i64::MIN,
        level: 9,
        physics: Physics {
            gravity: 1.25,
            floor: 4,
        },
    }
}

#[test]
fn test_config_canonical_round_trip_every_type_and_nesting() {
    let cfg = game();
    let bytes = canonical_bytes(&cfg);
    let back: Game = decode_canonical(&bytes).unwrap_or_else(|e| panic!("{e}"));
    // The adaptive field is not in the canonical form: it keeps its default.
    assert_eq!(back, Game { level: 1, ..cfg });
    assert_eq!(canonical_bytes(&back), bytes);
    // `-0.0` keeps its sign: floats are their bits.
    assert!(back.noise.is_sign_negative());
}

#[test]
fn test_config_canonical_empty_config_round_trips() {
    let bytes = canonical_bytes(&Empty {});
    assert_eq!(decode_canonical::<Empty>(&bytes), Ok(Empty {}));
}

#[test]
fn test_config_canonical_refuses_version_trailing_and_truncation() {
    let mut bytes = canonical_bytes(&game());
    let mut other = bytes.clone();
    other[0] = 2;
    assert_eq!(
        decode_canonical::<Game>(&other),
        Err(CanonicalDecodeError::Version { found: 2 })
    );
    bytes.push(0);
    assert!(matches!(
        decode_canonical::<Game>(&bytes),
        Err(CanonicalDecodeError::Decode(_))
    ));
    bytes.truncate(bytes.len() - 6);
    assert!(matches!(
        decode_canonical::<Game>(&bytes),
        Err(CanonicalDecodeError::Decode(_))
    ));
}

/// The first entry in key order is `big` (an `i64`, tag 8): its key at
/// byte 1 (`u16` length 3), its tag at byte 6.
#[test]
fn test_config_canonical_refuses_another_key_or_tag() {
    let bytes = canonical_bytes(&game());
    assert_eq!(&bytes[1..6], &[3, 0, b'b', b'i', b'g']);
    let mut key = bytes.clone();
    key[5] = b'x';
    assert_eq!(
        decode_canonical::<Game>(&key),
        Err(CanonicalDecodeError::Key { expected: "big" })
    );
    // A key length that differs is refused at the length, even when the
    // bytes that follow start with the expected key.
    let mut length = bytes.clone();
    length[1] = 4;
    assert_eq!(
        decode_canonical::<Game>(&length),
        Err(CanonicalDecodeError::Key { expected: "big" })
    );
    let mut tag = bytes;
    tag[6] = 7;
    assert_eq!(
        decode_canonical::<Game>(&tag),
        Err(CanonicalDecodeError::Tag {
            key: "big",
            found: 7,
            expected: 8
        })
    );
}

#[test]
fn test_config_canonical_refuses_a_bool_byte_and_a_nested_count() {
    let cfg = game();
    let bytes = canonical_bytes(&cfg);
    // `hints` (tag 11) is followed by its byte.
    let at = bytes
        .windows(7)
        .position(|w| w == [5, 0, b'h', b'i', b'n', b't', b's'])
        .unwrap_or(0)
        + 8;
    let mut bool_byte = bytes.clone();
    bool_byte[at] = 2;
    assert_eq!(
        decode_canonical::<Game>(&bool_byte),
        Err(CanonicalDecodeError::Bool {
            key: "hints",
            byte: 2
        })
    );
    // `physics` (tag 14) is followed by its `u16` entry count, 2.
    let at = bytes
        .windows(9)
        .position(|w| w == [7, 0, b'p', b'h', b'y', b's', b'i', b'c', b's'])
        .unwrap_or(0)
        + 10;
    assert_eq!(&bytes[at..at + 2], &[2, 0]);
    let mut count = bytes;
    count[at] = 3;
    assert_eq!(
        decode_canonical::<Game>(&count),
        Err(CanonicalDecodeError::NestedCount {
            key: "physics",
            found: 3,
            expected: 2
        })
    );
}

#[test]
fn test_config_canonical_runs_validate() {
    let cfg = Game { size: 2, ..game() };
    let bytes = canonical_bytes(&cfg);
    assert_eq!(
        decode_canonical::<Game>(&bytes),
        Err(CanonicalDecodeError::Invalid(ConfigError {
            field: "size",
            reason: "must be in 4 ..= 64"
        }))
    );
}
