// SPDX-License-Identifier: MIT OR Apache-2.0

//! The decoder `#[derive(EnvConfig)]` generates (ADR-0065 item 3).

use tessera_rl_core::{ConfigDecode, ConfigDecodeError, ConfigError, ConfigItem, ConfigSource};
use tessera_rl_macros::EnvConfig;
use tessera_rl_num::Fx32;

#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct Physics {
    #[config(default = 9.8, range = 0.0..=100.0)]
    gravity: f64,
}

#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct Game {
    #[config(default = 16, range = 4..=64)]
    size: u16,
    #[config(default = 0.5)]
    noise: f32,
    /// No `default`: the type's default, 0.
    seed: u32,
    #[config(default = true)]
    hints: bool,
    step: Fx32,
    #[config(default = 1, adaptive)]
    level: u8,
    physics: Physics,
}

#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct Empty {}

fn decode<C: ConfigDecode>(entries: &[(&str, ConfigItem<'_>)]) -> Result<C, ConfigDecodeError> {
    C::decode(&ConfigSource::new(entries))
}

#[test]
fn test_config_decode_empty_source_is_the_default() {
    assert_eq!(Game::decode(&ConfigSource::empty()), Ok(Game::default()));
    assert_eq!(Game::default().seed, 0);
    assert_eq!(Empty::decode(&ConfigSource::empty()), Ok(Empty {}));
}

#[test]
fn test_config_decode_partial_source_overwrites_only_the_given_fields() {
    let physics = [("gravity", ConfigItem::Float(3.5))];
    let cfg: Game = decode(&[
        ("seed", ConfigItem::Int(7)),
        ("noise", ConfigItem::Float(0.1)),
        ("step", ConfigItem::Str("0.25")),
        ("physics", ConfigItem::Table(ConfigSource::new(&physics))),
    ])
    .unwrap_or_default();
    let want = Game {
        seed: 7,
        noise: 0.1,
        step: Fx32::from_decimal("0.25").unwrap_or_default(),
        physics: Physics { gravity: 3.5 },
        ..Game::default()
    };
    assert_eq!(cfg, want);
}

#[test]
fn test_config_decode_refuses_unknown_repeated_and_adaptive_keys() {
    assert_eq!(
        decode::<Game>(&[("colour", ConfigItem::Int(1))]),
        Err(ConfigDecodeError::UnknownKey {
            key: "colour".to_owned()
        })
    );
    assert_eq!(
        decode::<Game>(&[("seed", ConfigItem::Int(1)), ("seed", ConfigItem::Int(2))]),
        Err(ConfigDecodeError::DuplicateKey {
            key: "seed".to_owned()
        })
    );
    assert_eq!(
        decode::<Game>(&[("level", ConfigItem::Int(2))]),
        Err(ConfigDecodeError::UnknownKey {
            key: "level".to_owned()
        })
    );
    let inner = [("weight", ConfigItem::Int(1))];
    assert_eq!(
        decode::<Game>(&[("physics", ConfigItem::Table(ConfigSource::new(&inner)))]),
        Err(ConfigDecodeError::UnknownKey {
            key: "weight".to_owned()
        })
    );
}

#[test]
fn test_config_decode_field_errors_name_the_field() {
    assert_eq!(
        decode::<Game>(&[("size", ConfigItem::Int(70_000))]),
        Err(ConfigDecodeError::OutOfRange { field: "size" })
    );
    assert_eq!(
        decode::<Game>(&[("physics", ConfigItem::Int(1))]),
        Err(ConfigDecodeError::WrongType {
            field: "physics",
            expected: "a table"
        })
    );
    assert_eq!(
        decode::<Game>(&[("hints", ConfigItem::Int(1))]),
        Err(ConfigDecodeError::WrongType {
            field: "hints",
            expected: "a bool"
        })
    );
}

#[test]
fn test_config_decode_validate_failures_are_invalid() {
    // In u16's range, outside the declared 4..=64.
    let size = decode::<Game>(&[("size", ConfigItem::Int(65))]);
    assert!(
        matches!(
            size,
            Err(ConfigDecodeError::Invalid(ConfigError {
                field: "size",
                ..
            }))
        ),
        "{size:?}"
    );
    let nan = decode::<Game>(&[("noise", ConfigItem::Float(f64::NAN))]);
    assert!(
        matches!(
            nan,
            Err(ConfigDecodeError::Invalid(ConfigError {
                field: "noise",
                ..
            }))
        ),
        "{nan:?}"
    );
}

macro_rules! nest {
    ($outer:ident => $inner:ident) => {
        #[derive(EnvConfig, Debug, Clone, PartialEq)]
        struct $outer {
            next: $inner,
        }
    };
}

#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct L9 {
    x: u8,
}
nest!(L8 => L9);
nest!(L7 => L8);
nest!(L6 => L7);
nest!(L5 => L6);
nest!(L4 => L5);
nest!(L3 => L4);
nest!(L2 => L3);
nest!(L1 => L2);

#[test]
fn test_config_decode_depth_is_bounded_at_eight_tables() {
    // Eight tables, the top level included: L2 down to L9.
    let t9 = [("x", ConfigItem::Int(3))];
    let t8 = [("next", ConfigItem::Table(ConfigSource::new(&t9)))];
    let t7 = [("next", ConfigItem::Table(ConfigSource::new(&t8)))];
    let t6 = [("next", ConfigItem::Table(ConfigSource::new(&t7)))];
    let t5 = [("next", ConfigItem::Table(ConfigSource::new(&t6)))];
    let t4 = [("next", ConfigItem::Table(ConfigSource::new(&t5)))];
    let t3 = [("next", ConfigItem::Table(ConfigSource::new(&t4)))];
    let t2 = [("next", ConfigItem::Table(ConfigSource::new(&t3)))];
    let deep = decode::<L2>(&t2);
    assert_eq!(deep.map(|c| c.next.next.next.next.next.next.next.x), Ok(3));
    // Nine: L1 down to L9.
    let t1 = [("next", ConfigItem::Table(ConfigSource::new(&t2)))];
    assert_eq!(decode::<L1>(&t1), Err(ConfigDecodeError::TooDeep));
}
