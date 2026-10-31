// SPDX-License-Identifier: MIT OR Apache-2.0

//! Runtime behaviour of the derives.

use tessera_rl_core::config::{CanonicalConfig, canonical_bytes, config_hash};
use tessera_rl_core::encoding::{
    Canonical, Decoder, Encoder, Sink, decode_from_slice, decode_state_enum, encode_state_enum,
    encode_to_vec,
};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    Bool8, ConfigError, DecodeError, EnvConfig, ObsLeaf, Observation, Pool, Reserved, StateEnum,
    StateHash,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation, StateEnum};
use tessera_rl_num::Fx32;

#[derive(StateEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
enum Phase {
    #[default]
    Opening = 1,
    Endgame = 2,
}

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Toy {
    pos: [Fx32; 2],
    speed: f64,
    hp: i16,
    phase: Phase,
    alive: Bool8,
    pad: Reserved<4>,
    units: Pool<u16, 4>,
    tail: Reserved<4>,
}

/// The same layout encoded by hand, as in the core tests.
fn hand_encoded(t: &Toy) -> Vec<u8> {
    struct Hand<'a>(&'a Toy);
    impl Canonical for Hand<'_> {
        const ENCODED_LEN: usize = 40;
        fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
            let t = self.0;
            t.pos.encode(out);
            t.speed.encode(out);
            t.hp.encode(out);
            encode_state_enum(t.phase, out);
            t.alive.encode(out);
            t.pad.encode(out);
            t.units.encode(out);
            t.tail.encode(out);
        }
        fn decode(_: &mut Decoder<'_>) -> Result<Self, DecodeError> {
            Err(DecodeError::TrailingBytes { count: 0 })
        }
    }
    encode_to_vec(&Hand(t))
}

fn toy() -> Toy {
    let mut units = Pool::default();
    units.insert(7);
    Toy {
        pos: [Fx32::from_raw(65_536), Fx32::from_raw(-1)],
        speed: -0.0,
        hp: -3,
        phase: Phase::Endgame,
        alive: Bool8::TRUE,
        pad: Reserved::ZERO,
        units,
        tail: Reserved::ZERO,
    }
}

#[test]
fn test_env_state_derive_matches_hand_encoding() {
    let value = toy();
    assert_eq!(size_of::<Toy>(), Toy::ENCODED_LEN);
    assert_eq!(encode_to_vec(&value), hand_encoded(&value));
    assert_eq!(decode_from_slice::<Toy>(&encode_to_vec(&value)), Ok(value));
}

#[derive(EnvState, Clone, Copy)]
#[repr(C)]
struct Big {
    grid: [[u8; 64]; 2],
    field: [[f32; 40]; 40],
}

#[test]
fn test_env_state_large_nested_arrays_use_state_init() {
    use tessera_rl_core::StateInit;
    let big = Big::state_default();
    assert!(big.grid.iter().flatten().all(|&b| b == 0));
    let decoded = decode_from_slice::<Big>(&encode_to_vec(&big)).unwrap();
    assert_eq!(decoded.field[39][39].to_bits(), 0);
}

#[test]
fn test_env_state_derive_hash_is_stable() {
    // Pinned: a change here means the canonical encoding changed.
    assert_eq!(StateHash::of(&toy()).get(), 0x0e2b_6c53_7a31_0c60);
    let mut other = toy();
    other.hp = 4;
    assert_ne!(StateHash::of(&toy()), StateHash::of(&other));
}

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Noisy {
    #[state(allow_nan)]
    noise: [f32; 2],
    level: f32,
}

#[test]
fn test_env_state_allow_nan_field_accepts_nan_only_where_marked() {
    let nan = f32::from_bits(0x7fc0_0042);
    let ok = Noisy {
        noise: [nan, 1.0],
        level: 0.5,
    };
    let decoded = decode_from_slice::<Noisy>(&encode_to_vec(&ok)).unwrap();
    assert_eq!(decoded.noise[0].to_bits(), 0x7fc0_0042);
    let bad = Noisy { level: nan, ..ok };
    assert!(matches!(
        decode_from_slice::<Noisy>(&encode_to_vec(&bad)),
        Err(DecodeError::NanInState { .. })
    ));
}

#[test]
fn test_state_enum_with_discriminants_one_and_two() {
    assert_eq!(Phase::default(), Phase::Opening);
    assert_eq!(Phase::Opening.to_repr(), 1);
    assert_eq!(Phase::from_repr(2), Some(Phase::Endgame));
    for invalid in [0u8, 3, 255] {
        assert_eq!(Phase::from_repr(invalid), None);
        assert_eq!(
            decode_from_slice::<Phase>(&[invalid]),
            Err(DecodeError::InvalidEnum {
                name: "Phase",
                value: i64::from(invalid)
            })
        );
    }
    let mut input = Decoder::new(&[2]);
    assert_eq!(decode_state_enum::<Phase>(&mut input), Ok(Phase::Endgame));
}

#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct Physics {
    #[config(default = 9.8, range = 0.0..=100.0)]
    gravity: f64,
}

#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct GridConfig {
    #[config(default = 16, range = 4..=64, shape)]
    size: u16,
    #[config(default = 200, range = 1..=10_000)]
    max_steps: u16,
    #[config(default = 1, adaptive)]
    difficulty: u8,
    physics: Physics,
}

#[test]
fn test_env_config_defaults_and_ranges() {
    let cfg = GridConfig::default();
    assert_eq!((cfg.size, cfg.max_steps, cfg.difficulty), (16, 200, 1));
    assert_eq!(cfg.physics.gravity, 9.8);
    assert_eq!(cfg.validate(), Ok(()));
    let small = GridConfig {
        size: 2,
        ..GridConfig::default()
    };
    assert_eq!(
        small.validate(),
        Err(ConfigError {
            field: "size",
            reason: "must be in 4 ..= 64"
        })
    );
    let heavy = GridConfig {
        physics: Physics { gravity: f64::NAN },
        ..GridConfig::default()
    };
    assert_eq!(heavy.validate().map_err(|e| e.field), Err("gravity"));
}

#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct Noise {
    #[config(default = 0.5)]
    noise: f64,
}

#[test]
fn test_env_config_rejects_nan_floats_without_range() {
    let cfg = Noise { noise: f64::NAN };
    assert_eq!(
        cfg.validate(),
        Err(ConfigError {
            field: "noise",
            reason: "must not be NaN"
        })
    );
}

#[test]
fn test_env_config_canonical_form_sorted_without_adaptive() {
    assert_eq!(GridConfig::ADAPTIVE_FIELDS, ["difficulty"]);
    assert_eq!(GridConfig::STATIC_ENTRIES, 3);
    let bytes = canonical_bytes(&GridConfig::default());
    let text = String::from_utf8_lossy(&bytes);
    let positions: Vec<usize> = ["max_steps", "physics", "gravity", "size"]
        .iter()
        .map(|k| text.find(k).unwrap())
        .collect();
    assert!(positions.windows(2).all(|w| w[0] < w[1]));
    assert!(!text.contains("difficulty"));
}

#[test]
fn test_env_config_hash_ignores_adaptive_and_tracks_static_fields() {
    let base = GridConfig::default();
    let hash = |cfg: &GridConfig| config_hash("tessera/Grid", 1, cfg).unwrap();
    assert_eq!(
        hash(&base),
        hash(&GridConfig {
            difficulty: 9,
            ..base.clone()
        })
    );
    assert_ne!(
        hash(&base),
        hash(&GridConfig {
            size: 17,
            ..base.clone()
        })
    );
    assert_ne!(
        hash(&base),
        hash(&GridConfig {
            max_steps: 3,
            ..base.clone()
        })
    );
    let heavier = GridConfig {
        physics: Physics { gravity: 10.0 },
        ..base.clone()
    };
    assert_ne!(hash(&base), hash(&heavier));
}

#[derive(Observation)]
struct GridObs {
    #[space(box(low = 0, high = 255, dtype = u8, shape = |c: &GridConfig| [3, c.size, c.size]))]
    pixels: ObsLeaf<u8>,
    #[space(discrete(n = 4))]
    heading: ObsLeaf<i32>,
}

#[derive(Observation)]
struct Constant {
    #[space(multi_binary(shape = [2, 2]))]
    bits: ObsLeaf<Bool8>,
}

#[test]
fn test_observation_space_resolves_from_config() {
    let space = |size| {
        GridObs::space(&GridConfig {
            size,
            ..GridConfig::default()
        })
        .unwrap()
    };
    let Space::Dict(entries) = space(8) else {
        panic!("expected a dict")
    };
    assert_eq!(entries.len(), GridObs::LEAVES);
    assert_eq!(
        entries[0],
        (
            "pixels".to_string(),
            Space::Box {
                shape: vec![3, 8, 8],
                dtype: DType::U8,
                bounds: BoxBounds::UInt {
                    low: vec![0],
                    high: vec![255]
                },
            }
        )
    );
    assert_eq!(
        entries[1],
        ("heading".to_string(), Space::Discrete { n: 4 })
    );
    assert_ne!(space(8), space(16));
    assert_eq!(GridObs::PIXELS.index(), 0);
    assert_eq!(GridObs::HEADING.index(), 1);
}

#[test]
fn test_observation_without_closures_is_generic_over_config() {
    let a = <Constant as Observation<()>>::space(&()).unwrap();
    let b = <Constant as Observation<GridConfig>>::space(&GridConfig::default()).unwrap();
    assert_eq!(a, b);
    assert_eq!(
        a,
        Space::Dict(vec![(
            "bits".into(),
            Space::MultiBinary { shape: vec![2, 2] }
        )])
    );
}
