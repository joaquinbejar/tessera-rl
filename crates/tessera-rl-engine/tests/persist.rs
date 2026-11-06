// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persisted snapshots (ADR-0011): the header, its checks and their order, on a
//! conformance environment with every field policy.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::encoding::{checksum, encode_to_vec};
use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Bool8, Environment, InvariantViolation, NoAdaptive, Observation,
    Persist, Pool, RecordHeader, Reserved, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodeLimits, DecodedRecord, EngineError, EnvPool, FORMAT_VERSION,
    IllegalActionPolicy, MAGIC, Masked, NoMasks, PersistError, PoolOptions, Reseed,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation, StateEnum};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

#[derive(StateEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
enum Kind {
    Zero = 0,
    #[default]
    Two = 2,
}

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Inner {
    #[state(allow_nan)]
    free: f64,
}

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Conf {
    strict: f64,
    inner: Inner,
    pool: Pool<f32, 4>,
    kind: Kind,
    flag: Bool8,
    res: Reserved<2>,
}

#[derive(EnvConfig, Debug, Clone)]
struct PlainConfig {
    #[config(default = 3)]
    size: u32,
    #[config(default = 0)]
    panic_at: u32,
}

#[derive(Observation)]
struct PlainObs {
    #[space(box(low = 0, high = 1, shape = [1]))]
    value: tessera_rl_core::ObsLeaf<i32>,
}

/// One agent, `SAME` picks the tier.
struct Plain<const SAME: bool>;

impl<const SAME: bool> Environment for Plain<SAME> {
    type State = Conf;
    type Config = PlainConfig;
    type Obs = PlainObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = if SAME {
        Tier::SamePlatform
    } else {
        Tier::CrossPlatform
    };

    fn observation_space(cfg: &PlainConfig) -> Result<Space, SpaceError> {
        PlainObs::space(cfg)
    }

    fn action_space(_: &PlainConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(_: &PlainConfig, _: &mut StepRng, state: &mut Conf) {
        *state = sample();
    }

    fn step(
        cfg: &PlainConfig,
        state: &mut Conf,
        _: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
        state.strict += 1.0;
        assert!(
            cfg.panic_at == 0 || state.strict < f64::from(cfg.panic_at),
            "injected"
        );
    }

    fn observe(
        _: &PlainConfig,
        _: &Conf,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, PlainObs>,
    ) {
    }
}

impl<const SAME: bool> ActionMask for Plain<SAME> {
    fn action_mask(_: &PlainConfig, _: &Conf, _: AgentSlot, out: &mut MaskWriter<'_>) {
        out.fill(true);
    }

    fn noop(_: &PlainConfig, _: &Conf, _: AgentSlot, _: &mut ActionWriter<'_>) {}
}

impl<const SAME: bool> Validate for Plain<SAME> {
    fn validate(_: &PlainConfig, s: &Conf) -> Result<(), InvariantViolation> {
        if s.strict < 0.0 {
            return Err(InvariantViolation {
                code: "negative strict",
            });
        }
        Ok(())
    }
}

impl<const SAME: bool> Persist for Plain<SAME> {
    const ENV_ID: &'static str = "tessera/Plain-v1";
    const ENV_VERSION: u32 = 1;
}

fn sample() -> Conf {
    let mut pool = Pool::<f32, 4>::default();
    pool.insert(-0.0).unwrap();
    pool.insert(1e-40).unwrap();
    Conf {
        strict: 0.0,
        inner: Inner {
            free: f64::from_bits(0x7ff8_dead_beef_0001),
        },
        pool,
        kind: Kind::Zero,
        flag: Bool8::TRUE,
        res: Reserved::ZERO,
    }
}

type Cross = Plain<false>;
type Same = Plain<true>;

fn opts(n: u32) -> PoolOptions {
    PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(1))
}

fn masked_with(options: PoolOptions) -> EnvPool<Cross, Masked> {
    let mut p = EnvPool::<Cross, Masked>::with_masks(PlainConfig::default(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn masked(n: u32) -> EnvPool<Cross, Masked> {
    masked_with(opts(n))
}

fn step<E>(p: &mut EnvPool<E, Masked>, n: usize)
where
    E: ActionMask,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
{
    let _ = p.step(ActionInput {
        discrete: &vec![0; n],
        continuous: &[],
    });
}

const LIB: &str = "fixture";
const LIMITS: DecodeLimits = DecodeLimits {
    max_instances: 1 << 24,
    max_decoded_bytes: 1 << 30,
};

fn file(p: &EnvPool<Cross, Masked>, ids: &[u32]) -> Vec<u8> {
    p.encode_with(ids, LIB, None).unwrap()
}

/// Rewrites the checksum so that the intended check is the one that fires.
fn reseal(bytes: &mut [u8]) {
    let n = bytes.len() - 8;
    let sum = checksum(&bytes[..n]);
    bytes[n..].copy_from_slice(&sum.to_le_bytes());
}

// Fixed offsets, as the design's table.
const ENCODING: usize = 12;
const TIER_AT: usize = 13;
const PROVIDER: usize = 14;
const AUTO_RESET: usize = 15;
const ILLEGAL_KIND: usize = 16;
const RESERVED: usize = 17;
const PENALTY: usize = 20;
const STEPS: usize = 24;
const ENV_VERSION: usize = 28;
const HASH: usize = 32;
const COUNT: usize = 40;
const RECORD_LEN: usize = 44;
const ENV_ID: usize = 48;

/// Offset of `asset_count`, past the two variable strings.
fn assets_at(bytes: &[u8]) -> usize {
    let id = usize::from(u16::from_le_bytes([bytes[ENV_ID], bytes[ENV_ID + 1]]));
    let lib_at = ENV_ID + 2 + id;
    let lib = usize::from(u16::from_le_bytes([bytes[lib_at], bytes[lib_at + 1]]));
    lib_at + 2 + lib
}

#[test]
fn test_persist_file_starts_with_magic_and_version() {
    let f = file(&masked(1), &[0]);
    assert_eq!(f[..8], MAGIC);
    assert_eq!(f[8..12], FORMAT_VERSION.to_le_bytes());
}

type Check = Box<dyn Fn(&PersistError) -> bool>;

#[test]
fn test_persist_header_mutations_are_named_errors() {
    let p = masked(2);
    let good = file(&p, &[0, 1]);
    assert_eq!(p.decode_with(&good, LIMITS, None).unwrap().len(), 2);
    let invalid = |f: &'static str| move |e: &PersistError| matches!(e, PersistError::InvalidField { field } if *field == f);
    let other = |f: &'static str| move |e: &PersistError| matches!(e, PersistError::Incompatible { field } if *field == f);
    let cases: Vec<(usize, Vec<u8>, Check)> = vec![
        (ENCODING, vec![1], Box::new(invalid("payload_encoding"))),
        (TIER_AT, vec![2], Box::new(invalid("tier"))),
        (TIER_AT, vec![1], Box::new(other("tier"))),
        (PROVIDER, vec![2], Box::new(invalid("mask_provider"))),
        (PROVIDER, vec![0], Box::new(other("mask_provider"))),
        (AUTO_RESET, vec![3], Box::new(invalid("auto_reset"))),
        (AUTO_RESET, vec![1], Box::new(other("auto_reset"))),
        (ILLEGAL_KIND, vec![2], Box::new(invalid("illegal_kind"))),
        (ILLEGAL_KIND, vec![1], Box::new(other("illegal_policy"))),
        (RESERVED, vec![1], Box::new(invalid("reserved"))),
        (PENALTY, vec![1], Box::new(invalid("illegal_penalty_bits"))),
        (STEPS, vec![1], Box::new(other("max_episode_steps"))),
        (ENV_VERSION, vec![2], Box::new(other("env_version"))),
        (HASH, vec![0xff], Box::new(other("config_hash"))),
        (RECORD_LEN, vec![0xff], Box::new(other("record_len"))),
        (ENV_ID + 2, vec![b'x'], Box::new(other("env_id"))),
    ];
    for (at, value, check) in cases {
        let mut bad = good.clone();
        bad[at..at + value.len()].copy_from_slice(&value);
        reseal(&mut bad);
        let err = p.decode_with(&bad, LIMITS, None).unwrap_err();
        assert!(check(&err), "at {at}: {err:?}");
    }
    let mut bad = good.clone();
    let at = assets_at(&bad);
    bad[at] = 1;
    reseal(&mut bad);
    assert!(matches!(
        p.decode_with(&bad, LIMITS, None),
        Err(PersistError::Incompatible { field: "assets" })
    ));
}

#[test]
fn test_persist_version_is_reported_before_the_checksum() {
    let p = masked(1);
    let mut bad = file(&p, &[0]);
    bad[8] = 9;
    assert!(matches!(
        p.decode_with(&bad, LIMITS, None),
        Err(PersistError::UnsupportedFormat { found: 9 })
    ));
    bad = file(&p, &[0]);
    bad[0] = b'X';
    assert!(matches!(
        p.decode_with(&bad, LIMITS, None),
        Err(PersistError::BadMagic)
    ));
}

#[test]
fn test_persist_every_bit_flip_and_truncation_is_refused() {
    let p = masked(1);
    let good = file(&p, &[0]);
    for i in 0..good.len() {
        for bit in 0..8 {
            let mut bad = good.clone();
            bad[i] ^= 1 << bit;
            assert!(
                p.decode_with(&bad, LIMITS, None).is_err(),
                "byte {i} bit {bit}"
            );
        }
    }
    for n in 0..good.len() {
        assert!(p.decode_with(&good[..n], LIMITS, None).is_err(), "len {n}");
    }
    let mut long = good;
    long.push(0);
    assert!(p.decode_with(&long, LIMITS, None).is_err());
}

#[test]
fn test_persist_payload_field_rules_run_after_the_checksum() {
    let p = masked(1);
    let good = file(&p, &[0]);
    let record = encode_to_vec(&DecodedRecord::from(p.snapshot(0).unwrap()));
    let payload = good.len() - 8 - record.len();
    let state = payload + record.len() - size_of::<Conf>();
    for (at, value) in [
        (state, f64::NAN.to_le_bytes().to_vec()),       // strict
        (state + 36, vec![9]),                          // kind
        (state + 37, vec![2]),                          // flag
        (state + 38, vec![1]),                          // reserved
        (state + 35, vec![7]),                          // occupancy
        (state + 28, (-0.0f32).to_le_bytes().to_vec()), // inactive entry
        (payload + 36, vec![2]),                        // header pending_reset
        (payload + 37, vec![9]),                        // header end_kind
    ] {
        let mut bad = good.clone();
        bad[at..at + value.len()].copy_from_slice(&value);
        reseal(&mut bad);
        assert!(
            matches!(
                p.decode_with(&bad, LIMITS, None),
                Err(PersistError::Record { index: 0, .. })
            ),
            "at {}",
            at - payload
        );
    }
}

#[test]
fn test_persist_oversized_input_is_refused_before_the_checksum() {
    let p = masked(1);
    let mut bad = file(&p, &[0]);
    let n = bad.len();
    bad[n - 1] ^= 1;
    let limits = DecodeLimits {
        max_instances: 8,
        max_decoded_bytes: 16,
    };
    assert!(matches!(
        p.decode_with(&bad, limits, None),
        Err(PersistError::TooLarge { limit: 16, .. })
    ));
}

#[test]
fn test_persist_round_trip_is_lossless_and_into_another_pool_size() {
    let mut p = masked(2);
    step(&mut p, 2);
    let f = file(&p, &[1, 0]);
    let mut q = masked(5);
    let records = q.decode(&f, DecodeLimits::default()).unwrap();
    q.import(&[3, 4], &records).unwrap();
    assert_eq!(q.state_hash(3).unwrap(), p.state_hash(1).unwrap());
    assert_eq!(q.state_hash(4).unwrap(), p.state_hash(0).unwrap());
    let s = q.state(3).unwrap();
    assert_eq!(s.inner.free.to_bits(), 0x7ff8_dead_beef_0001);
    assert_eq!(
        q.encode_with(&[3, 4], LIB, None).unwrap(),
        f,
        "one encoding per state"
    );
}

#[test]
fn test_persist_limits_and_counts_are_typed_errors() {
    let p = masked(2);
    let good = file(&p, &[0, 1]);
    let tight = |max_instances, max_decoded_bytes| DecodeLimits {
        max_instances,
        max_decoded_bytes,
    };
    assert!(matches!(
        p.decode_with(&good, tight(1, 1 << 30), None),
        Err(PersistError::TooManyInstances { count: 2, limit: 1 })
    ));
    assert!(matches!(
        p.decode_with(&good, tight(9, 10), None),
        Err(PersistError::TooLarge { limit: 10, .. })
    ));
    let held = good.len() as u64;
    assert!(matches!(
        p.decode_with(&good, tight(9, held), None),
        Err(PersistError::TooLarge { .. })
    ));
    for count in [u32::MAX, 3] {
        let mut bad = good.clone();
        bad[COUNT..COUNT + 4].copy_from_slice(&count.to_le_bytes());
        reseal(&mut bad);
        assert!(matches!(
            p.decode_with(&bad, LIMITS, None),
            Err(PersistError::TooManyInstances { .. })
        ));
    }
    let mut bad = good;
    bad[COUNT..COUNT + 4].copy_from_slice(&1u32.to_le_bytes());
    reseal(&mut bad);
    assert!(matches!(
        p.decode_with(&bad, LIMITS, None),
        Err(PersistError::PayloadLength)
    ));
    let small = masked(1);
    assert!(matches!(
        small.decode_with(&file(&p, &[0, 1]), LIMITS, None),
        Err(PersistError::TooManyInstances { count: 2, limit: 1 })
    ));
}

#[test]
fn test_persist_import_of_a_bad_last_record_changes_nothing() {
    let mut p = masked(3);
    step(&mut p, 3);
    let hashes: Vec<_> = (0..3).map(|i| p.state_hash(i).unwrap()).collect();
    let obs = p.outputs().rewards().to_vec();
    let masks: Vec<bool> = p.outputs().masks().iter().map(|m| m.get()).collect();
    let mut records = p.decode_with(&file(&p, &[0, 1, 2]), LIMITS, None).unwrap();
    let mut s = *records[2].state();
    s.strict = -1.0; // bit-valid, refused by `Validate`
    records[2] = DecodedRecord::from_parts(*records[2].header(), s).unwrap();
    assert!(p.import_all(&records).is_err());
    assert_eq!(
        (0..3).map(|i| p.state_hash(i).unwrap()).collect::<Vec<_>>(),
        hashes
    );
    assert_eq!(p.outputs().rewards(), obs);
    let after: Vec<bool> = p.outputs().masks().iter().map(|m| m.get()).collect();
    assert_eq!(after, masks);
}

#[test]
fn test_persist_poison_refuses_encode_and_decode_recovers() {
    let options = opts(1);
    let cfg = PlainConfig {
        panic_at: 2,
        ..PlainConfig::default()
    };
    let mut p = EnvPool::<Cross, Masked>::with_masks(cfg, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let f = p.encode_with(&[0], LIB, None).unwrap();
    while !matches!(
        p.step(ActionInput {
            discrete: &[0],
            continuous: &[],
        }),
        Err(EngineError::Panicked)
    ) {}
    assert!(matches!(
        p.encode_with(&[0], LIB, None),
        Err(PersistError::Engine(EngineError::Poisoned))
    ));
    assert!(matches!(
        p.encode(&[]),
        Err(PersistError::Engine(EngineError::Poisoned))
    ));
    let records = p.decode(&f, LIMITS).unwrap();
    p.import_all(&records).unwrap();
    assert!(p.encode_with(&[0], LIB, None).is_ok());
}

#[test]
fn test_persist_mask_provider_mismatch_is_refused_before_any_change() {
    let masked_pool = masked(1);
    let f = file(&masked_pool, &[0]);
    let mut plain = EnvPool::<Cross, NoMasks>::new(PlainConfig::default(), opts(1)).unwrap();
    plain.reset_all(Reseed::Keep).unwrap();
    let before = plain.state_hash(0).unwrap();
    assert!(matches!(
        plain.decode_with(&f, LIMITS, None),
        Err(PersistError::Incompatible {
            field: "mask_provider"
        })
    ));
    assert_eq!(plain.state_hash(0).unwrap(), before);
    assert!(plain.encode_with(&[0], LIB, None).is_ok());
}

#[test]
fn test_persist_options_are_compared_exactly() {
    let p = masked_with(PoolOptions {
        illegal_action: IllegalActionPolicy::Penalize(-0.0),
        ..opts(1)
    });
    let f = file(&p, &[0]);
    assert_eq!(f[ILLEGAL_KIND], 1);
    assert_eq!(f[PENALTY..PENALTY + 4], (-0.0f32).to_bits().to_le_bytes());
    let zero = masked_with(PoolOptions {
        illegal_action: IllegalActionPolicy::Penalize(0.0),
        ..opts(1)
    });
    assert!(matches!(
        zero.decode_with(&f, LIMITS, None),
        Err(PersistError::Incompatible {
            field: "illegal_policy"
        })
    ));
    for other in [
        PoolOptions {
            auto_reset: AutoReset::SameStep,
            ..opts(1)
        },
        PoolOptions {
            max_episode_steps: NonZeroU32::new(9),
            ..opts(1)
        },
    ] {
        assert!(
            masked_with(other)
                .decode_with(&file(&masked(1), &[0]), LIMITS, None)
                .is_err()
        );
    }
    let cfg = PlainConfig {
        size: 4,
        ..PlainConfig::default()
    };
    let q = EnvPool::<Cross, Masked>::with_masks(cfg, opts(1)).unwrap();
    assert!(matches!(
        q.decode_with(&file(&masked(1), &[0]), LIMITS, None),
        Err(PersistError::Incompatible {
            field: "config_hash"
        })
    ));
}

#[test]
fn test_persist_chunk_size_and_seed_may_differ() {
    let f = file(&masked(2), &[0, 1]);
    let other = masked_with(PoolOptions {
        chunk_size: NonZeroU32::new(1).unwrap(),
        seed: Seed::new(99),
        ..opts(2)
    });
    assert!(other.decode_with(&f, LIMITS, None).is_ok());
}

#[test]
fn test_persist_library_version_is_never_compared_and_is_capped_on_encode() {
    let p = masked(1);
    let f = p.encode_with(&[0], "9.9.9-other", None).unwrap();
    assert!(p.decode_with(&f, LIMITS, None).is_ok());
    let f = p.encode_with(&[0], "\u{fffd}", None).unwrap();
    assert!(p.decode_with(&f, LIMITS, None).is_ok());
    assert!(matches!(
        p.encode_with(&[0], &"v".repeat(65), None),
        Err(PersistError::FieldTooLong {
            field: "library_version",
            ..
        })
    ));
    assert!(matches!(
        p.encode_with(&[9], LIB, None),
        Err(PersistError::Engine(EngineError::InstanceOutOfRange { .. }))
    ));
}

#[test]
fn test_persist_same_offsets_decode_the_same_bytes() {
    let p = masked(2);
    let f = file(&p, &[0, 1]);
    for shift in 0..17 {
        let mut shifted = vec![0xaa; shift];
        shifted.extend_from_slice(&f);
        let got = p.decode_with(&shifted[shift..], LIMITS, None).unwrap();
        assert_eq!(
            encode_to_vec(&got[1]),
            encode_to_vec(&DecodedRecord::from(p.snapshot(1).unwrap()))
        );
    }
}

fn same_pool() -> EnvPool<Same, Masked> {
    let mut p = EnvPool::<Same, Masked>::with_masks(PlainConfig::default(), opts(1)).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

#[test]
fn test_persist_same_platform_needs_an_equal_explicit_id() {
    let p = same_pool();
    for absent in [None, Some("")] {
        assert!(matches!(
            p.encode_with(&[0], LIB, absent),
            Err(PersistError::BuildIdRequired)
        ));
    }
    let f = p.encode_with(&[0], LIB, Some("build-a")).unwrap();
    assert!(p.decode_with(&f, LIMITS, Some("build-a")).is_ok());
    assert!(matches!(
        p.decode_with(&f, LIMITS, None),
        Err(PersistError::BuildIdRequired)
    ));
    assert!(matches!(
        p.decode_with(&f, LIMITS, Some("build-b")),
        Err(PersistError::Incompatible {
            field: "descriptor"
        })
    ));
    assert!(matches!(
        p.encode_with(&[0], LIB, Some(&"i".repeat(129))),
        Err(PersistError::FieldTooLong {
            field: "build id",
            ..
        })
    ));
    // A changed descriptor byte, resealed.
    let needle = b"build-a";
    let at = f.windows(needle.len()).position(|w| w == needle).unwrap();
    let mut bad = f;
    bad[at] ^= 1;
    reseal(&mut bad);
    assert!(p.decode_with(&bad, LIMITS, Some("build-a")).is_err());
}

proptest::proptest! {
    /// Encode then decode returns the record, for any field values the canonical path admits.
    #[test]
    fn test_persist_round_trip_holds_for_any_state(strict in 0.0f64..1e300, free in proptest::num::u64::ANY, on in proptest::bool::ANY) {
        let mut p = masked(1);
        let mut s = sample();
        s.strict = strict;
        s.inner.free = f64::from_bits(free);
        s.flag = Bool8::new(on);
        let record = DecodedRecord::from_parts(*p.snapshot(0).unwrap().header(), s).unwrap();
        p.import(&[0], &[record]).unwrap();
        let f = file(&p, &[0]);
        let back = p.decode_with(&f, LIMITS, None).unwrap();
        proptest::prop_assert_eq!(encode_to_vec(&back[0]), encode_to_vec(&DecodedRecord::from(p.snapshot(0).unwrap())));
    }
}
