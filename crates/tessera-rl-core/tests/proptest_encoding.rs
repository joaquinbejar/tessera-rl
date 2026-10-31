// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property tests of the canonical encoding (ADR-0024 layer 4): encode and
//! decode round-trip bit for bit for core types and record headers, every
//! decoded byte string re-encodes to itself, and invalid tags are rejected.

use proptest::prelude::*;
use tessera_rl_core::encoding::{decode_from_slice, encode_to_vec};
use tessera_rl_core::{
    AgentRow, Bool8, EngineHeader, NoAdaptive, RecordHeader, StateHash, StateInit,
};
use tessera_rl_num::identity::{BranchSalt, LogicalId, RngIdentity, Seed};
use tessera_rl_num::{Fx32, Fx64};

fn bool8() -> impl Strategy<Value = Bool8> {
    any::<bool>().prop_map(Bool8::new)
}

fn identity() -> impl Strategy<Value = RngIdentity> {
    (
        any::<u64>(),
        any::<u64>(),
        any::<u32>(),
        any::<u32>(),
        any::<u32>(),
    )
        .prop_map(|(seed, salt, logical, episode, step)| {
            RngIdentity::from_parts(
                Seed::new(seed),
                BranchSalt::new(salt),
                LogicalId::new(logical),
                episode,
                step,
            )
        })
}

fn row() -> impl Strategy<Value = AgentRow> {
    (
        any::<u32>(),
        any::<f32>(),
        any::<u32>(),
        bool8(),
        bool8(),
        any::<u16>(),
    )
        .prop_map(
            |(generation, episode_return, illegal_actions, alive, acting, invalid_agent_events)| {
                AgentRow {
                    generation,
                    episode_return,
                    illegal_actions,
                    alive,
                    acting,
                    invalid_agent_events,
                }
            },
        )
}

fn header() -> impl Strategy<Value = RecordHeader<3, NoAdaptive>> {
    (
        identity(),
        any::<u32>(),
        bool8(),
        0u8..=3,
        bool8(),
        [row(), row(), row()],
    )
        .prop_map(
            |(identity, episode_steps, pending_reset, end_kind, fresh, agents)| {
                let mut h = <RecordHeader<3, NoAdaptive> as StateInit>::state_default();
                h.engine = EngineHeader {
                    identity,
                    episode_steps,
                    pending_reset,
                    end_kind,
                    fresh,
                    ..EngineHeader::default()
                };
                h.agents = agents;
                h
            },
        )
}

fn same_bits<T: tessera_rl_core::Canonical>(a: &T, b: &T) -> bool {
    encode_to_vec(a) == encode_to_vec(b)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn test_encoding_primitives_round_trip_bit_for_bit(
        a in any::<u64>(), b in any::<i32>(), c in any::<u16>(), f in any::<f32>(), d in any::<f64>(),
    ) {
        prop_assert_eq!(decode_from_slice::<u64>(&encode_to_vec(&a)).unwrap(), a);
        prop_assert_eq!(decode_from_slice::<i32>(&encode_to_vec(&b)).unwrap(), b);
        prop_assert_eq!(decode_from_slice::<u16>(&encode_to_vec(&c)).unwrap(), c);
        // Floats keep their bits: signed zeros, subnormals, NaN payloads.
        prop_assert_eq!(decode_from_slice::<f32>(&encode_to_vec(&f)).unwrap().to_bits(), f.to_bits());
        prop_assert_eq!(decode_from_slice::<f64>(&encode_to_vec(&d)).unwrap().to_bits(), d.to_bits());
    }

    #[test]
    fn test_encoding_fixed_point_round_trips(a in any::<i32>(), b in any::<i64>()) {
        let (x, y) = (Fx32::from_raw(a), Fx64::from_raw(b));
        prop_assert_eq!(decode_from_slice::<Fx32>(&encode_to_vec(&x)).unwrap(), x);
        prop_assert_eq!(decode_from_slice::<Fx64>(&encode_to_vec(&y)).unwrap(), y);
    }

    #[test]
    fn test_encoding_bool8_accepts_only_zero_and_one(byte in any::<u8>()) {
        let decoded = decode_from_slice::<Bool8>(&[byte]);
        prop_assert_eq!(decoded.is_ok(), byte <= 1);
    }

    #[test]
    fn test_encoding_record_header_round_trips_and_hashes_its_bytes(h in header()) {
        let bytes = encode_to_vec(&h);
        let back = decode_from_slice::<RecordHeader<3, NoAdaptive>>(&bytes).unwrap();
        prop_assert!(same_bits(&back, &h));
        prop_assert_eq!(StateHash::of(&back), StateHash::of(&h));
    }

    #[test]
    fn test_encoding_end_kind_beyond_three_is_rejected(h in header(), kind in 4u8..) {
        let mut bad = h;
        bad.engine.end_kind = kind;
        prop_assert!(decode_from_slice::<RecordHeader<3, NoAdaptive>>(&encode_to_vec(&bad)).is_err());
    }

    /// Whatever decodes re-encodes to exactly the same bytes (canonical).
    #[test]
    fn test_encoding_decoded_header_bytes_are_canonical(
        bytes in proptest::collection::vec(any::<u8>(), 40 + 3 * 16)
    ) {
        if let Ok(h) = decode_from_slice::<RecordHeader<3, NoAdaptive>>(&bytes) {
            prop_assert_eq!(encode_to_vec(&h), bytes);
        }
    }

    #[test]
    fn test_encoding_trailing_or_missing_bytes_are_rejected(h in header(), extra in 1usize..8) {
        let mut bytes = encode_to_vec(&h);
        let short = bytes[..bytes.len() - extra].to_vec();
        bytes.extend(std::iter::repeat_n(0u8, extra));
        prop_assert!(decode_from_slice::<RecordHeader<3, NoAdaptive>>(&bytes).is_err());
        prop_assert!(decode_from_slice::<RecordHeader<3, NoAdaptive>>(&short).is_err());
    }
}
