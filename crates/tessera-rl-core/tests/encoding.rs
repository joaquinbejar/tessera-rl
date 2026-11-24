// SPDX-License-Identifier: MIT OR Apache-2.0

//! Canonical encoding and `StateHash` (SPEC 8.3, ADR-0029).

use tessera_rl_core::encoding::{
    Canonical, Decoder, Encoder, HashSink, Sink, SliceSink, StateHasher, decode_f64_allow_nan,
    decode_from_slice, decode_state_enum, encode_state_enum, encode_to_slice, encode_to_vec,
};
use tessera_rl_core::{Bool8, DecodeError, EncodeError, Pool, Reserved, StateEnum, StateHash};
use tessera_rl_num::Fx32;
use tessera_rl_num::identity::{BranchSalt, LogicalId, RngIdentity, Seed};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
enum Phase {
    #[default]
    Opening = 1,
    Endgame = 2,
}

impl StateEnum for Phase {
    type Repr = u8;
    const NAME: &'static str = "Phase";

    fn to_repr(self) -> u8 {
        self as u8
    }

    fn from_repr(repr: u8) -> Option<Self> {
        match repr {
            1 => Some(Self::Opening),
            2 => Some(Self::Endgame),
            _ => None,
        }
    }
}

/// What `#[derive(EnvState)]` will generate: fields in declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Toy {
    pos: [Fx32; 2],
    speed: f64,
    hp: i16,
    phase: Phase,
    alive: Bool8,
    pad: Reserved<4>,
    units: Pool<u16, 4>,
}

impl Canonical for Toy {
    const ENCODED_LEN: usize = <[Fx32; 2]>::ENCODED_LEN
        + f64::ENCODED_LEN
        + i16::ENCODED_LEN
        + u8::ENCODED_LEN
        + Bool8::ENCODED_LEN
        + Reserved::<4>::ENCODED_LEN
        + Pool::<u16, 4>::ENCODED_LEN;

    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        self.pos.encode(out);
        self.speed.encode(out);
        self.hp.encode(out);
        encode_state_enum(self.phase, out);
        self.alive.encode(out);
        self.pad.encode(out);
        self.units.encode(out);
    }

    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        Ok(Self {
            pos: Canonical::decode(input)?,
            speed: Canonical::decode(input)?,
            hp: Canonical::decode(input)?,
            phase: decode_state_enum(input)?,
            alive: Canonical::decode(input)?,
            pad: Canonical::decode(input)?,
            units: Canonical::decode(input)?,
        })
    }
}

fn toy() -> Toy {
    let mut units = Pool::default();
    units.insert(7);
    units.insert(9);
    Toy {
        pos: [Fx32::from_raw(65_536), Fx32::from_raw(-1)],
        speed: -0.0,
        hp: -3,
        phase: Phase::Endgame,
        alive: Bool8::TRUE,
        pad: Reserved::ZERO,
        units,
    }
}

#[test]
fn test_toy_encoded_len_is_sum_of_field_widths() {
    // 8 + 8 + 2 + 1 + 1 + 4 + (8 + 4). The canonical length does not
    // depend on the in-memory layout.
    assert_eq!(Toy::ENCODED_LEN, 36);
}

#[test]
fn test_canonical_toy_round_trips_and_fixes_byte_order() {
    let value = toy();
    let bytes = encode_to_vec(&value);
    assert_eq!(bytes.len(), Toy::ENCODED_LEN);
    assert_eq!(&bytes[..8], &[0, 0, 1, 0, 0xff, 0xff, 0xff, 0xff]);
    assert_eq!(&bytes[8..16], &(-0.0f64).to_bits().to_le_bytes());
    assert_eq!(&bytes[16..20], &[0xfd, 0xff, 2, 1]);
    assert_eq!(decode_from_slice::<Toy>(&bytes), Ok(value));
}

#[test]
fn test_canonical_signed_zero_and_nan_payload_are_bit_exact() {
    let neg_zero = encode_to_vec(&-0.0f64);
    assert_ne!(neg_zero, encode_to_vec(&0.0f64));
    let payload = f64::from_bits(0x7ff8_0000_dead_beef);
    let bytes = encode_to_vec(&payload);
    assert_eq!(bytes, 0x7ff8_0000_dead_beef_u64.to_le_bytes());
    let mut input = Decoder::new(&bytes);
    assert_eq!(
        decode_f64_allow_nan(&mut input).map(f64::to_bits),
        Ok(0x7ff8_0000_dead_beef)
    );
}

#[test]
fn test_decode_allow_nan_accepts_nan_arrays() {
    use tessera_rl_core::encoding::DecodeAllowNan;
    let values = [1.0f32, f32::from_bits(0x7fc0_0123)];
    let bytes = encode_to_vec(&values);
    let mut input = Decoder::new(&bytes);
    let got = <[f32; 2]>::decode_allow_nan(&mut input).unwrap();
    assert_eq!(got.map(f32::to_bits), values.map(f32::to_bits));
    assert!(decode_from_slice::<[f32; 2]>(&bytes).is_err());
}

#[test]
fn test_decode_rejects_nan_without_allow_nan() {
    let bytes = encode_to_vec(&f32::from_bits(0x7fc0_0001));
    assert_eq!(
        decode_from_slice::<f32>(&bytes),
        Err(DecodeError::NanInState { bits: 0x7fc0_0001 })
    );
}

/// ADR-0055: floats report NaN (any payload), arrays and pools forward to
/// their elements, and float-free types report nothing.
#[test]
fn test_first_nan_floats_arrays_and_pools_report_nan_only() {
    let nan32 = f32::from_bits(0x7fc0_0001);
    assert_eq!(1.5_f32.first_nan(), None);
    assert_eq!(f32::INFINITY.first_nan(), None);
    assert!(nan32.first_nan().is_some());
    assert!(f64::from_bits(0x7ff8_0000_0000_0002).first_nan().is_some());
    assert_eq!((-0.0_f64).first_nan(), None);
    assert_eq!([0.0_f32, 1.0].first_nan(), None);
    assert!([0.0_f32, nan32].first_nan().is_some());
    let mut pool = Pool::<f32, 4>::default();
    let _ = pool.insert(1.0);
    assert_eq!(pool.first_nan(), None);
    let _ = pool.insert(nan32);
    assert!(pool.first_nan().is_some());
    assert_eq!(7_u32.first_nan(), None);
    assert_eq!(Bool8::TRUE.first_nan(), None);
}

#[test]
fn test_decode_rejects_invalid_field_bytes() {
    let good = encode_to_vec(&toy());
    let corrupt = |index: usize, byte: u8| {
        let mut b = good.clone();
        b[index] = byte;
        decode_from_slice::<Toy>(&b)
    };
    assert_eq!(
        corrupt(18, 0),
        Err(DecodeError::InvalidEnum {
            name: "Phase",
            value: 0
        })
    );
    assert_eq!(corrupt(19, 2), Err(DecodeError::InvalidBool8 { byte: 2 }));
    assert_eq!(
        corrupt(21, 1),
        Err(DecodeError::NonZeroReserved { byte: 1 })
    );
    assert_eq!(
        corrupt(28, 5),
        Err(DecodeError::InactivePoolEntry { index: 2 })
    );
}

#[test]
fn test_decode_rejects_short_and_trailing_input() {
    let good = encode_to_vec(&toy());
    assert!(matches!(
        decode_from_slice::<Toy>(&good[..good.len() - 1]),
        Err(DecodeError::UnexpectedEof { .. })
    ));
    let mut long = good.clone();
    long.push(0);
    assert_eq!(
        decode_from_slice::<Toy>(&long),
        Err(DecodeError::TrailingBytes { count: 1 })
    );
}

#[test]
fn test_encode_to_slice_checks_length() {
    let mut buf = [0u8; 36];
    assert_eq!(encode_to_slice(&toy(), &mut buf), Ok(()));
    assert_eq!(buf.to_vec(), encode_to_vec(&toy()));
    let mut short = [0u8; 35];
    assert_eq!(
        encode_to_slice(&toy(), &mut short),
        Err(EncodeError::BufferLength {
            expected: 36,
            actual: 35
        })
    );
}

#[test]
fn test_slice_sink_overflow_is_detected() {
    let mut buf = [0u8; 2];
    let mut sink = SliceSink::new(&mut buf);
    sink.write(&[1, 2, 3]);
    assert!(!sink.is_full());
}

#[test]
fn test_state_hash_streaming_equals_buffered() {
    let value = toy();
    let buffered = xxhash_rust::xxh3::xxh3_64(&encode_to_vec(&value));
    assert_eq!(StateHash::of(&value).get(), buffered);

    let identity =
        RngIdentity::from_parts(Seed::new(9), BranchSalt::new(3), LogicalId::new(1), 4, 5);
    let mut hasher = StateHasher::new();
    hasher.add(&identity);
    hasher.add(&value);
    let mut both = encode_to_vec(&identity);
    both.extend(encode_to_vec(&value));
    assert_eq!(hasher.finish().get(), xxhash_rust::xxh3::xxh3_64(&both));
}

#[test]
fn test_state_hash_distinguishes_signed_zero() {
    let mut a = toy();
    let b = a;
    a.speed = 0.0;
    assert_ne!(StateHash::of(&a), StateHash::of(&b));
}

#[test]
fn test_rng_identity_encodes_32_bytes_in_adr_order() {
    let identity =
        RngIdentity::from_parts(Seed::new(1), BranchSalt::new(2), LogicalId::new(3), 4, 5);
    let bytes = encode_to_vec(&identity);
    let mut want = Vec::new();
    want.extend(1u64.to_le_bytes());
    want.extend(2u64.to_le_bytes());
    want.extend(3u32.to_le_bytes());
    want.extend(4u32.to_le_bytes());
    want.extend(5u32.to_le_bytes());
    want.extend([0; 4]);
    assert_eq!(bytes, want);
    assert_eq!(decode_from_slice::<RngIdentity>(&bytes), Ok(identity));
}

/// The xxHash sanity buffer: byte i is the top byte of a running product
/// of `PRIME32` by `PRIME64` (xxhash `tests/sanity_test_vectors.h`).
fn sanity_buffer(len: usize) -> Vec<u8> {
    let mut state: u64 = 2_654_435_761;
    (0..len)
        .map(|_| {
            let byte = (state >> 56) as u8;
            state = state.wrapping_mul(11_400_714_785_074_694_797);
            byte
        })
        .collect()
}

#[test]
fn test_xxh3_64_matches_upstream_sanity_vectors() {
    let hash = |len| {
        let mut sink = HashSink::new();
        sink.write(&sanity_buffer(len));
        sink.digest()
    };
    assert_eq!(hash(0), 0x2D06_8005_38D3_94C2);
    assert_eq!(hash(1), 0xC44B_DFF4_074E_ECDB);
    assert_eq!(hash(6), 0x27B5_6A84_CD2D_7325);
    assert_eq!(hash(12), 0xA713_DAF0_DFBB_77E7);
    assert_eq!(hash(24), 0xA3FE_70BF_9D35_10EB);
}

/// Writes `data` into a fresh sink in the given pieces.
fn digest_in_pieces(data: &[u8], cuts: &[usize]) -> u64 {
    let mut sink = HashSink::new();
    let mut start = 0;
    for &cut in cuts.iter().chain(core::iter::once(&data.len())) {
        sink.write(data.get(start..cut).unwrap_or(&[]));
        start = cut;
    }
    sink.digest()
}

#[test]
fn test_hash_sink_two_way_split_at_every_boundary_matches_one_shot() {
    let b = HashSink::BUFFER_LEN;
    for len in [b - 1, b, b + 1, 2 * b, 2 * b + 3, 3 * b + 1] {
        let data = sanity_buffer(len);
        let want = tessera_rl_core::encoding::checksum(&data);
        for cut in 0..=len {
            assert_eq!(digest_in_pieces(&data, &[cut]), want, "len {len} cut {cut}");
        }
    }
}

#[test]
fn test_hash_sink_fixed_chunk_sizes_across_the_buffer_match_one_shot() {
    let b = HashSink::BUFFER_LEN;
    let data = sanity_buffer(4 * b + 7);
    let want = tessera_rl_core::encoding::checksum(&data);
    for chunk in (1..=b + 2).chain([2 * b - 1, 2 * b, 2 * b + 1, 4 * b + 7]) {
        let cuts: Vec<usize> = (chunk..data.len()).step_by(chunk).collect();
        assert_eq!(digest_in_pieces(&data, &cuts), want, "chunk {chunk}");
    }
}

#[test]
fn test_hash_sink_large_write_after_partial_buffer_matches_one_shot() {
    let b = HashSink::BUFFER_LEN;
    let data = sanity_buffer(3 * b + 5);
    let want = tessera_rl_core::encoding::checksum(&data);
    for small in [1, 8, b - 1, b] {
        assert_eq!(digest_in_pieces(&data, &[small]), want, "small {small}");
        assert_eq!(
            digest_in_pieces(&data, &[small, small + b, small + 2 * b]),
            want,
            "small {small}, b-sized"
        );
    }
}

#[test]
fn test_hash_sink_digest_is_repeatable_and_writes_continue_after_it() {
    let data = sanity_buffer(3 * HashSink::BUFFER_LEN);
    let (a, b) = data.split_at(700);
    let mut sink = HashSink::new();
    sink.write(a);
    let first = sink.digest();
    assert_eq!(first, sink.digest());
    assert_eq!(first, tessera_rl_core::encoding::checksum(a));
    sink.write(b);
    assert_eq!(sink.digest(), tessera_rl_core::encoding::checksum(&data));
}

#[test]
fn test_hash_sink_empty_writes_change_nothing() {
    let data = sanity_buffer(100);
    let mut sink = HashSink::new();
    sink.write(&[]);
    sink.write(&data);
    sink.write(&[]);
    assert_eq!(sink.digest(), tessera_rl_core::encoding::checksum(&data));
    assert_eq!(HashSink::new().digest(), 0x2D06_8005_38D3_94C2);
}

proptest::proptest! {
    #[test]
    fn test_hash_sink_random_splits_match_one_shot(
        data in proptest::collection::vec(proptest::num::u8::ANY, 0..3000),
        mut cuts in proptest::collection::vec(0usize..3000, 0..40),
    ) {
        cuts.retain(|&c| c <= data.len());
        cuts.sort_unstable();
        proptest::prop_assert_eq!(
            digest_in_pieces(&data, &cuts),
            tessera_rl_core::encoding::checksum(&data)
        );
    }
}

mod config {
    use tessera_rl_core::config::{
        CONFIG_FORMAT_VERSION, CanonicalConfig, canonical_bytes, config_hash, write_entry,
    };
    use tessera_rl_core::encoding::{Encoder, Sink};

    // What `#[derive(EnvConfig)]` will generate, keys sorted.
    struct Cfg {
        size: u16,
        gravity: f64,
        curriculum: u8,
    }

    impl CanonicalConfig for Cfg {
        const ADAPTIVE_FIELDS: &'static [&'static str] = &["curriculum"];
        const STATIC_ENTRIES: u16 = 2;

        fn write_entries<S: Sink>(&self, out: &mut Encoder<S>) {
            write_entry(out, "gravity", &self.gravity);
            write_entry(out, "size", &self.size);
        }
    }

    #[test]
    fn test_config_canonical_form_layout() {
        let cfg = Cfg {
            size: 16,
            gravity: 9.8,
            curriculum: 3,
        };
        let bytes = canonical_bytes(&cfg);
        let mut want = vec![CONFIG_FORMAT_VERSION];
        want.extend(7u16.to_le_bytes());
        want.extend(b"gravity");
        want.push(10);
        want.extend(9.8f64.to_bits().to_le_bytes());
        want.extend(4u16.to_le_bytes());
        want.extend(b"size");
        want.push(2);
        want.extend(16u16.to_le_bytes());
        assert_eq!(bytes, want);
    }

    #[test]
    fn test_config_hash_ignores_adaptive_and_tracks_static_fields() {
        let a = Cfg {
            size: 16,
            gravity: 9.8,
            curriculum: 3,
        };
        let b = Cfg { curriculum: 7, ..a };
        let c = Cfg { size: 17, ..a };
        let hash = |cfg: &Cfg| config_hash("tessera/Toy", 1, cfg).unwrap();
        assert_ne!(a.curriculum, b.curriculum);
        assert_eq!(hash(&a), hash(&b));
        assert_ne!(hash(&a), hash(&c));
        assert_ne!(hash(&a), config_hash("tessera/Toy", 2, &a).unwrap());
        assert_ne!(hash(&a), config_hash("tessera/Other", 1, &a).unwrap());
    }
}
