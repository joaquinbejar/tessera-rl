// SPDX-License-Identifier: MIT OR Apache-2.0

//! The block path of `Canonical::encode_slice` (#387) writes exactly the
//! bytes of one `encode` per item: for every primitive, for nested arrays,
//! across the 256-byte block boundary, and for float bit patterns that are
//! not ordinary values (NaN payloads, signed zeros).

use proptest::prelude::*;
use tessera_rl_core::encoding::{Canonical, Encoder, SliceSink, encode_to_vec};

/// One `encode` per item, into a buffer: the reference.
fn per_item<T: Canonical>(items: &[T]) -> Vec<u8> {
    let mut buf = vec![0u8; items.len() * T::ENCODED_LEN];
    let mut out = Encoder::new(SliceSink::new(&mut buf));
    for item in items {
        item.encode(&mut out);
    }
    assert!(out.sink().is_full());
    buf
}

/// The `encode_slice` path, into a buffer of the same size.
fn sliced<T: Canonical>(items: &[T]) -> Vec<u8> {
    let mut buf = vec![0u8; items.len() * T::ENCODED_LEN];
    let mut out = Encoder::new(SliceSink::new(&mut buf));
    T::encode_slice(items, &mut out);
    assert!(out.sink().is_full());
    buf
}

macro_rules! same_bytes {
    ($($name:ident: $ty:ty),*) => {$(
        proptest! {
            #[test]
            fn $name(items in prop::collection::vec(any::<$ty>(), 0..600)) {
                prop_assert_eq!(sliced(&items), per_item(&items));
            }
        }
    )*};
}

same_bytes!(
    test_encode_slice_u8_matches_per_item_encode: u8,
    test_encode_slice_i8_matches_per_item_encode: i8,
    test_encode_slice_u16_matches_per_item_encode: u16,
    test_encode_slice_i16_matches_per_item_encode: i16,
    test_encode_slice_u32_matches_per_item_encode: u32,
    test_encode_slice_i32_matches_per_item_encode: i32,
    test_encode_slice_u64_matches_per_item_encode: u64,
    test_encode_slice_i64_matches_per_item_encode: i64
);

proptest! {
    #[test]
    fn test_encode_slice_f32_bits_match_per_item_encode(bits in prop::collection::vec(any::<u32>(), 0..600)) {
        let items: Vec<f32> = bits.into_iter().map(f32::from_bits).collect();
        prop_assert_eq!(sliced(&items), per_item(&items));
    }

    #[test]
    fn test_encode_slice_f64_bits_match_per_item_encode(bits in prop::collection::vec(any::<u64>(), 0..600)) {
        let items: Vec<f64> = bits.into_iter().map(f64::from_bits).collect();
        prop_assert_eq!(sliced(&items), per_item(&items));
    }

    #[test]
    fn test_encode_slice_nested_arrays_match_per_item_encode(words in prop::collection::vec(any::<u64>(), 99)) {
        // [[u64; 33]; 3]: 792 bytes, three 256-byte blocks and a tail.
        let mut value = [[0u64; 33]; 3];
        for (dst, w) in value.as_flattened_mut().iter_mut().zip(&words) {
            *dst = *w;
        }
        prop_assert_eq!(encode_to_vec(&value), per_item(&words));
    }
}

#[test]
fn test_encode_slice_signed_zero_and_nan_payload_keep_their_bits() {
    let items = [
        -0.0f32,
        0.0,
        f32::from_bits(0x7fc0_1234),
        f32::from_bits(0xffa0_0001),
    ];
    let bytes = sliced(&items);
    let back: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    assert_eq!(back, [0x8000_0000, 0, 0x7fc0_1234, 0xffa0_0001]);
}
