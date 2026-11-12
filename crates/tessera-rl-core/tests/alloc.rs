// SPDX-License-Identifier: MIT OR Apache-2.0

//! Zero-allocation checks for hashing, slice encoding, decoding and
//! canonical comparison (SPEC 5.5, ADR-0004).
//!
//! The counting allocator of `support/counting.rs` attributes allocator
//! calls by thread (issue #43). This binary holds a single measuring test,
//! which runs each scenario (the former tests, by name) with its own
//! objects and windows, and reports every failing scenario together.

#[path = "support/counting.rs"]
mod counting;

use counting::window;
use tessera_rl_core::encoding::{
    StateHasher, canonical_eq, decode_from_slice, encode_to_slice, encode_to_vec,
};
use tessera_rl_core::{Bool8, Pool, Reserved, StateHash};
use tessera_rl_num::Fx32;
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed};

/// Allocation calls of the counted threads while `f` runs.
fn allocations(f: impl FnOnce()) -> u64 {
    window(f).1.calls
}

fn pool() -> Pool<[Fx32; 2], 8> {
    let mut p = Pool::default();
    p.insert([Fx32::from_raw(3), Fx32::from_raw(-4)]);
    p.insert([Fx32::from_raw(5), Fx32::from_raw(6)]);
    p
}

fn test_allocations_counter_detects_heap_use() {
    assert!(allocations(|| drop(std::hint::black_box(vec![1u8; 16]))) > 0);
}

fn test_state_hash_does_not_allocate() {
    let value = pool();
    let identity = RngIdentity::new(Seed::new(7), LogicalId::DEFAULT);
    let n = allocations(|| {
        std::hint::black_box(StateHash::of(&value));
        let mut hasher = StateHasher::new();
        hasher.add(&identity);
        hasher.add(&value);
        std::hint::black_box(hasher.finish());
    });
    assert_eq!(n, 0);
}

fn test_slice_encoding_and_decoding_do_not_allocate() {
    let value = pool();
    let bytes = encode_to_vec(&value);
    let mut buf = vec![0u8; bytes.len()];
    let flags = [Bool8::TRUE; 3];
    let n = allocations(|| {
        let _ = std::hint::black_box(encode_to_slice(&value, &mut buf));
        let _ = std::hint::black_box(decode_from_slice::<Pool<[Fx32; 2], 8>>(&bytes));
        std::hint::black_box(canonical_eq(&flags, &flags));
        std::hint::black_box(canonical_eq(&Reserved::<128>::ZERO, &Reserved::ZERO));
    });
    assert_eq!(n, 0);
    assert_eq!(buf, bytes);
}

/// The scenarios this binary runs.
const SCENARIOS: usize = 3;

#[test]
fn test_core_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_allocations_counter_detects_heap_use",
                test_allocations_counter_detects_heap_use,
            ),
            (
                "test_state_hash_does_not_allocate",
                test_state_hash_does_not_allocate,
            ),
            (
                "test_slice_encoding_and_decoding_do_not_allocate",
                test_slice_encoding_and_decoding_do_not_allocate,
            ),
        ],
    );
}
