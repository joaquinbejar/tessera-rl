// SPDX-License-Identifier: MIT OR Apache-2.0

//! Zero-allocation checks for hashing, slice encoding, decoding and
//! canonical comparison (SPEC 5.5, ADR-0004).
//!
//! This file installs a counting global allocator for this test binary
//! only. It is test code, never compiled into the library, so it is not one
//! of the audited `unsafe` modules; its `unsafe` is limited to forwarding to
//! the system allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use tessera_rl_core::encoding::{
    StateHasher, canonical_eq, decode_from_slice, encode_to_slice, encode_to_vec,
};
use tessera_rl_core::{Bool8, Pool, Reserved, StateHash};
use tessera_rl_num::Fx32;
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed};

thread_local! {
    // Per thread, so allocations of other test threads are not counted.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

struct Counting;

#[expect(unsafe_code, reason = "test-only counting allocator")]
// SAFETY: every method forwards to `System` with the same arguments, so the
// `GlobalAlloc` contract is upheld by the system allocator. The counters are
// const-initialized thread locals, which never allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: forwarded unchanged; the caller upholds `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn count() {
    let _ = COUNTING.try_with(|on| {
        if on.get() {
            let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        }
    });
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Runs `f` and returns how many allocations it made on this thread.
fn allocations(f: impl FnOnce()) -> u64 {
    ALLOCATIONS.with(|n| n.set(0));
    COUNTING.with(|on| on.set(true));
    f();
    COUNTING.with(|on| on.set(false));
    ALLOCATIONS.with(Cell::get)
}

fn pool() -> Pool<[Fx32; 2], 8> {
    let mut p = Pool::default();
    p.insert([Fx32::from_raw(3), Fx32::from_raw(-4)]);
    p.insert([Fx32::from_raw(5), Fx32::from_raw(6)]);
    p
}

#[test]
fn test_allocations_counter_detects_heap_use() {
    assert!(allocations(|| drop(std::hint::black_box(vec![1u8; 16]))) > 0);
}

#[test]
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

#[test]
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
