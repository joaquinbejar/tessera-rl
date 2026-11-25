// SPDX-License-Identifier: MIT OR Apache-2.0
//! The output-storage types stay `UnwindSafe + RefUnwindSafe` with the
//! `output_allocator` option (ADR-0084): a compile-time check, so a change
//! that drops either auto trait fails to build, not only the report-only
//! facade API snapshot.

use core::panic::{RefUnwindSafe, UnwindSafe};

use tessera_rl::classic::CartPole;
use tessera_rl::engine::{OutputStorage, Outputs, PoolOptions};
use tessera_rl::search::LeafBatch;

const fn assert_unwind_safe<T: UnwindSafe + RefUnwindSafe + ?Sized>() {}
const fn assert_ref_unwind_safe<T: RefUnwindSafe + ?Sized>() {}

#[test]
fn test_output_types_unwind_safe_with_allocator_option_compiles() {
    assert_unwind_safe::<PoolOptions>();
    assert_unwind_safe::<OutputStorage>();
    assert_unwind_safe::<Outputs>();
    // `LeafBatch` holds an `&mut` scratch slice, so it was never
    // `UnwindSafe` (as on main); what the allocator field could break is its
    // `RefUnwindSafe`, through the `Outputs` it borrows.
    assert_ref_unwind_safe::<LeafBatch<'static, CartPole>>();
}
