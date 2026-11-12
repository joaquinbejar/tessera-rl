// SPDX-License-Identifier: MIT OR Apache-2.0

//! A step through the per-entity fallback (ADR-0051) does not allocate, on
//! the fixture of `squad/mod.rs`. The counting allocator of
//! `support/counting.rs` attributes allocator calls by thread (issue #43);
//! this binary holds one test.

mod squad;

#[path = "support/counting.rs"]
mod counting;

use squad::{both, illegal_count, input, pool};
use tessera_rl_engine::{DEBUG_CHECKS, IllegalActionPolicy};

#[test]
fn test_groups_a_fallback_step_does_not_allocate() {
    if DEBUG_CHECKS {
        return;
    }
    let mut p = pool(IllegalActionPolicy::Noop);
    let row = both([1, 3, 2, 0, 0, 3]);
    p.step(input(&row)).unwrap();
    let next = both([1, 3, 1, 0, 0, 3]);
    let (stepped, tally) = counting::window(|| p.step(input(&next)).map(|_| ()));
    stepped.unwrap();
    assert_eq!(tally.calls, 0, "{tally:?}");
    assert!(illegal_count(&p) > 2);
}
