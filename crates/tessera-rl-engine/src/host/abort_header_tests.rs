// SPDX-License-Identifier: MIT OR Apache-2.0

//! The record header after an aborted reset (issue #290). Every pending
//! instance that was reset before records an episode end: the snapshot
//! reader's rule "a pending reset without an episode end"
//! (`snapshot::check_record`) refuses a non-fresh pending header whose
//! `end_kind` is 0. Host pools have no snapshot API yet; this keeps their
//! headers within that rule for when they do. The reader's other rule, "a
//! pending reset in `same_step` mode", conflicts with ADR-0065 item 1 for
//! host records and is left to the host-snapshot ADR.

use core::num::NonZeroU32;
use std::sync::Arc;

use tessera_rl_core::RecordLayout;
use tessera_rl_num::identity::Seed;

use super::HostPool;
use crate::{ActionInput, AutoReset, PoolOptions, Reseed};

#[path = "../../tests/host_script/mod.rs"]
mod host_script;

use host_script::{ABORT, NOTHING, SLOTS, Script, ScriptConfig, Shared, TERMINATE, TRUNCATE};

const N: usize = 4;

type Pool = HostPool<Script<0, false>, SLOTS>;

fn pool(shared: &Arc<Shared>, mode: AutoReset) -> Pool {
    let mut o = PoolOptions::new(NonZeroU32::new(4).unwrap(), Seed::new(9));
    o.auto_reset = mode;
    Pool::new(ScriptConfig::new(shared), o).unwrap()
}

fn step(p: &mut Pool, special: &[(usize, i32)]) {
    let mut a = vec![NOTHING; N * SLOTS];
    for &(i, v) in special {
        a[i * SLOTS] = v;
    }
    p.step(ActionInput {
        discrete: &a,
        continuous: &[],
    })
    .unwrap();
}

/// Snapshot reader's rule (r1): a pending, non-fresh header has an
/// episode end. Returns how many instances are pending.
fn assert_r1(p: &Pool, ctx: &str) -> usize {
    let mut pending = 0;
    for (i, h) in p.headers.iter().enumerate() {
        let e = h.engine();
        if e.pending_reset.get() && !e.fresh.get() {
            pending += 1;
            assert_ne!(
                e.end_kind, 0,
                "{ctx}: instance {i} pending without an episode end"
            );
        }
    }
    pending
}

#[test]
fn test_host_abort_header_same_step_reset_abort_after_an_end_keeps_the_end() {
    // `end_kind`: bit 0 terminated, bit 1 truncated; an aborted step
    // truncates (ADR-0065 item 1).
    for (end, kind) in [(TERMINATE, 1), (TRUNCATE, 2), (ABORT, 2)] {
        let shared = Shared::new();
        let mut p = pool(&shared, AutoReset::SameStep);
        p.reset_all(Reseed::Keep).unwrap();
        shared.abort_resets(1, 1);
        step(&mut p, &[(1, end)]);
        let ctx = format!("same_step scratch, end {end}");
        assert_eq!(
            assert_r1(&p, &ctx),
            1,
            "{ctx}: the aborted reset leaves it pending"
        );
        assert_eq!(
            p.headers[1].engine().end_kind,
            kind,
            "{ctx}: the header keeps the ended step's end"
        );
        // The retry succeeds: nothing pending.
        step(&mut p, &[]);
        assert_eq!(assert_r1(&p, &ctx), 0, "{ctx}");
    }
}

#[test]
fn test_host_abort_header_live_reset_abort_records_an_end_in_every_mode() {
    for mode in [
        AutoReset::NextStep,
        AutoReset::SameStep,
        AutoReset::Disabled,
    ] {
        let ctx = format!("{mode:?} explicit reset");
        let shared = Shared::new();
        let mut p = pool(&shared, mode);
        p.reset_all(Reseed::Keep).unwrap();
        step(&mut p, &[]);
        shared.abort_resets(2, 1);
        p.reset(&[2], Reseed::Keep).unwrap();
        assert_eq!(assert_r1(&p, &ctx), 1, "{ctx}");
    }
}

#[test]
fn test_host_abort_header_next_step_retry_abort_records_an_end() {
    let shared = Shared::new();
    let mut p = pool(&shared, AutoReset::NextStep);
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p, &[(0, TERMINATE)]);
    assert_eq!(assert_r1(&p, "next_step end"), 1);
    shared.abort_resets(0, 1);
    step(&mut p, &[]);
    assert_eq!(assert_r1(&p, "next_step retry aborts"), 1);
    step(&mut p, &[]);
    assert_eq!(assert_r1(&p, "next_step retry succeeds"), 0);
}
