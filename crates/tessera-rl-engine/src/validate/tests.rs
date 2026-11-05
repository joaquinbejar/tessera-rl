// SPDX-License-Identifier: MIT OR Apache-2.0

//! The split-phase checks and the restricted action copy (ADR-0047).

use core::sync::atomic::AtomicBool;

use tessera_rl_core::{Bool8, NoAdaptive, RecordHeader, StateInit};
use tessera_rl_num::identity::{BranchSalt, LogicalId, RngIdentity, Seed};

use super::*;

type H = RecordHeader<2, NoAdaptive>;

const AGENTS: usize = 2;

/// A running instance: both agents alive and acting.
fn running() -> H {
    let mut h = H::state_default();
    h.engine.pending_reset = Bool8::FALSE;
    h.engine.fresh = Bool8::FALSE;
    for row in &mut h.agents {
        row.alive = Bool8::TRUE;
        row.acting = Bool8::TRUE;
    }
    h
}

fn headers(n: usize) -> Vec<H> {
    (0..n).map(|_| running()).collect()
}

/// One discrete head with 3 choices, no continuous head.
fn layout() -> ActionLayout {
    ActionLayout {
        discrete: vec![3],
        continuous: Vec::new(),
    }
}

fn flags(n: usize) -> Vec<AtomicBool> {
    (0..n).map(|_| AtomicBool::new(false)).collect()
}

fn input(discrete: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete,
        continuous: &[],
    }
}

const SENTINEL: i32 = 9;

#[test]
fn test_check_send_ids_empty_is_empty_send() {
    let mut seen = vec![Bool8::FALSE; 4];
    let err = check_send_ids(&[], &mut seen, &flags(4), &flags(4));
    assert_eq!(err, Err(EngineError::EmptySend));
}

#[test]
fn test_check_send_ids_duplicate_and_out_of_range_are_refused() {
    let mut seen = vec![Bool8::FALSE; 4];
    assert_eq!(
        check_send_ids(&[1, 1], &mut seen, &flags(4), &flags(4)),
        Err(EngineError::DuplicateInstance { id: 1 })
    );
    assert_eq!(
        check_send_ids(&[0, 4], &mut seen, &flags(4), &flags(4)),
        Err(EngineError::InstanceOutOfRange { id: 4, num_envs: 4 })
    );
    assert!(
        seen.iter().all(|s| !s.get()),
        "the scratch is cleared after a refusal"
    );
}

#[test]
fn test_check_send_ids_in_flight_or_borrowed_id_is_refused() {
    let mut seen = vec![Bool8::FALSE; 4];
    let in_flight = flags(4);
    let live = flags(4);
    in_flight[2].store(true, Ordering::Release);
    assert_eq!(
        check_send_ids(&[0, 2], &mut seen, &in_flight, &live),
        Err(EngineError::InFlight { id: 2 })
    );
    live[3].store(true, Ordering::Release);
    assert_eq!(
        check_send_ids(&[3], &mut seen, &in_flight, &live),
        Err(EngineError::Borrowed { id: 3 })
    );
    // Disjoint ids are fine while others are in flight or borrowed.
    assert_eq!(
        check_send_ids(&[0, 1], &mut seen, &in_flight, &live),
        Ok(())
    );
}

#[test]
fn test_check_send_half_states_give_their_errors() {
    let in_flight = [AtomicBool::new(false), AtomicBool::new(true)];
    let live = [AtomicBool::new(true), AtomicBool::new(false)];
    assert_eq!(
        check_send_half(2, &in_flight, &live),
        Err(EngineError::InvalidHalf { half: 2 })
    );
    assert_eq!(
        check_send_half(1, &in_flight, &live),
        Err(EngineError::HalfInFlight { half: 1 })
    );
    assert_eq!(
        check_send_half(0, &in_flight, &live),
        Err(EngineError::HalfBorrowed { half: 0 })
    );
    let free = [AtomicBool::new(false), AtomicBool::new(false)];
    assert_eq!(check_send_half(1, &free, &free), Ok(1));
}

#[test]
fn test_check_cycle_lag_two_admits_only_after_the_commit() {
    assert_eq!(check_cycle(0, None), Ok(()));
    assert_eq!(check_cycle(1, None), Ok(()));
    assert_eq!(
        check_cycle(2, None),
        Err(EngineError::CycleAhead {
            cycle: 2,
            needed: 0
        })
    );
    assert_eq!(check_cycle(2, Some(0)), Ok(()));
    assert_eq!(
        check_cycle(5, Some(2)),
        Err(EngineError::CycleAhead {
            cycle: 5,
            needed: 3
        })
    );
    assert_eq!(check_cycle(5, Some(3)), Ok(()));
    assert_eq!(check_cycle(u64::MAX, Some(u64::MAX - 2)), Ok(()));
}

#[test]
fn test_check_recv_nothing_or_too_little_never_waits() {
    assert_eq!(check_recv(0, 0, 1), Err(EngineError::NothingInFlight));
    assert_eq!(
        check_recv(2, 1, 4),
        Err(EngineError::BatchUnreachable {
            in_flight: 2,
            ready: 1,
            batch_size: 4
        })
    );
    assert_eq!(check_recv(0, 4, 4), Ok(()));
    assert_eq!(check_recv(3, 1, 4), Ok(()));
    assert_eq!(check_recv(1, 0, 1), Ok(()));
}

#[test]
fn test_subset_copy_writes_only_the_sent_rows() {
    let hs = headers(4);
    let mut out = vec![SENTINEL; 4 * AGENTS];
    // ids 3 and 1, in that order: rows [1, 2] for id 3, [0, 2] for id 1.
    let acts = [1, 2, 0, 2];
    check_and_copy_subset_actions(
        &layout(),
        AutoReset::NextStep,
        &hs,
        &[3, 1],
        input(&acts),
        &mut out,
        &mut [],
    )
    .unwrap();
    assert_eq!(out, [SENTINEL, SENTINEL, 0, 2, SENTINEL, SENTINEL, 1, 2]);
}

#[test]
fn test_range_copy_writes_only_the_half_rows() {
    let hs = headers(4);
    let mut out = vec![SENTINEL; 4 * AGENTS];
    let acts = [2, 1, 0, 1];
    check_and_copy_range_actions(
        &layout(),
        AutoReset::NextStep,
        &hs,
        2..4,
        input(&acts),
        &mut out,
        &mut [],
    )
    .unwrap();
    assert_eq!(out, [SENTINEL, SENTINEL, SENTINEL, SENTINEL, 2, 1, 0, 1]);
}

#[test]
fn test_subset_copy_invalid_action_changes_nothing() {
    let hs = headers(4);
    let mut out = vec![SENTINEL; 4 * AGENTS];
    // The second id's second agent has 3, outside 0..3.
    let acts = [1, 2, 0, 3];
    let err = check_and_copy_subset_actions(
        &layout(),
        AutoReset::NextStep,
        &hs,
        &[0, 2],
        input(&acts),
        &mut out,
        &mut [],
    );
    assert!(matches!(
        err,
        Err(EngineError::DiscreteOutOfRange {
            instance: 2,
            agent: 1,
            ..
        })
    ));
    assert!(out.iter().all(|&v| v == SENTINEL), "no row was written");
}

#[test]
fn test_subset_copy_wrong_input_length_changes_nothing() {
    let hs = headers(4);
    let mut out = vec![SENTINEL; 4 * AGENTS];
    let err = check_and_copy_subset_actions(
        &layout(),
        AutoReset::NextStep,
        &hs,
        &[0, 2],
        input(&[1, 2, 0]),
        &mut out,
        &mut [],
    );
    assert_eq!(
        err,
        Err(EngineError::ActionLength {
            buffer: "discrete",
            expected: 4,
            actual: 3
        })
    );
    assert!(out.iter().all(|&v| v == SENTINEL));
}

#[test]
fn test_subset_copy_only_requires_alive_acting_slots() {
    let mut hs = headers(2);
    hs[1].agents[0].acting = Bool8::FALSE;
    hs[1].agents[1].alive = Bool8::FALSE;
    let mut out = vec![SENTINEL; 2 * AGENTS];
    // Garbage for both slots of id 1: not validated, stored as zero.
    let acts = [-5, 77];
    check_and_copy_subset_actions(
        &layout(),
        AutoReset::NextStep,
        &hs,
        &[1],
        input(&acts),
        &mut out,
        &mut [],
    )
    .unwrap();
    assert_eq!(out, [SENTINEL, SENTINEL, 0, 0]);
}

#[test]
fn test_subset_copy_pending_reset_follows_the_auto_reset_mode() {
    let mut hs = headers(2);
    hs[0].engine.pending_reset = Bool8::TRUE;
    let mut out = vec![SENTINEL; 2 * AGENTS];
    let acts = [-1, -1];
    // NextStep: the instance resets instead of stepping; no action needed.
    check_and_copy_subset_actions(
        &layout(),
        AutoReset::NextStep,
        &hs,
        &[0],
        input(&acts),
        &mut out,
        &mut [],
    )
    .unwrap();
    assert_eq!(out, [0, 0, SENTINEL, SENTINEL]);
    // Disabled: an ended instance must be reset by the caller first.
    let mut out = vec![SENTINEL; 2 * AGENTS];
    let err = check_and_copy_subset_actions(
        &layout(),
        AutoReset::Disabled,
        &hs,
        &[0],
        input(&acts),
        &mut out,
        &mut [],
    );
    assert_eq!(err, Err(EngineError::PendingReset { id: 0 }));
    assert!(out.iter().all(|&v| v == SENTINEL));
}

#[test]
fn test_subset_copy_exhausted_episode_index_is_refused() {
    let mut hs = headers(2);
    hs[1].engine.pending_reset = Bool8::TRUE;
    hs[1].engine.identity = RngIdentity::from_parts(
        Seed::new(1),
        BranchSalt::ROOT,
        LogicalId::new(0),
        u32::MAX,
        0,
    );
    let mut out = vec![SENTINEL; 2 * AGENTS];
    let err = check_and_copy_subset_actions(
        &layout(),
        AutoReset::NextStep,
        &hs,
        &[1],
        input(&[0, 0]),
        &mut out,
        &mut [],
    );
    assert_eq!(
        err,
        Err(EngineError::CounterExhausted {
            id: 1,
            counter: "episode_index"
        })
    );
    assert!(out.iter().all(|&v| v == SENTINEL));
}

#[test]
fn test_range_copy_bad_range_is_refused() {
    let hs = headers(4);
    let mut out = vec![SENTINEL; 4 * AGENTS];
    let err = check_and_copy_range_actions(
        &layout(),
        AutoReset::NextStep,
        &hs,
        2..5,
        input(&[0; 6]),
        &mut out,
        &mut [],
    );
    assert_eq!(
        err,
        Err(EngineError::InstanceOutOfRange { id: 5, num_envs: 4 })
    );
    assert!(out.iter().all(|&v| v == SENTINEL));
}

#[test]
fn test_subset_and_range_agree_with_the_full_copy() {
    let hs = headers(4);
    let acts: Vec<i32> = (0..8).map(|i| i % 3).collect();
    let mut full = vec![0; 8];
    check_and_copy_actions(
        &layout(),
        AutoReset::NextStep,
        &hs,
        input(&acts),
        &mut full,
        &mut [],
    )
    .unwrap();
    let mut range = vec![0; 8];
    check_and_copy_range_actions(
        &layout(),
        AutoReset::NextStep,
        &hs,
        0..4,
        input(&acts),
        &mut range,
        &mut [],
    )
    .unwrap();
    let mut subset = vec![0; 8];
    check_and_copy_subset_actions(
        &layout(),
        AutoReset::NextStep,
        &hs,
        &[0, 1, 2, 3],
        input(&acts),
        &mut subset,
        &mut [],
    )
    .unwrap();
    assert_eq!(full, range);
    assert_eq!(full, subset);
}

#[test]
fn test_expected_len_overflow_is_an_action_length_error() {
    assert_eq!(
        expected_len("discrete", &[usize::MAX, 2, 1], 7),
        Err(EngineError::ActionLength {
            buffer: "discrete",
            expected: usize::MAX,
            actual: 7
        })
    );
    assert_eq!(
        expected_len("continuous", &[2, usize::MAX / 2 + 1], 0),
        Err(EngineError::ActionLength {
            buffer: "continuous",
            expected: usize::MAX,
            actual: 0
        })
    );
    assert_eq!(expected_len("discrete", &[4, 2, 3], 24), Ok(24));
    assert_eq!(expected_len("discrete", &[4, 2, 0], 0), Ok(0));
}
