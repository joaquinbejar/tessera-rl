// SPDX-License-Identifier: MIT OR Apache-2.0

//! The split-phase checks and the restricted action copy (ADR-0047).

use crate::sys::sync::AtomicBool;

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
        ..ActionLayout::default()
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

/// A safe stand-in for the engine's admitted rows: each admitted id owns
/// its rows, copied out of and back into a full `[N, A]` action buffer.
struct Owned {
    ids: Vec<u32>,
    instances: usize,
    headers: Vec<H>,
    discrete: Vec<Vec<i32>>,
    continuous: Vec<Vec<f32>>,
}

impl AdmittedRows<H> for Owned {
    fn len(&self) -> usize {
        self.ids.len()
    }

    fn instances(&self) -> usize {
        self.instances
    }

    fn id(&self, position: usize) -> Option<u32> {
        self.ids.get(position).copied()
    }

    fn row(&mut self, position: usize) -> Option<ActionRow<'_, H>> {
        let continuous = match self.continuous.get_mut(position) {
            Some(c) => c.as_mut_slice(),
            None => &mut [],
        };
        Some(ActionRow {
            header: self.headers.get(position)?,
            discrete: self.discrete.get_mut(position)?,
            continuous,
        })
    }
}

/// Runs a subset send of `ids` over `hs` and returns the result and the
/// full action buffer, which starts filled with `SENTINEL`.
fn subset(
    hs: &[H],
    ids: &[u32],
    acts: &[i32],
    mode: AutoReset,
) -> (Result<(), EngineError>, Vec<i32>) {
    let mut out = vec![SENTINEL; hs.len() * AGENTS];
    let mut rows = Owned {
        ids: ids.to_vec(),
        instances: hs.len(),
        continuous: Vec::new(),
        headers: ids.iter().map(|&id| hs[id as usize]).collect(),
        discrete: ids
            .iter()
            .map(|&id| out[id as usize * AGENTS..(id as usize + 1) * AGENTS].to_vec())
            .collect(),
    };
    let result = check_and_copy_subset_actions(&layout(), mode, &mut rows, input(acts));
    for (&id, row) in ids.iter().zip(&rows.discrete) {
        out[id as usize * AGENTS..(id as usize + 1) * AGENTS].copy_from_slice(row);
    }
    (result, out)
}

/// Runs a range send of `r` over the half's own views of `hs` and returns
/// the result and the full action buffer, which starts with `SENTINEL`.
fn range(
    hs: &[H],
    r: core::ops::Range<usize>,
    acts: &[i32],
    mode: AutoReset,
) -> (Result<(), EngineError>, Vec<i32>) {
    let mut out = vec![SENTINEL; hs.len() * AGENTS];
    let rows = r.start * AGENTS..r.end * AGENTS;
    let result = check_and_copy_range_actions(
        &layout(),
        mode,
        r.start,
        &hs[r],
        input(acts),
        &mut out[rows],
        &mut [],
    );
    (result, out)
}

#[test]
fn test_subset_copy_writes_only_the_sent_rows() {
    // ids 3 and 1, in that order: rows [1, 2] for id 3, [0, 2] for id 1.
    let (result, out) = subset(&headers(4), &[3, 1], &[1, 2, 0, 2], AutoReset::NextStep);
    result.unwrap();
    assert_eq!(out, [SENTINEL, SENTINEL, 0, 2, SENTINEL, SENTINEL, 1, 2]);
}

#[test]
fn test_range_copy_writes_only_the_half_rows() {
    let (result, out) = range(&headers(4), 2..4, &[2, 1, 0, 1], AutoReset::NextStep);
    result.unwrap();
    assert_eq!(out, [SENTINEL, SENTINEL, SENTINEL, SENTINEL, 2, 1, 0, 1]);
}

#[test]
fn test_subset_copy_invalid_action_changes_nothing() {
    // The second id's second agent has 3, outside 0..3.
    let (result, out) = subset(&headers(4), &[0, 2], &[1, 2, 0, 3], AutoReset::NextStep);
    assert!(matches!(
        result,
        Err(EngineError::DiscreteOutOfRange {
            instance: 2,
            agent: 1,
            ..
        })
    ));
    assert!(out.iter().all(|&v| v == SENTINEL), "no row was written");
}

#[test]
fn test_range_copy_invalid_action_reports_the_absolute_id() {
    let (result, out) = range(&headers(4), 2..4, &[0, 0, 5, 0], AutoReset::NextStep);
    assert!(matches!(
        result,
        Err(EngineError::DiscreteOutOfRange {
            instance: 3,
            agent: 0,
            ..
        })
    ));
    assert!(out.iter().all(|&v| v == SENTINEL));
}

#[test]
fn test_subset_copy_wrong_input_length_changes_nothing() {
    let (result, out) = subset(&headers(4), &[0, 2], &[1, 2, 0], AutoReset::NextStep);
    assert_eq!(
        result,
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
    // Garbage for both slots of id 1: not validated, stored as zero.
    let (result, out) = subset(&hs, &[1], &[-5, 77], AutoReset::NextStep);
    result.unwrap();
    assert_eq!(out, [SENTINEL, SENTINEL, 0, 0]);
}

#[test]
fn test_subset_copy_pending_reset_follows_the_auto_reset_mode() {
    let mut hs = headers(2);
    hs[0].engine.pending_reset = Bool8::TRUE;
    // NextStep: the instance resets instead of stepping; no action needed.
    let (result, out) = subset(&hs, &[0], &[-1, -1], AutoReset::NextStep);
    result.unwrap();
    assert_eq!(out, [0, 0, SENTINEL, SENTINEL]);
    // Disabled: an ended instance must be reset by the caller first.
    let (result, out) = subset(&hs, &[0], &[-1, -1], AutoReset::Disabled);
    assert_eq!(result, Err(EngineError::PendingReset { id: 0 }));
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
    let (result, out) = subset(&hs, &[1], &[0, 0], AutoReset::NextStep);
    assert_eq!(
        result,
        Err(EngineError::CounterExhausted {
            id: 1,
            counter: "episode_index"
        })
    );
    assert!(out.iter().all(|&v| v == SENTINEL));
}

#[test]
fn test_range_copy_wrong_output_length_is_refused() {
    let hs = headers(4);
    let mut out = vec![SENTINEL; 3];
    let result = check_and_copy_range_actions(
        &layout(),
        AutoReset::NextStep,
        2,
        &hs[2..4],
        input(&[0; 4]),
        &mut out,
        &mut [],
    );
    assert_eq!(
        result,
        Err(EngineError::ActionLength {
            buffer: "discrete",
            expected: 4,
            actual: 3
        })
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
    let (result, by_range) = range(&hs, 0..4, &acts, AutoReset::NextStep);
    result.unwrap();
    let (result, by_subset) = subset(&hs, &[0, 1, 2, 3], &acts, AutoReset::NextStep);
    result.unwrap();
    assert_eq!(full, by_range);
    assert_eq!(full, by_subset);
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
    // A zero factor wins over an overflowing product.
    assert_eq!(expected_len("discrete", &[usize::MAX, 2, 0], 0), Ok(0));
}

/// One discrete head with 3 choices and one continuous head.
fn layout_c() -> ActionLayout {
    ActionLayout {
        discrete: vec![3],
        continuous: vec![(-1.0, 1.0)],
        ..ActionLayout::default()
    }
}

const SENTINEL_C: f32 = 9.0;

#[test]
fn test_subset_copy_continuous_head_out_of_order_ids_takes_each_source_position() {
    let hs = headers(4);
    let ids = [3_u32, 1];
    let mut out_d = vec![SENTINEL; 4 * AGENTS];
    let mut out_c = vec![SENTINEL_C; 4 * AGENTS];
    let rows_of = |v: &[f32], id: u32| v[id as usize * AGENTS..(id as usize + 1) * AGENTS].to_vec();
    let mut rows = Owned {
        ids: ids.to_vec(),
        instances: hs.len(),
        headers: ids.iter().map(|&id| hs[id as usize]).collect(),
        discrete: ids
            .iter()
            .map(|&id| out_d[id as usize * AGENTS..(id as usize + 1) * AGENTS].to_vec())
            .collect(),
        continuous: ids.iter().map(|&id| rows_of(&out_c, id)).collect(),
    };
    let input = ActionInput {
        discrete: &[1, 2, 0, 2],
        continuous: &[0.5, -0.5, 0.25, -0.25],
    };
    check_and_copy_subset_actions(&layout_c(), AutoReset::NextStep, &mut rows, input).unwrap();
    for ((&id, d), c) in ids.iter().zip(&rows.discrete).zip(&rows.continuous) {
        let span = id as usize * AGENTS..(id as usize + 1) * AGENTS;
        out_d[span.clone()].copy_from_slice(d);
        out_c[span].copy_from_slice(c);
    }
    assert_eq!(
        out_c,
        [
            SENTINEL_C, SENTINEL_C, 0.25, -0.25, SENTINEL_C, SENTINEL_C, 0.5, -0.5
        ]
    );
    assert_eq!(out_d, [SENTINEL, SENTINEL, 0, 2, SENTINEL, SENTINEL, 1, 2]);
}

#[test]
fn test_range_copy_continuous_head_copies_every_value_of_the_half() {
    let hs = headers(4);
    let mut out_d = [SENTINEL; 4 * AGENTS];
    let mut out_c = [SENTINEL_C; 4 * AGENTS];
    let input = ActionInput {
        discrete: &[2, 1, 0, 1],
        continuous: &[0.1, 0.2, 0.3, 0.4],
    };
    check_and_copy_range_actions(
        &layout_c(),
        AutoReset::NextStep,
        2,
        &hs[2..4],
        input,
        &mut out_d[4..8],
        &mut out_c[4..8],
    )
    .unwrap();
    assert_eq!(
        out_c,
        [
            SENTINEL_C, SENTINEL_C, SENTINEL_C, SENTINEL_C, 0.1, 0.2, 0.3, 0.4
        ]
    );
    assert_eq!(out_d[4..8], [2, 1, 0, 1]);
}

#[test]
fn test_subset_copy_empty_continuous_source_with_a_head_is_a_length_error() {
    let hs = headers(4);
    let mut rows = Owned {
        ids: vec![1],
        instances: hs.len(),
        headers: vec![hs[1]],
        discrete: vec![vec![SENTINEL; AGENTS]],
        continuous: vec![vec![SENTINEL_C; AGENTS]],
    };
    let input = ActionInput {
        discrete: &[0, 0],
        continuous: &[],
    };
    let err = check_and_copy_subset_actions(&layout_c(), AutoReset::NextStep, &mut rows, input);
    assert!(
        matches!(
            err,
            Err(EngineError::ActionLength {
                buffer: "continuous",
                ..
            })
        ),
        "{err:?}"
    );
    assert_eq!(rows.continuous[0], [SENTINEL_C; AGENTS], "nothing written");
}

#[test]
fn test_subset_copy_short_row_is_a_length_error_before_any_copy() {
    let hs = headers(4);
    // The admitted row of id 1 is one value short: the row-length guard
    // refuses before the copy pass, so id 3 stays untouched too.
    let mut rows = Owned {
        ids: vec![3, 1],
        instances: hs.len(),
        headers: vec![hs[3], hs[1]],
        discrete: vec![vec![SENTINEL; AGENTS], vec![SENTINEL; AGENTS - 1]],
        continuous: Vec::new(),
    };
    let err = check_and_copy_subset_actions(
        &layout(),
        AutoReset::NextStep,
        &mut rows,
        input(&[1, 2, 0, 2]),
    );
    assert!(
        matches!(err, Err(EngineError::ActionLength { .. })),
        "{err:?}"
    );
    assert_eq!(
        rows.discrete[0], [SENTINEL; AGENTS],
        "the first id is not copied"
    );
}

#[test]
fn test_subset_copy_missing_row_reports_the_pool_size() {
    struct Missing;
    impl AdmittedRows<H> for Missing {
        fn len(&self) -> usize {
            1
        }
        fn instances(&self) -> usize {
            10
        }
        fn id(&self, _: usize) -> Option<u32> {
            Some(42)
        }
        fn row(&mut self, _: usize) -> Option<ActionRow<'_, H>> {
            None
        }
    }
    let err =
        check_and_copy_subset_actions(&layout(), AutoReset::NextStep, &mut Missing, input(&[0, 0]));
    assert_eq!(
        err,
        Err(EngineError::InstanceOutOfRange {
            id: 42,
            num_envs: 10
        })
    );
}
