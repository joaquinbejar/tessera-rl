// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-entity illegal-action fallback (ADR-0051) on the fixture of
//! `squad/mod.rs`. The allocation test of a fallback step has its own
//! binary, `entity_groups_alloc.rs`.

#![expect(clippy::unwrap_used, reason = "test fixture")]

mod squad;

use core::num::NonZeroUsize;

use squad::{
    HEADS, Pool, Squad, SquadConfig, UNITS, both, cfg, illegal_count, input, options, pool,
};
use tessera_rl_core::{Bool8, Environment, NoAdaptive, RecordHeader, RecordLayout};
use tessera_rl_engine::{
    AsyncOptions, DEBUG_CHECKS, DecodedRecord, EngineError, EnvPool, IllegalActionPolicy, Masked,
    PipelinedOptions, Reseed, SyncOptions,
};

#[test]
fn test_groups_one_illegal_unit_leaves_the_others_orders() {
    if DEBUG_CHECKS {
        return;
    }
    let mut p = pool(IllegalActionPolicy::Noop);
    p.step(input(&both([1, 1, 2, 0, 2, 2]))).unwrap();
    assert_eq!(p.state(0).unwrap().last[0], [11, 0, 22]);
    assert_eq!(illegal_count(&p), 1);
}

#[test]
fn test_groups_an_illegal_child_falls_back_whole() {
    if DEBUG_CHECKS {
        return;
    }
    let mut p = pool(IllegalActionPolicy::Noop);
    p.step(input(&both([1, 3, 1, 0, 0, 0]))).unwrap();
    assert_eq!(p.state(0).unwrap().last[0], [0, 10, 0]);
    assert_eq!(illegal_count(&p), 1);
}

#[test]
fn test_groups_count_per_group_and_penalize_once_per_agent() {
    if DEBUG_CHECKS {
        return;
    }
    for orders in [[1, 3, 1, 0, 0, 0], [1, 3, 2, 0, 0, 3]] {
        let mut p = pool(IllegalActionPolicy::Penalize(-5.0));
        let out = p.step(input(&both(orders))).unwrap();
        assert_eq!(out.rewards()[0], -5.0);
        let expected = 1 + u32::from(orders[2] == 2) + u32::from(orders[5] == 3);
        assert_eq!(illegal_count(&p), expected);
    }
}

#[test]
fn test_groups_legal_orders_run_uncounted() {
    let mut p = pool(IllegalActionPolicy::Noop);
    p.step(input(&both([2, 2, 1, 1, 0, 0]))).unwrap();
    assert_eq!(p.state(0).unwrap().last[0], [22, 11, 0]);
    assert_eq!(illegal_count(&p), 0);
}

#[test]
fn test_groups_strict_path_agrees_with_the_fallback_and_names_the_group() {
    let p = pool(IllegalActionPolicy::Noop);
    let bad = both([1, 0, 2, 0, 0, 0]);
    assert_eq!(
        p.check_actions(input(&bad)),
        Err(EngineError::IllegalAction {
            instance: 0,
            agent: 0,
            head: 2,
            group: 1,
            value: 2,
        })
    );
    assert_eq!(p.check_actions(input(&both([1, 0, 1, 1, 0, 0]))), Ok(()));
    let mut s = 0xfeed_u64;
    for _ in 0..2000 {
        let orders: [i32; HEADS] = core::array::from_fn(|h| {
            s = s.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
            i32::try_from((s >> 33) % if h % 2 == 0 { 3 } else { 4 }).unwrap()
        });
        let row = both(orders);
        let strict = p.check_actions(input(&row)).is_ok();
        let illegal =
            (0..UNITS).any(|u| orders[u * 2 + 1] > orders[u * 2] || (u == 1 && orders[u * 2] == 2));
        assert_eq!(strict, !illegal);
    }
}

#[test]
fn test_groups_partial_noop_is_identical_across_workers_and_never_stale() {
    if DEBUG_CHECKS {
        return;
    }
    let run = |workers: Option<usize>| {
        let mut p: Pool =
            EnvPool::with_masks(cfg(true), options(4, IllegalActionPolicy::Noop)).unwrap();
        if let Some(w) = workers {
            p = p
                .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
                .unwrap();
        }
        p.reset_all(Reseed::Keep).unwrap();
        // Unit 0 illegal (child), agent order differs by instance.
        let row = [both([1, 3, 1, 0, 0, 0]), both([1, 3, 1, 1, 0, 0])].concat();
        let all = [row.clone(), row].concat();
        p.step(input(&all)).unwrap();
        (0..4)
            .map(|i| p.state_hash(i).unwrap().get())
            .collect::<Vec<_>>()
    };
    let inline = run(None);
    assert_eq!(inline, run(Some(1)));
    assert_eq!(inline, run(Some(3)));
    let mut p: Pool =
        EnvPool::with_masks(cfg(true), options(1, IllegalActionPolicy::Noop)).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p.step(input(&both([1, 3, 1, 0, 0, 0]))).unwrap();
    let s = p.state(0).unwrap();
    // Agent 0's no-op type is 0, agent 1's is 1; the target stays zero.
    assert_eq!((s.last[0][0], s.last[1][0]), (0, 10));
}

fn with_illegal(p: &Pool, count: u32) -> DecodedRecord<Squad> {
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut header = *good.header();
    header.agents[0].illegal_actions = count;
    DecodedRecord::from_parts(header, *good.state()).unwrap()
}

#[test]
fn test_groups_snapshot_bound_is_steps_times_groups() {
    let mut p = pool(IllegalActionPolicy::Noop);
    for _ in 0..2 {
        p.step(input(&both([0; HEADS]))).unwrap();
    }
    let at = with_illegal(&p, 6);
    p.import(&[0], &[at]).unwrap();
    let over = with_illegal(&p, 7);
    assert!(matches!(
        p.import(&[0], &[over]),
        Err(EngineError::InvalidRecord { .. })
    ));
}

#[test]
fn test_groups_out_of_range_child_is_refused_before_legality_in_every_mode() {
    let bad = both([1, 4, 0, 0, 0, 0]);
    let check =
        |e: &EngineError| assert!(matches!(e, EngineError::DiscreteOutOfRange { .. }), "{e:?}");

    let mut sync = pool(IllegalActionPolicy::Noop);
    check(&sync.step(input(&bad)).map(|_| ()).unwrap_err());

    let mut piped =
        EnvPool::<Squad, Masked>::with_masks(cfg(false), options(4, IllegalActionPolicy::Noop))
            .unwrap()
            .pipelined(PipelinedOptions::new(NonZeroUsize::new(1).unwrap()))
            .unwrap();
    piped.reset_all(Reseed::Keep).unwrap();
    let token = piped.idle_token(0).unwrap();
    let half = [bad.clone(), bad.clone()].concat();
    let err = piped
        .send(token, input(&half))
        .map_err(|e| e.error)
        .unwrap_err();
    check(&err);

    let mut asy =
        EnvPool::<Squad, Masked>::with_masks(cfg(false), options(2, IllegalActionPolicy::Noop))
            .unwrap()
            .async_mode(AsyncOptions::new(
                NonZeroUsize::new(1).unwrap(),
                NonZeroUsize::new(1).unwrap(),
            ))
            .unwrap();
    asy.reset_all(Reseed::Keep).unwrap();
    while asy.try_recv().ok().flatten().is_some() {}
    check(&asy.send(&[0], input(&bad)).unwrap_err());
}

#[test]
fn test_groups_without_parents_fall_back_per_unit() {
    if DEBUG_CHECKS {
        return;
    }
    let flat = SquadConfig {
        partial_noop: false,
        with_box: false,
        flat: true,
    };
    let mut p: Pool = EnvPool::with_masks(flat, options(1, IllegalActionPolicy::Noop)).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    // Unit 0 illegal (target 3); units 1 and 2 legal and must keep orders.
    p.step(input(&both([1, 3, 1, 2, 2, 1]))).unwrap();
    assert_eq!(p.state(0).unwrap().last[0], [0, 12, 21]);
    assert_eq!(illegal_count(&p), 1);
    // Only unit 2 illegal: the earlier groups are untouched.
    p.step(input(&both([2, 2, 1, 0, 1, 3]))).unwrap();
    assert_eq!(p.state(0).unwrap().last[0], [22, 10, 0]);
    assert_eq!(illegal_count(&p), 2);
    // Illegal, legal, illegal: the legal unit between them keeps its order,
    // so each group is read alone, not the whole row from it on.
    p.step(input(&both([1, 3, 1, 2, 1, 3]))).unwrap();
    assert_eq!(p.state(0).unwrap().last[0], [0, 12, 0]);
    assert_eq!(illegal_count(&p), 4);
}

#[test]
fn test_groups_beside_a_continuous_head_are_refused_at_construction() {
    let with_box = SquadConfig {
        partial_noop: false,
        with_box: true,
        flat: false,
    };
    let built =
        EnvPool::<Squad, Masked>::with_masks(with_box, options(1, IllegalActionPolicy::Noop));
    assert!(matches!(
        built.map(|_| ()),
        Err(EngineError::InvalidMaskLayout {
            what: "entity groups beside continuous heads"
        })
    ));
}

#[test]
fn test_groups_mask_head_shape_reports_each_block() {
    let p = pool(IllegalActionPolicy::Noop);
    let root = p.mask_head_shape(2).unwrap();
    assert_eq!(
        (root.offset, root.size, root.parent_sizes),
        (15, 3, [None, None])
    );
    let child = p.mask_head_shape(3).unwrap();
    assert_eq!(
        (child.offset, child.size, child.parent_sizes),
        (18, 4, [Some(3), None])
    );
    assert!(p.mask_head_shape(HEADS).is_none());
}

fn obs_of(bytes: &[u8]) -> Vec<i32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect()
}

#[test]
fn test_groups_fall_back_the_same_in_pipelined_and_async() {
    if DEBUG_CHECKS {
        return;
    }
    let orders = both([1, 3, 1, 0, 2, 2]);
    let mut piped =
        EnvPool::<Squad, Masked>::with_masks(cfg(false), options(4, IllegalActionPolicy::Noop))
            .unwrap()
            .pipelined(PipelinedOptions::new(NonZeroUsize::new(1).unwrap()))
            .unwrap();
    piped.reset_all(Reseed::Keep).unwrap();
    let token = piped.idle_token(0).unwrap();
    piped
        .send(token, input(&[orders.clone(), orders.clone()].concat()))
        .map_err(|e| e.error)
        .unwrap();
    let done = piped.recv().unwrap();
    assert_eq!(obs_of(done.observation(0).unwrap())[..3], [0, 10, 22]);

    let mut asy =
        EnvPool::<Squad, Masked>::with_masks(cfg(false), options(2, IllegalActionPolicy::Noop))
            .unwrap()
            .async_mode(AsyncOptions::new(
                NonZeroUsize::new(1).unwrap(),
                NonZeroUsize::new(1).unwrap(),
            ))
            .unwrap();
    asy.reset_all(Reseed::Keep).unwrap();
    while asy.try_recv().ok().flatten().is_some() {}
    asy.send(&[0], input(&orders)).unwrap();
    let done = asy.recv().unwrap();
    assert_eq!(obs_of(done.observation(0, 0).unwrap())[..3], [0, 10, 22]);
}

/// A seeded mix of legal and illegal orders for every agent of `n` instances.
fn orders(n: usize, s: &mut u64) -> Vec<i32> {
    (0..n * 2 * HEADS)
        .map(|k| {
            *s = s.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
            i32::try_from((*s >> 33) % if k % 2 == 0 { 3 } else { 4 }).unwrap()
        })
        .collect()
}

fn fnv(hash: u64, word: u64) -> u64 {
    word.to_le_bytes().iter().fold(hash, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Folds what a step left: state hash, illegal counts and the mask row.
fn fold_step(mut hash: u64, p: &Pool, masks: &[Bool8]) -> u64 {
    for i in 0..4 {
        hash = fnv(hash, p.state_hash(i).unwrap().get());
        let record = DecodedRecord::from(p.snapshot(i).unwrap());
        for row in &record.header().agents {
            hash = fnv(hash, u64::from(row.illegal_actions));
        }
    }
    masks.iter().fold(hash, |h, m| fnv(h, u64::from(m.get())))
}

const STEPS: usize = 64;
const SPLIT: usize = 23;
const FNV_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

fn golden_pool(workers: Option<usize>) -> Pool {
    let mut p: Pool =
        EnvPool::with_masks(cfg(true), options(4, IllegalActionPolicy::Noop)).unwrap();
    if let Some(w) = workers {
        p = p
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap();
    }
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn golden_run(workers: Option<usize>) -> u64 {
    let mut p = golden_pool(workers);
    let (mut s, mut hash) = (0x7e55_u64, FNV_BASIS);
    for _ in 0..STEPS {
        let row = orders(4, &mut s);
        let masks = p.step(input(&row)).unwrap().masks().to_vec();
        hash = fold_step(hash, &p, &masks);
    }
    hash
}

/// Fixture golden of design section "Tests": any change to the group
/// fallback, the row layout or the mask bytes moves it.
const SQUAD_GOLDEN: u64 = 0x8b50_5ff9_f26f_6750;

#[test]
fn test_groups_golden_is_identical_across_workers() {
    if DEBUG_CHECKS {
        return;
    }
    let inline = golden_run(None);
    assert_eq!(inline, SQUAD_GOLDEN, "{inline:#x}");
    assert_eq!(golden_run(Some(1)), inline);
    assert_eq!(golden_run(Some(3)), inline);
}

#[test]
fn test_groups_continue_through_snapshot_and_restore_on_the_same_trajectory() {
    if DEBUG_CHECKS {
        return;
    }
    let mut p = golden_pool(None);
    let (mut s, mut hash) = (0x7e55_u64, FNV_BASIS);
    let mut records = Vec::new();
    let mut saved = (0, 0);
    for step in 0..STEPS {
        if step == SPLIT {
            records = (0..4).map(|i| p.snapshot(i).unwrap()).collect();
            saved = (s, hash);
        }
        let row = orders(4, &mut s);
        let masks = p.step(input(&row)).unwrap().masks().to_vec();
        hash = fold_step(hash, &p, &masks);
    }
    let whole = hash;
    // Restored into a fresh pool whose masks must be re-derived from state.
    let mut q = golden_pool(None);
    q.restore_all(&records).unwrap();
    (s, hash) = saved;
    for _ in SPLIT..STEPS {
        let row = orders(4, &mut s);
        let masks = q.step(input(&row)).unwrap().masks().to_vec();
        hash = fold_step(hash, &q, &masks);
    }
    assert_eq!(hash, whole);
}

#[test]
fn test_groups_header_layout_is_unchanged() {
    assert_eq!(<Squad as Environment>::MAX_AGENTS, 2);
    assert_eq!(<RecordHeader<2, NoAdaptive> as RecordLayout>::AGENTS, 2);
    let _ = Bool8::TRUE;
}
