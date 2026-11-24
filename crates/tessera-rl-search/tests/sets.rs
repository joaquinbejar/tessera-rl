// SPDX-License-Identifier: MIT OR Apache-2.0

//! Search over `Set` observations (ADR-0082 point 7 as amended, #490) on
//! `tessera/Gather-v1`: the evaluator's rows hold the set's item and mask
//! leaves exactly as the live pool exports them, absent slots zero (the
//! engine's zeroing pass on the scratch pool's re-observe), and the layout
//! names the set's leaves. Run also with `tessera-rl-engine/debug-checks`.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::{Gather, GatherConfig};
use tessera_rl_core::space::DType;
use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, NoMasks, PoolOptions, Record, Reseed};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{
    Evaluator, EvaluatorError, Eviction, LeafBatch, Mcts, PuctConfig, SearchGroup, SearchOptions,
    SearchPool,
};

/// Three items on the board: slots 3 to 7 are absent from the start, and
/// collected items empty more slots as the search goes.
fn cfg() -> GatherConfig {
    GatherConfig {
        items: 3,
        ..GatherConfig::default()
    }
}

fn live() -> EnvPool<Gather, NoMasks> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(9))
    };
    let mut p = EnvPool::<Gather, NoMasks>::new(cfg(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn root(moves: &[i32]) -> Record<Gather> {
    let mut p = live();
    for &a in moves {
        p.step(ActionInput {
            discrete: &[a],
            continuous: &[],
        })
        .unwrap();
    }
    p.snapshot(0).unwrap()
}

fn present(p: &EnvPool<Gather, NoMasks>) -> usize {
    p.state(0)
        .unwrap()
        .present
        .iter()
        .filter(|f| f.get())
        .count()
}

/// Moves that walk greedily to an item and stop one move before
/// collecting it: the search from there collects it inside the tree.
fn before_collecting() -> Vec<i32> {
    let mut p = live();
    let mut moves = Vec::new();
    for _ in 0..40 {
        let before = p.snapshot(0).unwrap();
        let distance = |p: &EnvPool<Gather, NoMasks>| {
            let s = p.state(0).unwrap();
            let [c, r] = s.agent;
            s.items
                .iter()
                .zip(s.present.iter())
                .filter(|(_, f)| f.get())
                .map(|(i, _)| (i[0] - c).abs() + (i[1] - r).abs())
                .min()
                .unwrap_or(0)
        };
        let mut best = (i16::MAX, 0, false);
        for a in 0..4 {
            p.restore(0, &before).unwrap();
            p.step(ActionInput {
                discrete: &[a],
                continuous: &[],
            })
            .unwrap();
            let collected = present(&p) < 3;
            let d = distance(&p);
            if collected || d < best.0 {
                best = (d, a, collected);
            }
            if collected {
                break;
            }
        }
        p.restore(0, &before).unwrap();
        if best.2 {
            return moves;
        }
        p.step(ActionInput {
            discrete: &[best.1],
            continuous: &[],
        })
        .unwrap();
        moves.push(best.1);
    }
    panic!("no item reached in 40 moves");
}

fn pool(capacity: u32) -> SearchPool<Gather, NoMasks> {
    let mut o = SearchOptions::new(
        NonZeroU32::new(capacity).unwrap(),
        NonZeroU32::new(8).unwrap(),
    );
    o.eviction = Eviction::Strict;
    SearchPool::new(cfg(), o, &root(&before_collecting()), EnvPool::new).unwrap()
}

/// The live pool's exported leaves of instance 0, concatenated.
fn live_row(p: &EnvPool<Gather, NoMasks>) -> Vec<u8> {
    let out = p.outputs().unwrap();
    (0..tessera_rl_engine::MAX_LEAVES)
        .map_while(|l| out.observation(l).map(<[u8]>::to_vec))
        .flatten()
        .collect()
}

/// Keeps every evaluated leaf's record and row; uniform priors.
#[derive(Default)]
struct Keep {
    seen: Vec<(Record<Gather>, Vec<u8>)>,
}

impl Evaluator<Gather> for Keep {
    fn evaluate(
        &mut self,
        _: &[Record<Gather>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        priors.fill(0.25);
        values.fill(0.0);
        Ok(())
    }

    fn evaluate_batch(
        &mut self,
        batch: &mut LeafBatch<'_, Gather>,
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        let rows = batch.observations().to_vec();
        let n = batch.records().len();
        let width = rows.len() / n;
        for (i, r) in batch.records().iter().enumerate() {
            self.seen
                .push((*r, rows[i * width..(i + 1) * width].to_vec()));
        }
        priors.fill(0.25);
        values.fill(0.0);
        Ok(())
    }
}

#[test]
fn test_sets_search_builds_and_names_the_set_leaves() {
    let s = pool(64);
    let leaves = s.observation_leaves();
    let names: Vec<&str> = leaves.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["agent", "items", "items_mask"]);
    assert_eq!(leaves[1].shape, [8, 2]);
    assert_eq!(
        (leaves[2].dtype, leaves[2].shape.as_slice()),
        (DType::Bool, &[8][..])
    );
    let sets = s.observation_sets();
    assert_eq!(sets.len(), 1);
    assert_eq!((sets[0].path.as_str(), sets[0].max), ("items", 8));
    assert_eq!((sets[0].items.as_slice(), sets[0].mask), (&[1][..], 2));
    // Offsets are contiguous and cover one row.
    let end = leaves.last().map(|l| l.offset + l.bytes).unwrap();
    assert_eq!(end, live_row(&live()).len());
}

/// Rev's required test (#490 design review): for every evaluated leaf,
/// the evaluator's row equals the live pool's exported leaves for the
/// same record, byte for byte, including states with absent slots, whose
/// item rows are zero.
#[test]
fn test_sets_evaluator_rows_equal_the_live_pool_with_absent_slots_zero() {
    let s = pool(512);
    let mut c = PuctConfig::default();
    c.simulations = 96;
    c.batch_size = 8;
    let mut m = Mcts::new(s, c).unwrap();
    let mut keep = Keep::default();
    m.run(&mut keep).unwrap();
    assert!(keep.seen.len() > 30, "{} leaves", keep.seen.len());

    let (items, mask) = {
        let l = m.pool().observation_leaves();
        (l[1].clone(), l[2].clone())
    };
    let mut p = live();
    let mut absent_rows = 0;
    let mut fewer = false;
    for (record, row) in &keep.seen {
        p.restore(0, record).unwrap();
        assert_eq!(
            row,
            &live_row(&p),
            "row of a leaf differs from the live pool"
        );
        let present = &row[mask.offset..mask.offset + mask.bytes];
        let item_bytes = items.bytes / 8;
        for (slot, &flag) in present.iter().enumerate() {
            if flag == 0 {
                absent_rows += 1;
                let at = items.offset + slot * item_bytes;
                assert!(
                    row[at..at + item_bytes].iter().all(|&b| b == 0),
                    "absent slot {slot} not zero"
                );
            }
        }
        fewer |= present.iter().filter(|&&f| f != 0).count() < 3;
    }
    assert!(absent_rows > 0);
    assert!(
        fewer,
        "no leaf with a collected item: the search never emptied a slot"
    );
}

#[test]
fn test_sets_group_bound_counts_the_set_leaves() {
    let s = pool(64);
    let mut c = PuctConfig::default();
    c.batch_size = 8;
    let bound = SearchGroup::<Gather, NoMasks>::bound_bytes(
        &cfg(),
        &[(
            SearchOptions::new(NonZeroU32::new(64).unwrap(), NonZeroU32::new(8).unwrap()),
            c,
        )],
    );
    assert!(bound.is_some(), "a set observation has a bound");
    drop(s);
}
