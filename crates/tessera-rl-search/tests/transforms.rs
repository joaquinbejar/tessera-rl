// SPDX-License-Identifier: MIT OR Apache-2.0

//! Search on a pool with observation transforms (ADR-0078 item 7): the
//! scratch pool has the live pool's transforms and frozen statistics,
//! every node keeps its frame stacks, so the evaluator sees exactly the
//! observations the live pool would produce.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers unwrap and index: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Record, Reseed, TransformContext,
    Transforms,
};
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{
    Evaluator, EvaluatorError, LeafBatch, Mcts, PuctConfig, SearchError, SearchGroup,
    SearchOptions, SearchPool,
};

type Live = EnvPool<Connect4, Masked>;

fn transforms() -> Transforms {
    Transforms::new()
        .normalize("board", 3.0)
        .frame_stack("board", 3)
}

fn live() -> Live {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(9))
    };
    let mut p = Live::with_masks(Connect4Config {}, options)
        .unwrap()
        .with_transforms(transforms())
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

/// Plays `moves` (columns) on `p`, then takes a barrier, so the statistics
/// are not the identity.
fn play(p: &mut Live, moves: &[i32]) {
    for &c in moves {
        let me = p.state(0).unwrap().to_move();
        let mut act = [0, 0];
        act[me] = c;
        p.step(ActionInput {
            discrete: &act,
            continuous: &[],
        })
        .unwrap();
    }
    p.epoch_barrier().unwrap();
}

fn root_of(p: &Live) -> (Record<Connect4>, Vec<u8>, TransformContext) {
    let mut bytes = vec![0; p.transform_state_bytes()];
    p.snapshot_transforms(0, &mut bytes).unwrap();
    (
        p.snapshot(0).unwrap(),
        bytes,
        p.pool_context().unwrap().adaptive.normalization,
    )
}

fn options() -> SearchOptions {
    SearchOptions::new(NonZeroU32::new(64).unwrap(), NonZeroU32::new(4).unwrap())
}

fn search(
    root: &Record<Connect4>,
    bytes: &[u8],
    context: TransformContext,
) -> SearchPool<Connect4, Masked> {
    let mut s =
        SearchPool::new_with_transforms(Connect4Config {}, options(), root, bytes, |c, o| {
            EnvPool::with_masks(c, o)?.with_transforms(transforms())
        })
        .unwrap();
    s.set_transform_context(context).unwrap();
    s
}

/// The live observation of the mover after playing `column` from the
/// position of `p` (a copy).
fn live_after(moves: &[i32], column: i32) -> (Vec<u8>, Vec<u8>) {
    let mut p = live();
    play(&mut p, moves);
    let me = p.state(0).unwrap().to_move();
    let mut act = [0, 0];
    act[me] = column;
    p.step(ActionInput {
        discrete: &act,
        continuous: &[],
    })
    .unwrap();
    let mut bytes = vec![0; p.transform_state_bytes()];
    p.snapshot_transforms(0, &mut bytes).unwrap();
    (p.outputs().unwrap().observation(0).unwrap().to_vec(), bytes)
}

#[test]
fn test_search_transforms_children_keep_the_live_frame_stacks() {
    let moves = [3, 2, 4];
    let mut p = live();
    play(&mut p, &moves);
    let (root, bytes, context) = root_of(&p);
    let mut s = search(&root, &bytes, context.clone());
    assert_eq!(s.transform_state_bytes(), p.transform_state_bytes());
    let mut out = [s.root(); 2];
    s.expand(&[(s.root(), 1), (s.root(), 5)], &mut out).unwrap();
    for (h, column) in out.into_iter().zip([1, 5]) {
        let (obs, live_bytes) = live_after(&moves, column);
        assert_eq!(s.transform_bytes(h).unwrap(), &live_bytes[..]);
        // The node continues exactly in a live pool with the same context.
        let mut q = live();
        let mut pc = q.pool_context().unwrap();
        pc.adaptive.normalization = context.clone();
        q.restore_pool_context(pc, None).unwrap();
        q.restore_with_transforms(0, s.record(h).unwrap(), s.transform_bytes(h).unwrap())
            .unwrap();
        assert_eq!(q.outputs().unwrap().observation(0).unwrap(), &obs[..]);
    }
}

/// Records the first batch's observations.
struct Capture(Option<Vec<u8>>);

impl Evaluator<Connect4> for Capture {
    fn evaluate(
        &mut self,
        _: &[Record<Connect4>],
        _: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        Ok(())
    }

    fn evaluate_batch(
        &mut self,
        batch: &mut LeafBatch<'_, Connect4>,
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        if self.0.is_none() {
            self.0 = Some(batch.observations().to_vec());
        }
        let n = priors.len() / 7;
        priors.fill(1.0 / 7.0);
        values[..n].fill(0.0);
        Ok(())
    }
}

#[test]
fn test_search_transforms_evaluator_sees_the_live_root_observation() {
    let mut p = live();
    play(&mut p, &[3, 3]);
    let (root, bytes, context) = root_of(&p);
    let s = search(&root, &bytes, context);
    let mut cfg = PuctConfig::default();
    cfg.simulations = 8;
    cfg.batch_size = 4;
    let mut m = Mcts::new(s, cfg).unwrap();
    let mut capture = Capture(None);
    m.run(&mut capture).unwrap();
    let seen = capture.0.unwrap();
    // The mover's row of the live pool, transformed and stacked.
    let all = p.outputs().unwrap().observation(0).unwrap();
    let per = all.len() / 2;
    let me = p.state(0).unwrap().to_move();
    assert_eq!(&seen[..per], &all[me * per..(me + 1) * per]);
}

#[test]
fn test_search_transforms_refusals() {
    let mut p = live();
    play(&mut p, &[3]);
    let (root, bytes, context) = root_of(&p);
    // A plain root on a stacked scratch pool.
    let plain = SearchPool::new(Connect4Config {}, options(), &root, |c, o| {
        EnvPool::with_masks(c, o)?.with_transforms(transforms())
    });
    assert!(matches!(
        plain.err(),
        Some(SearchError::StackedRoot { found: 0, .. })
    ));
    let built =
        SearchPool::new_with_transforms(Connect4Config {}, options(), &root, &bytes, |c, o| {
            EnvPool::with_masks(c, o)?.with_transforms(transforms())
        })
        .unwrap();
    // The scratch pool is frozen from construction, before any context.
    assert!(built.transform_context().unwrap().frozen);
    let mut s = search(&root, &bytes, context.clone());
    assert!(matches!(
        s.set_root(&root),
        Err(SearchError::StackedRoot { .. })
    ));
    assert!(matches!(
        s.set_root_with_transforms(&root, &bytes[1..]),
        Err(SearchError::StackedRoot { .. })
    ));
    s.set_root_with_transforms(&root, &bytes).unwrap();
    // A context of another transform list.
    let mut other = context;
    other.identity ^= 1;
    assert!(matches!(
        s.set_transform_context(other),
        Err(SearchError::TransformIdentity { .. })
    ));
}

#[test]
fn test_search_transforms_bound_counts_the_frame_stacks() {
    let shapes = [(options(), PuctConfig::default())];
    let plain = SearchGroup::<Connect4, Masked>::bound_bytes(&Connect4Config {}, &shapes).unwrap();
    let stacked = SearchGroup::<Connect4, Masked>::bound_bytes_with(
        &Connect4Config {},
        &shapes,
        &transforms(),
    )
    .unwrap();
    let bytes = live().transform_state_bytes();
    let s = u64::try_from(bytes).unwrap();
    // Every node (tree and journal) and the load and child rows hold S.
    assert!(
        stacked >= plain + 2 * 64 * s + 2 * 4 * s,
        "{stacked} vs {plain}"
    );
    let per = SearchPool::<Connect4, Masked>::bytes_per_node_with(7, bytes);
    assert_eq!(
        per,
        SearchPool::<Connect4, Masked>::bytes_per_node(7) + 2 * s
    );
}

/// Fails on its third batch, after the first expansion committed
/// children.
struct FailSecond(u32);

impl Evaluator<Connect4> for FailSecond {
    fn evaluate(
        &mut self,
        _: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        self.0 += 1;
        if self.0 == 3 {
            return Err(EvaluatorError::Failed {
                reason: "injected".to_owned(),
            });
        }
        priors.fill(1.0 / 7.0);
        values.fill(0.0);
        Ok(())
    }
}

/// A failed call rolls the frame stacks back with the rest of the tree.
#[test]
fn test_search_transforms_rollback_restores_frame_stacks() {
    let mut p = live();
    play(&mut p, &[3, 2]);
    let (root, bytes, context) = root_of(&p);
    let s = search(&root, &bytes, context);
    let mut cfg = PuctConfig::default();
    cfg.simulations = 16;
    cfg.batch_size = 4;
    let mut m = Mcts::new(s, cfg).unwrap();
    let before = m.pool().tree_hash();
    assert!(m.run(&mut FailSecond(0)).is_err());
    assert_eq!(m.pool().tree_hash(), before);
}
