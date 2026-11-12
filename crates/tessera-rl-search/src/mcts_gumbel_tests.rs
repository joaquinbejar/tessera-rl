// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Gumbel phase step driven by hand (design note for #69, r3): `want`
//! is pure, `advance_phase` changes nothing except right after the
//! backup of a phase's last visit, and a halving with pending virtual
//! loss is a typed error that the call rolls back.

use core::num::NonZeroU32;

use tessera_rl_engine::{EnvPool, Masked, PoolOptions, Record, Reseed};
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_num::identity::Seed;

use super::{Mcts, PuctConfig};
use crate::gumbel::{GumbelConfig, RootSelection};
use crate::{Evaluator, EvaluatorError, SearchError, SearchOptions, SearchPool};

struct Uniform;

impl Evaluator<Connect4> for Uniform {
    fn evaluate(
        &mut self,
        leaves: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        for (i, r) in leaves.iter().enumerate() {
            let h = r.hash().get();
            for a in 0..7 {
                priors[i * 7 + a] = f64::from(1 + ((h >> (8 * a)) & 0xf) as u32);
            }
            let s: f64 = priors[i * 7..i * 7 + 7].iter().sum();
            for p in &mut priors[i * 7..i * 7 + 7] {
                *p /= s;
            }
            values[i] = f64::from((h >> 48) as u32 & 0xff) / 127.5 - 1.0;
        }
        Ok(())
    }
}

fn mcts(sims: u32, batch: u32) -> Mcts<Connect4, Masked> {
    let mut p = EnvPool::<Connect4, Masked>::with_masks(
        Connect4Config {},
        PoolOptions::new(NonZeroU32::MIN, Seed::new(4)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let o = SearchOptions::new(NonZeroU32::new(1_024).unwrap(), NonZeroU32::new(8).unwrap());
    let pool = SearchPool::new(
        Connect4Config {},
        o,
        &p.snapshot(0).unwrap(),
        EnvPool::with_masks,
    )
    .unwrap();
    let c = PuctConfig {
        simulations: sims,
        batch_size: batch,
        root: RootSelection::Gumbel(GumbelConfig::default()),
        ..PuctConfig::default()
    };
    Mcts::new(pool, c).unwrap()
}

type State = (u32, u32, u32, u32, bool, Vec<u32>);

fn state(m: &Mcts<Connect4, Masked>) -> (u64, State) {
    let g = m.gumbel.as_ref().unwrap();
    (m.pool.tree_hash(), g.snapshot())
}

/// Starts a call by hand: the root, its value and the Gumbel setup.
fn begin(m: &mut Mcts<Connect4, Masked>) -> super::CallStart {
    let start = m.call_begin().unwrap();
    m.evaluate_root(&mut Uniform).unwrap();
    let mut rng = start.rng.clone();
    m.prepare_root_priors(&mut rng);
    start
}

/// One batch, with `want` asked twice and `advance_phase` called at
/// every wrong time: before collection, after staging (virtual loss
/// pending), and a second time after the right one.
fn batch(m: &mut Mcts<Connect4, Masked>, done: u32) -> Option<u32> {
    let before = state(m);
    let want = m.want(done)?;
    assert_eq!(m.want(done), Some(want), "want is pure");
    assert_eq!(state(m), before, "want changes nothing");
    m.advance_phase().unwrap();
    assert_eq!(state(m), before, "advance_phase before collection");
    let n = m.collect_and_stage(want).unwrap();
    let k = m.eval_len();
    if k > 0 {
        m.evaluate_staged(&mut Uniform, k).unwrap();
        m.validate_staged(k).unwrap();
    }
    let staged = state(m);
    m.advance_phase().unwrap();
    assert_eq!(state(m), staged, "advance_phase with leaves pending");
    m.commit_and_backup(k).unwrap();
    m.advance_phase().unwrap();
    let after = state(m);
    m.advance_phase().unwrap();
    assert_eq!(state(m), after, "a second advance_phase");
    Some(done + n.max(1))
}

#[test]
fn test_gumbel_want_twice_and_advance_phase_at_a_wrong_time_change_nothing() {
    let mut m = mcts(64, 8);
    let start = begin(&mut m);
    let mut done = 0;
    let mut phases = vec![m.gumbel.as_ref().unwrap().snapshot().0];
    while let Some(d) = batch(&mut m, done) {
        done = d;
        phases.push(m.gumbel.as_ref().unwrap().snapshot().0);
    }
    phases.dedup();
    // 16 > 7 legal: m' = 7, P = 3, then the last survivor's phase.
    assert_eq!(phases, vec![0, 1, 2, 3], "every phase was reached");
    m.finish_root().unwrap();
    m.call_commit(start.calls, start.rng);
    // The same call through `run` gives the same tree.
    let mut r = mcts(64, 8);
    r.run(&mut Uniform).unwrap();
    assert_eq!(r.pool.tree_hash(), m.pool.tree_hash());
    assert_eq!(r.improved_policy(), m.improved_policy());
}

#[test]
fn test_gumbel_halving_with_pending_virtual_loss_is_rejected_and_rolled_back() {
    let mut m = mcts(64, 8);
    let tree = m.pool.tree_hash();
    let _start = begin(&mut m);
    let mut done = 0;
    // Run to the end of phase 0 (21 visits), stopping before its
    // halving.
    loop {
        let want = m.want(done).unwrap();
        let n = m.collect_and_stage(want).unwrap();
        let k = m.eval_len();
        if k > 0 {
            m.evaluate_staged(&mut Uniform, k).unwrap();
            m.validate_staged(k).unwrap();
        }
        m.commit_and_backup(k).unwrap();
        done += n.max(1);
        if m.gumbel.as_ref().unwrap().phase_complete() {
            break;
        }
        m.advance_phase().unwrap();
    }
    assert_eq!(done, 21);
    let root = m.pool.root;
    m.add_virtual_loss(root, 3).unwrap();
    // A debug build stops at the `debug_assert!`; a release build returns
    // the typed error. Either way nothing is halved.
    let before = m.gumbel.as_ref().unwrap().snapshot();
    let r = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| m.advance_phase()));
    if cfg!(debug_assertions) {
        assert!(r.is_err(), "the debug assertion fires");
    } else {
        assert!(matches!(
            r.unwrap(),
            Err(SearchError::GumbelInvariant { .. })
        ));
    }
    assert_eq!(m.gumbel.as_ref().unwrap().snapshot(), before);
    m.call_rollback();
    assert_eq!(m.pool.tree_hash(), tree);
    assert!(m.root_value().is_none(), "v_root rolled back");
}
