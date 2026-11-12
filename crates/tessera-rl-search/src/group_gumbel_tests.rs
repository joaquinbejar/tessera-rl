// SPDX-License-Identifier: MIT OR Apache-2.0

//! A group call under the Gumbel root driven round by round: between the
//! stages, asking every slot's `want` twice and calling `advance_phase`
//! again change nothing, and the call equals [`SearchGroup::run`].

use core::num::NonZeroU32;

use tessera_rl_engine::{EnvPool, Masked, PoolOptions, Record, Reseed};
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_num::identity::Seed;

use super::SearchGroup;
use crate::gumbel::{GumbelConfig, RootSelection};
use crate::{DEFAULT_GROUP_BUDGET, Evaluator, EvaluatorError, Mcts, PuctConfig};
use crate::{SearchOptions, SearchPool};

struct Hashed;

impl Evaluator<Connect4> for Hashed {
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

fn group() -> SearchGroup<Connect4, Masked> {
    let mut p = EnvPool::<Connect4, Masked>::with_masks(
        Connect4Config {},
        PoolOptions::new(NonZeroU32::MIN, Seed::new(4)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let root = p.snapshot(0).unwrap();
    let slots = [(1, 30, 8), (16, 64, 8), (4, 23, 3)]
        .into_iter()
        .map(|(m, n, b)| {
            let o =
                SearchOptions::new(NonZeroU32::new(1_024).unwrap(), NonZeroU32::new(8).unwrap());
            let pool = SearchPool::new(Connect4Config {}, o, &root, EnvPool::with_masks).unwrap();
            let c = PuctConfig {
                simulations: n,
                batch_size: b,
                root: RootSelection::Gumbel(GumbelConfig {
                    m,
                    ..GumbelConfig::default()
                }),
                ..PuctConfig::default()
            };
            Mcts::new(pool, c).unwrap()
        })
        .collect();
    SearchGroup::new(&Connect4Config {}, slots, DEFAULT_GROUP_BUDGET).unwrap()
}

type Snap = Vec<(u64, Option<u32>, (u32, u32, u32, u32, bool, Vec<u32>))>;

fn snap(g: &SearchGroup<Connect4, Masked>) -> Snap {
    (0..g.len())
        .map(|i| {
            let m = &g.slots[i];
            (
                m.pool().tree_hash(),
                m.want(g.done[i]),
                m.gumbel_snapshot().unwrap(),
            )
        })
        .collect()
}

#[test]
fn test_group_gumbel_want_twice_and_extra_advance_phase_change_nothing() {
    let mut g = group();
    g.participants.extend(0..g.len());
    for i in 0..g.len() {
        g.starts[i] = Some(g.slots[i].call_begin().unwrap());
    }
    g.prologue(&mut Hashed).unwrap();
    let mut rounds = 0;
    loop {
        let before = snap(&g);
        assert_eq!(snap(&g), before, "want asked twice");
        for i in 0..g.len() {
            g.slots[i].advance_phase().unwrap();
        }
        assert_eq!(snap(&g), before, "advance_phase between rounds");
        if !g.round(&mut Hashed).unwrap() {
            break;
        }
        rounds += 1;
    }
    assert!(rounds > 8);
    for i in 0..g.len() {
        g.slots[i].finish_root().unwrap();
        let start = g.starts[i].take().unwrap();
        let rng = g.rngs[i].take().unwrap();
        g.slots[i].call_commit(start.calls, rng);
    }
    let mut r = group();
    r.run(&mut Hashed).unwrap();
    for i in 0..g.len() {
        assert_eq!(
            g.slots[i].pool().tree_hash(),
            r.slots[i].pool().tree_hash(),
            "slot {i}"
        );
        assert_eq!(g.slots[i].improved_policy(), r.slots[i].improved_policy());
    }
}
