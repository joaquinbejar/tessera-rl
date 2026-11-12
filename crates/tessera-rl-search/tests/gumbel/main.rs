// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Gumbel root (design note for #69, step G2): an independent slow
//! reference compared bit for bit, the `m = 1` anchor against forced-root
//! PUCT, the sign of `q` when the child's mover differs and when the same
//! player moves again, a second call on the same root, the root value
//! after a reroot, group-equals-solo (ADR-0045 section 4), goldens at 1,
//! 2, 7 and 64 workers and under eviction, rollback, and the refusals.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::float_cmp,
    clippy::manual_midpoint,
    reason = "test helpers: a failure is the assertion"
)]

mod env;
mod reference;

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::RecordLayout;
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Record, Reseed, SyncOptions,
};
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_num::identity::{RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_search::{
    DEFAULT_GROUP_BUDGET, Evaluator, EvaluatorError, Eviction, GumbelConfig, LeafBatch, Mcts,
    PuctConfig, RootNoise, RootSelection, SearchError, SearchGroup, SearchOptions, SearchPool,
};

use env::{ACTIONS, Ladder, LadderConfig};

type C4 = Mcts<Connect4, Masked>;
type Lad = Mcts<Ladder, Masked>;

fn gumbel(simulations: u32, batch: u32, m: u32) -> PuctConfig {
    let mut g = GumbelConfig::default();
    g.m = m;
    let mut c = PuctConfig::default();
    c.simulations = simulations;
    c.batch_size = batch;
    c.root = RootSelection::Gumbel(g);
    c
}

// ---------------------------------------------------------------- Ladder

/// Root priors (weights over 20): action 2 legal with prior 0, action 8
/// illegal, so 7 actions can be considered.
const WEIGHTS: [f64; 9] = [4.0, 3.0, 0.0, 2.0, 3.0, 1.0, 5.0, 2.0, 0.0];
/// The value every visit of root action `a` backs up, from the root
/// mover's view.
const H: [f64; 9] = [0.5, -0.25, 0.75, 0.125, -0.5, 0.25, -0.125, 0.375, 0.0];
const V_ROOT: f64 = 0.0625;

/// Root priors `WEIGHTS / 20`; elsewhere uniform. The value is `H` of
/// the leaf's first action (`V_ROOT` at the root) from the root mover's
/// (agent 0's) view, so from the leaf mover's view it is negated when
/// that mover is agent 1.
#[derive(Default)]
struct LadderEval {
    rows: u32,
}

impl Evaluator<Ladder> for LadderEval {
    fn evaluate(
        &mut self,
        leaves: &[Record<Ladder>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        let w = ACTIONS as usize;
        for (i, r) in leaves.iter().enumerate() {
            self.rows += 1;
            let s = r.state();
            let row = &mut priors[i * w..i * w + w];
            if s.depth == 0 {
                for (p, wt) in row.iter_mut().zip(WEIGHTS) {
                    *p = wt / 20.0;
                }
            } else {
                row.fill(0.125);
                row[8] = 0.0;
            }
            let h = if s.depth == 0 {
                V_ROOT
            } else {
                H[s.first as usize - 1]
            };
            values[i] = if s.mover == 0 { h } else { -h };
        }
        Ok(())
    }
}

fn ladder_record(alternate: u32) -> Record<Ladder> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(5))
    };
    let mut p = EnvPool::<Ladder, Masked>::with_masks(LadderConfig { alternate }, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p.snapshot(0).unwrap()
}

fn ladder(alternate: u32, cfg: PuctConfig) -> Lad {
    let o = SearchOptions::new(NonZeroU32::new(512).unwrap(), NonZeroU32::new(8).unwrap());
    let pool = SearchPool::new(
        LadderConfig { alternate },
        o,
        &ladder_record(alternate),
        EnvPool::with_masks,
    )
    .unwrap();
    Mcts::new(pool, cfg).unwrap()
}

/// The `g` of call `call` (0 for the first) on the root, from its
/// identity on `StreamId::SEARCH`, by the reference's own variate.
fn reference_g<E: tessera_rl_core::Environment>(
    root: &Record<E>,
    call: u32,
    actions: usize,
) -> Vec<f64>
where
    E::Header: RecordLayout,
{
    let id = root.header().engine().identity;
    let id = RngIdentity::from_parts(
        id.seed(),
        id.branch_salt(),
        id.logical_id(),
        id.episode_index(),
        call,
    );
    let mut rng = StepRng::new(&RngAuthority::engine_only(), &id, StreamId::SEARCH);
    (0..actions)
        .map(|_| reference::gumbel_variate(rng.next_u64()))
        .collect()
}

/// The root edges of a Ladder tree: visits, values, priors, legality.
fn root_edges(m: &Lad) -> (Vec<u32>, Vec<f64>, Vec<f64>, Vec<bool>) {
    let root = m.pool().root();
    let mut out = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for a in 0..ACTIONS {
        let s = m.pool().edge_stats(root, a).unwrap();
        out.0.push(s.visits);
        out.1.push(s.value);
        out.2.push(s.prior);
        out.3.push(a != env::ILLEGAL);
    }
    out
}

fn observed(m: &mut Lad) -> reference::Output {
    let (visits, values, ..) = root_edges(m);
    reference::Output {
        visits,
        values: values.iter().map(|v| v.to_bits()).collect(),
        considered: m.considered().unwrap().to_vec(),
        policy: m
            .improved_policy()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect(),
        winner: m.select_action(0.0).unwrap(),
    }
}

/// The reference's prediction for the next call of `m` with `n`
/// simulations and `m_cfg` actions, from the tree's current root.
fn predict(m: &Lad, call: u32, n: u32, m_cfg: u32, v_root: f64) -> reference::Output {
    let (visits, values, priors, legal) = root_edges(m);
    let root = *m.pool().record(m.pool().root()).unwrap();
    let g = reference_g(&root, call, ACTIONS as usize);
    let d = GumbelConfig::default();
    reference::run(&reference::Input {
        priors: &priors,
        legal: &legal,
        g: &g,
        m: m_cfg,
        n,
        c_visit: d.c_visit,
        c_scale: d.c_scale,
        v_root,
        visits,
        values,
        h: &H,
    })
}

fn check_against_reference(alternate: u32, m_cfg: u32, n: u32, batch: u32) {
    let mut m = ladder(alternate, gumbel(n, batch, m_cfg));
    // The root priors are written by the first call: predict from a tree
    // evaluated with the same priors (the root alone).
    let mut probe = ladder(alternate, gumbel(1, 1, 1));
    probe.run(&mut LadderEval::default()).unwrap();
    let (_, _, priors, legal) = root_edges(&probe);
    let root = *m.pool().record(m.pool().root()).unwrap();
    let g = reference_g(&root, 0, ACTIONS as usize);
    let d = GumbelConfig::default();
    let want = reference::run(&reference::Input {
        priors: &priors,
        legal: &legal,
        g: &g,
        m: m_cfg,
        n,
        c_visit: d.c_visit,
        c_scale: d.c_scale,
        v_root: V_ROOT,
        visits: vec![0; 9],
        values: vec![0.0; 9],
        h: &H,
    });
    m.run(&mut LadderEval::default()).unwrap();
    let got = observed(&mut m);
    assert_eq!(
        got, want,
        "alternate {alternate}, m {m_cfg}, n {n}, batch {batch}"
    );
    // A second call on the same root: halving counts only its own n,
    // completed Q and sigma see every visit.
    let want = predict(&m, 1, n, m_cfg, V_ROOT);
    m.run(&mut LadderEval::default()).unwrap();
    let got = observed(&mut m);
    assert_eq!(
        got, want,
        "second call: alternate {alternate}, m {m_cfg}, n {n}"
    );
}

/// Every `m'` in {1, 2, 3, 5, 7} and `n` in {1, 3, m', 17, 64}, budgets
/// ending inside a phase and `n < m'`, batches 1, 3 and 8, both turn
/// orders: the search equals the reference bit for bit.
#[test]
fn test_gumbel_root_equals_independent_reference_bit_for_bit() {
    for m_cfg in [1, 2, 3, 5, 7] {
        for n in [1, 3, m_cfg, 17, 64] {
            for batch in [1, 3, 8] {
                for alternate in [0, 1] {
                    check_against_reference(alternate, m_cfg, n, batch);
                }
            }
        }
    }
}

/// `q` is from the root mover's view when the child's mover differs:
/// every root edge holds `N * H[a]` exactly for these dyadic values.
#[test]
fn test_gumbel_q_sign_when_child_mover_differs_is_root_movers_view() {
    check_sign(1);
}

/// The same when the root's mover also moves at the child.
#[test]
fn test_gumbel_q_sign_when_same_player_moves_again_is_root_movers_view() {
    check_sign(0);
}

fn check_sign(alternate: u32) {
    let mut m = ladder(alternate, gumbel(64, 8, 16));
    m.run(&mut LadderEval::default()).unwrap();
    let (visits, values, ..) = root_edges(&m);
    assert!(visits.iter().filter(|&&n| n > 0).count() >= 2);
    for a in 0..9 {
        assert_eq!(values[a], f64::from(visits[a]) * H[a], "action {a}");
    }
}

// -------------------------------------------------------------- Connect4

fn record(moves: &str) -> Record<Connect4> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(21))
    };
    let mut p = EnvPool::<Connect4, Masked>::with_masks(Connect4Config {}, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    for ch in moves.chars() {
        let me = p.state(0).unwrap().to_move();
        let mut act = [0, 0];
        act[me] = i32::try_from(ch.to_digit(10).unwrap()).unwrap();
        p.step(ActionInput {
            discrete: &act,
            continuous: &[],
        })
        .unwrap();
    }
    p.snapshot(0).unwrap()
}

fn c4_with(moves: &str, capacity: u32, eviction: Eviction, cfg: PuctConfig, workers: usize) -> C4 {
    let mut o = SearchOptions::new(
        NonZeroU32::new(capacity).unwrap(),
        NonZeroU32::new(8).unwrap(),
    );
    o.eviction = eviction;
    let build = move |c, opts| {
        let p = EnvPool::<Connect4, Masked>::with_masks(c, opts)?;
        if workers == 0 {
            Ok(p)
        } else {
            p.with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
        }
    };
    let pool = SearchPool::new(Connect4Config {}, o, &record(moves), build).unwrap();
    Mcts::new(pool, cfg).unwrap()
}

fn c4(moves: &str, cfg: PuctConfig) -> C4 {
    c4_with(moves, 4_096, Eviction::Strict, cfg, 0)
}

/// Priors and value from the leaf's state hash (as in `tests/mcts.rs`),
/// counting rows and remembering every record hash.
#[derive(Default)]
struct Canned {
    calls: u32,
    rows: u32,
    seen: Vec<u64>,
}

impl Evaluator<Connect4> for Canned {
    fn evaluate(
        &mut self,
        leaves: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        self.calls += 1;
        for (i, r) in leaves.iter().enumerate() {
            self.rows += 1;
            let h = r.hash().get();
            self.seen.push(h);
            let weights: Vec<f64> = (0..7)
                .map(|a| f64::from(1 + ((h >> (8 * a)) & 0xff) as u32))
                .collect();
            let sum: f64 = weights.iter().sum();
            for (a, w) in weights.iter().enumerate() {
                priors[i * 7 + a] = w / sum;
            }
            values[i] = f64::from((h >> 48) as u32 & 0xffff) / 32_767.5 - 1.0;
        }
        Ok(())
    }
}

fn visits(m: &C4) -> [u32; 7] {
    let mut v = [0; 7];
    m.root_visits(&mut v);
    v
}

type Print = ([u32; 7], u64, u64, Vec<u64>, Vec<u32>, u32);

/// Everything a later call can observe: visits, tree, call counter,
/// `pi'` bits, `S_0` and the chosen action.
fn fingerprint(m: &mut C4) -> Print {
    (
        visits(m),
        m.pool().tree_hash(),
        m.calls(),
        m.improved_policy()
            .unwrap()
            .iter()
            .map(|p| p.to_bits())
            .collect(),
        m.considered().unwrap().to_vec(),
        m.select_action(0.0).unwrap(),
    )
}

/// `m = 1`, batch 1: every simulation goes to the single considered
/// action, and below it the search is PUCT from that child. The child's
/// edges equal those of a PUCT search rooted at the child's record with
/// one simulation fewer (the first one expanded the child).
#[test]
fn test_gumbel_m1_batch1_equals_forced_root_puct() {
    let n = 40;
    let mut g = c4("33", gumbel(n, 1, 1));
    g.run(&mut Canned::default()).unwrap();
    let a = g.considered().unwrap()[0];
    assert_eq!(g.select_action(0.0).unwrap(), a);
    let v = visits(&g);
    assert_eq!(v[a as usize], n);
    let root = g.pool().root();
    let child = g.pool().child(root, a).unwrap().unwrap();
    let mut puct = PuctConfig::default();
    puct.simulations = n - 1;
    puct.batch_size = 1;
    let mut p = c4_with("33", 4_096, Eviction::Strict, puct, 0);
    let rec = *g.pool().record(child).unwrap();
    p.pool_mut().set_root(&rec).unwrap();
    p.run(&mut Canned::default()).unwrap();
    let proot = p.pool().root();
    for b in 0..7 {
        let (x, y) = (
            g.pool().edge_stats(child, b).unwrap(),
            p.pool().edge_stats(proot, b).unwrap(),
        );
        assert_eq!(
            (x.visits, x.value.to_bits(), x.prior.to_bits()),
            (y.visits, y.value.to_bits(), y.prior.to_bits()),
            "edge {b}"
        );
    }
}

/// After a reroot the root value is unknown: the next Gumbel call
/// evaluates the root record once more (one extra row) and keeps its
/// priors; the call after that does not. PUCT never does.
#[test]
fn test_gumbel_reroot_reevaluates_root_value_once() {
    for gumbel_root in [true, false] {
        let cfg = if gumbel_root {
            gumbel(48, 8, 16)
        } else {
            let mut c = PuctConfig::default();
            c.simulations = 48;
            c
        };
        let mut m = c4("33", cfg);
        m.run(&mut Canned::default()).unwrap();
        let root = m.pool().root();
        let child = m.pool().child(root, 3).unwrap().unwrap();
        m.pool_mut().reroot(child).unwrap();
        let rec = m.pool().record(child).unwrap().hash().get();
        let priors: Vec<u64> = (0..7)
            .map(|a| m.pool().edge_stats(child, a).unwrap().prior.to_bits())
            .collect();
        let mut ev = Canned::default();
        m.run(&mut ev).unwrap();
        let root_rows = ev.seen.iter().filter(|&&h| h == rec).count();
        assert_eq!(root_rows, usize::from(gumbel_root), "gumbel {gumbel_root}");
        let after: Vec<u64> = (0..7)
            .map(|a| m.pool().edge_stats(child, a).unwrap().prior.to_bits())
            .collect();
        assert_eq!(after, priors, "the root's priors are kept");
        let mut ev = Canned::default();
        m.run(&mut ev).unwrap();
        assert_eq!(ev.seen.iter().filter(|&&h| h == rec).count(), 0);
    }
}

/// Fails at its `at`-th evaluator call, else answers like `Canned`.
struct FailAt {
    inner: Canned,
    at: u32,
}

impl Evaluator<Connect4> for FailAt {
    fn evaluate(
        &mut self,
        leaves: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        if self.inner.calls + 1 == self.at {
            self.inner.calls += 1;
            return Err(EvaluatorError::Failed {
                reason: "boom".to_owned(),
            });
        }
        self.inner.evaluate(leaves, priors, values)
    }
}

/// A failure at the root re-evaluation or mid-call restores the tree,
/// the counter, `v_root` and the published `pi'`; the retry equals a
/// run that never failed.
#[test]
fn test_gumbel_failed_call_rolls_back_then_retry_equals_fresh() {
    let start = || {
        let mut m = c4_with("", 64, Eviction::Oldest, gumbel(50, 8, 16), 0);
        m.run(&mut Canned::default()).unwrap();
        let root = m.pool().root();
        let a = m.select_action(0.0).unwrap();
        let child = m.pool().child(root, a).unwrap().unwrap();
        m.pool_mut().reroot(child).unwrap();
        m.run(&mut Canned::default()).unwrap();
        // Reroot again: the next call re-evaluates the root value.
        let root = m.pool().root();
        let a = m.select_action(0.0).unwrap();
        let child = m.pool().child(root, a).unwrap().unwrap();
        m.pool_mut().reroot(child).unwrap();
        m
    };
    let mut m = start();
    let tree = m.pool().tree_hash();
    let calls = m.calls();
    for at in [1, 2, 4] {
        let err = m
            .run(&mut FailAt {
                inner: Canned::default(),
                at,
            })
            .unwrap_err();
        assert!(matches!(err, SearchError::Evaluator(_)), "{err}");
        assert_eq!(m.pool().tree_hash(), tree, "fail at {at}");
        assert_eq!(m.calls(), calls);
        assert!(m.improved_policy().is_none(), "the root changed");
    }
    let mut ev = Canned::default();
    m.run(&mut ev).unwrap();
    let mut fresh = start();
    let mut fev = Canned::default();
    fresh.run(&mut fev).unwrap();
    assert_eq!(ev.seen, fev.seen, "the same evaluator rows, root included");
    assert_eq!(fingerprint(&mut m), fingerprint(&mut fresh));
}

fn group_specs() -> Vec<(&'static str, PuctConfig)> {
    vec![
        ("", gumbel(64, 8, 16)),
        ("33", gumbel(50, 8, 1)),
        ("3412", gumbel(37, 3, 4)),
        ("44", {
            let mut c = PuctConfig::default();
            c.simulations = 40;
            c.noise = Some(RootNoise {
                alpha: 0.3,
                epsilon: 0.25,
            });
            c
        }),
        ("2", gumbel(9, 1, 7)),
    ]
}

/// ADR-0045 section 4 under the Gumbel root: each slot equals its own
/// independent search over three calls with reuse, with phase ends in
/// different rounds (`m' = 1` next to `m' = 7`), a PUCT slot alongside,
/// and a slot rerooted between calls (its extra root row travels in the
/// shared root batch).
#[test]
fn test_group_gumbel_slots_equal_independent_searches_over_calls_with_reuse() {
    let specs = group_specs();
    let build = || -> Vec<C4> { specs.iter().map(|(m, c)| c4(m, *c)).collect() };
    let mut solo = build();
    let mut g = SearchGroup::new(&Connect4Config {}, build(), DEFAULT_GROUP_BUDGET).unwrap();
    for call in 0..3 {
        for m in &mut solo {
            m.run(&mut Canned::default()).unwrap();
        }
        g.run(&mut Canned::default()).unwrap();
        for (i, m) in solo.iter_mut().enumerate() {
            let gm = g.slot_mut(i).unwrap();
            if i == 3 {
                assert_eq!(
                    (visits(m), m.pool().tree_hash(), m.calls()),
                    (visits(gm), gm.pool().tree_hash(), gm.calls()),
                    "call {call}, slot {i}"
                );
            } else {
                assert_eq!(fingerprint(gm), fingerprint(m), "call {call}, slot {i}");
            }
        }
        // Reroot slot 0 on its chosen child, in both.
        let a = solo[0].select_action(0.0).unwrap();
        for m in [&mut solo[0], g.slot_mut(0).unwrap()] {
            let root = m.pool().root();
            let child = m.pool().child(root, a).unwrap().unwrap();
            m.pool_mut().reroot(child).unwrap();
        }
    }
}

/// A failure in a later round rolls every Gumbel slot back; the retry
/// equals a group that never failed.
#[test]
fn test_group_gumbel_failure_rolls_back_every_slot_then_retry_succeeds() {
    let specs = group_specs();
    let build = || {
        let slots = specs.iter().map(|(m, c)| c4(m, *c)).collect();
        SearchGroup::new(&Connect4Config {}, slots, DEFAULT_GROUP_BUDGET).unwrap()
    };
    let mut g = build();
    g.run(&mut Canned::default()).unwrap();
    let before: Vec<_> = (0..specs.len())
        .map(|i| {
            let m = g.slot(i).unwrap();
            (visits(m), m.pool().tree_hash(), m.calls())
        })
        .collect();
    for at in [1, 5] {
        assert!(
            g.run(&mut FailAt {
                inner: Canned::default(),
                at,
            })
            .is_err()
        );
        for (i, want) in before.iter().enumerate() {
            let m = g.slot(i).unwrap();
            assert_eq!(
                &(visits(m), m.pool().tree_hash(), m.calls()),
                want,
                "slot {i}"
            );
        }
    }
    g.run(&mut Canned::default()).unwrap();
    let mut fresh = build();
    fresh.run(&mut Canned::default()).unwrap();
    fresh.run(&mut Canned::default()).unwrap();
    for i in [0, 1, 2, 4] {
        assert_eq!(
            fingerprint(g.slot_mut(i).unwrap()),
            fingerprint(fresh.slot_mut(i).unwrap()),
            "slot {i}"
        );
    }
}

/// Goldens: visits, `pi'` bits, `S_0`, the winner and the tree on
/// Connect4 (pinned on macOS arm64), identical at 1, 2, 7 and 64
/// workers and without workers.
#[test]
fn test_gumbel_canned_golden_at_every_worker_count() {
    for workers in [0, 1, 2, 7, 64] {
        let mut m = c4_with("33", 4_096, Eviction::Strict, gumbel(64, 8, 16), workers);
        m.run(&mut Canned::default()).unwrap();
        assert_eq!(fingerprint(&mut m), golden(GOLDEN), "workers {workers}");
    }
}

/// The same under eviction with a small capacity (24 slots, 15 live at
/// the end: evicted subtrees are not refilled). The root visits follow
/// the schedule and equal the unevicted ones; the values below differ.
#[test]
fn test_gumbel_canned_golden_under_eviction() {
    for workers in [0, 7] {
        let mut m = c4_with("33", 24, Eviction::Oldest, gumbel(64, 8, 16), workers);
        m.run(&mut Canned::default()).unwrap();
        assert_eq!(m.pool().live_nodes(), GOLDEN_EVICTED_LIVE);
        assert_eq!(
            fingerprint(&mut m),
            golden(GOLDEN_EVICTED),
            "workers {workers}"
        );
    }
}

type Golden = ([u32; 7], u64, [u64; 7], &'static [u32], u32);

fn golden(g: Golden) -> Print {
    (g.0, g.1, 1, g.2.to_vec(), g.3.to_vec(), g.4)
}

const GOLDEN: Golden = (
    [3, 21, 18, 3, 8, 3, 8],
    15_193_610_498_785_244_960,
    [
        0x3c2f_5d1d_8f3e_4429,
        0x3fef_ffe3_ad25_34e0,
        0x3e70_2d3e_337d_bd34,
        0x3d40_c5d9_0277_fc57,
        0x3eea_7f16_90ea_1db1,
        0x3d8a_f435_0b1e_61fc,
        0x3eab_3694_dcbb_55ca,
    ],
    &[6, 4, 1, 3, 2, 0, 5],
    1,
);
const GOLDEN_EVICTED_LIVE: usize = 15;
const GOLDEN_EVICTED: Golden = (
    [3, 21, 18, 3, 8, 3, 8],
    3_143_531_525_314_830_660,
    [
        0x3c08_9d29_40aa_d266,
        0x3fef_fffa_32a8_1dad,
        0x3e81_45f8_7fd5_b0c1,
        0x3d1a_5387_4b80_3b2b,
        0x3ec4_cb4a_ec18_e6f9,
        0x3d65_2734_d6df_dde8,
        0x3e85_5b4b_d4c2_1b8f,
    ],
    &[6, 4, 1, 3, 2, 0, 5],
    1,
);

/// Gumbel with Dirichlet noise, `m = 0`, and a negative or non-finite
/// `c_visit` or `c_scale` are refused; so is `temperature > 0`, and an
/// action before any call. The noise refusal is why the two samplers
/// never share a call's `StreamId::SEARCH` draws.
#[test]
fn test_gumbel_rejects_noise_bad_options_and_temperature() {
    let pool = || {
        SearchPool::new(
            Connect4Config {},
            SearchOptions::new(NonZeroU32::new(64).unwrap(), NonZeroU32::new(8).unwrap()),
            &record(""),
            EnvPool::with_masks,
        )
        .unwrap()
    };
    let mut c = gumbel(16, 8, 4);
    c.noise = Some(RootNoise {
        alpha: 0.3,
        epsilon: 0.25,
    });
    assert!(matches!(
        Mcts::new(pool(), c),
        Err(SearchError::Options { option: "root", .. })
    ));
    for (m, cv, cs) in [
        (0, 50.0, 1.0),
        (4, f64::NAN, 1.0),
        (4, -1.0, 1.0),
        (4, 50.0, f64::INFINITY),
        (4, 50.0, -0.5),
    ] {
        let mut g = GumbelConfig::default();
        g.m = m;
        g.c_visit = cv;
        g.c_scale = cs;
        let mut c = gumbel(16, 8, 4);
        c.root = RootSelection::Gumbel(g);
        assert!(Mcts::new(pool(), c).is_err(), "{m} {cv} {cs}");
    }
    let mut m = Mcts::new(pool(), gumbel(16, 8, 4)).unwrap();
    assert!(m.select_action(0.0).is_err(), "no call yet");
    assert!(m.improved_policy().is_none());
    m.run(&mut Canned::default()).unwrap();
    assert!(m.select_action(1.0).is_err());
    assert!(m.select_action(0.0).is_ok());
    let p: f64 = m.improved_policy().unwrap().iter().sum();
    assert!((p - 1.0).abs() < 1e-12);
    // PUCT has no Gumbel results.
    let mut c = PuctConfig::default();
    c.simulations = 16;
    let mut puct = Mcts::new(pool(), c).unwrap();
    puct.run(&mut Canned::default()).unwrap();
    assert!(puct.improved_policy().is_none() && puct.considered().is_none());
}

/// The batch evaluator path: rows never cross a phase. With batch 8 and
/// `m' = 7`, `n = 64`, phase 0 has `7 * 3 = 21` visits, so no batch of the
/// call straddles visit 21.
#[derive(Default)]
struct Sizes {
    inner: Canned,
    sizes: Vec<usize>,
}

impl Evaluator<Connect4> for Sizes {
    fn evaluate(
        &mut self,
        leaves: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        self.inner.evaluate(leaves, priors, values)
    }

    fn evaluate_batch(
        &mut self,
        batch: &mut LeafBatch<'_, Connect4>,
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        self.sizes.push(batch.len());
        self.inner.evaluate(batch.records(), priors, values)
    }
}

/// The root row, then each batch's evaluated leaves.
const BATCH_SIZES: &[usize] = &[1, 7, 8, 6, 8, 8, 4, 8, 8, 4, 3];

#[test]
fn test_gumbel_batches_never_cross_a_phase() {
    let mut m = c4("", gumbel(64, 8, 16));
    let mut ev = Sizes::default();
    m.run(&mut ev).unwrap();
    // The root row, then batches; the running total of leaves passes
    // every phase boundary exactly (no terminal leaf this early).
    let mut total = 0;
    let mut seen = Vec::new();
    for &s in &ev.sizes[1..] {
        total += s;
        seen.push(total);
    }
    assert!(seen.contains(&21), "{:?}", ev.sizes);
    assert_eq!(*seen.last().unwrap(), 64);
    // Phases of 21, 20, 20 and 3 visits (m' = 7, n = 64), batch 8: a
    // batch stops early on a repeated unexpanded root edge, and never
    // passes a phase end.
    assert_eq!(ev.sizes, BATCH_SIZES);
}

/// Answers like `Canned`, except for the record `target` from its
/// second sighting on: other priors (reversed) and, when `value`, the
/// negated value.
struct Second {
    inner: Canned,
    target: u64,
    seen: u32,
    priors: bool,
    value: bool,
}

impl Evaluator<Connect4> for Second {
    fn evaluate(
        &mut self,
        leaves: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        self.inner.evaluate(leaves, priors, values)?;
        for (i, r) in leaves.iter().enumerate() {
            if r.hash().get() != self.target {
                continue;
            }
            self.seen += 1;
            if self.seen < 2 {
                continue;
            }
            if self.priors {
                priors[i * 7..i * 7 + 7].reverse();
            }
            if self.value {
                values[i] = -values[i];
            }
        }
        Ok(())
    }
}

/// What a re-evaluated root leaves: tree, root priors, `pi'`, `S_0`.
type Reeval = (u64, Vec<u64>, Vec<u64>, Vec<u32>);

fn reeval_print(m: &C4) -> Reeval {
    let root = m.pool().root();
    (
        m.pool().tree_hash(),
        (0..7)
            .map(|a| m.pool().edge_stats(root, a).unwrap().prior.to_bits())
            .collect(),
        m.improved_policy()
            .unwrap()
            .iter()
            .map(|p| p.to_bits())
            .collect(),
        m.considered().unwrap().to_vec(),
    )
}

/// Runs a Gumbel call, reroots on its chosen action, then runs again with an
/// evaluator whose second answer for the new root changes its priors
/// and/or its value; `group` drives both calls through a one-slot plus
/// one-other-slot group (the shared root batch).
fn reeval_case(group: bool, priors: bool, value: bool) -> Reeval {
    // 5 simulations over 7 actions: some root edges stay unvisited, so
    // v_root enters pi' through v_mix.
    let cfg = gumbel(5, 8, 16);
    let mut first = c4("33", cfg);
    first.run(&mut Canned::default()).unwrap();
    let a = first.select_action(0.0).unwrap();
    let child = first.pool().child(first.pool().root(), a).unwrap().unwrap();
    let target = first.pool().record(child).unwrap().hash().get();
    let mut ev = Second {
        inner: Canned::default(),
        target,
        seen: 0,
        priors,
        value,
    };
    if group {
        let mut g = SearchGroup::new(
            &Connect4Config {},
            vec![c4("", cfg), c4("33", cfg)],
            DEFAULT_GROUP_BUDGET,
        )
        .unwrap();
        g.run(&mut ev).unwrap();
        let m = g.slot_mut(1).unwrap();
        let child = m.pool().child(m.pool().root(), a).unwrap().unwrap();
        m.pool_mut().reroot(child).unwrap();
        g.run(&mut ev).unwrap();
        assert_eq!(ev.seen, 2, "seen as a leaf, then once as the root");
        reeval_print(g.slot(1).unwrap())
    } else {
        let mut m = c4("33", cfg);
        m.run(&mut ev).unwrap();
        let child = m.pool().child(m.pool().root(), a).unwrap().unwrap();
        m.pool_mut().reroot(child).unwrap();
        m.run(&mut ev).unwrap();
        assert_eq!(ev.seen, 2, "seen as a leaf, then once as the root");
        reeval_print(&m)
    }
}

/// The root re-evaluation after a reroot keeps the priors of the first
/// evaluation (other priors in the second answer change nothing) and
/// takes `v_root` from the second answer (another value changes `pi'`),
/// solo and in a group's shared root batch.
#[test]
fn test_gumbel_reroot_reevaluation_keeps_first_priors_and_takes_second_value() {
    for group in [false, true] {
        let same = reeval_case(group, false, false);
        assert_eq!(
            reeval_case(group, true, false),
            same,
            "group {group}: priors kept"
        );
        let value_only = reeval_case(group, false, true);
        assert_ne!(
            value_only.2, same.2,
            "group {group}: v_root is the second value"
        );
        assert_eq!(value_only.1, same.1);
        assert_eq!(
            reeval_case(group, true, true),
            value_only,
            "group {group}: only the value is taken"
        );
    }
}
