// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ValueMode::Returns` (ADR-0096) on a one-agent environment with dense
//! rewards: the closed-form discounted backup compared exactly, unbounded
//! terminal rewards, the configuration and one-mover refusals,
//! non-finite values refused with the tree unchanged, min-max scale
//! invariance under PUCT and the Gumbel root, and bit-identity across
//! worker counts and schedules. `Terminal` stays ADR-0043's bit for bit.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, NoAdaptive, ObsLeaf, Observation, RecordHeader, SpaceError,
    StepOut, Tier,
};
use tessera_rl_engine::{
    AutoReset, EngineError, EnvPool, Masked, NoMasks, PoolOptions, Record, Reseed, Schedule,
    SyncOptions,
};
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};
use tessera_rl_search::{
    Evaluator, EvaluatorError, Eviction, GumbelConfig, Mcts, NodeHandle, PuctConfig, ReturnSource,
    RootSelection, SearchError, SearchOptions, SearchPool, ValueMode,
};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Path {
    steps: u32,
    /// The actions so far, one base-4 digit each (distinct states).
    code: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct WalkConfig {
    #[config(default = 3)]
    actions: u32,
    #[config(default = 3)]
    length: u32,
    /// Every reward times `scale`.
    #[config(default = 1)]
    scale: u32,
}

#[derive(Observation)]
struct PathObs {
    #[space(box(low = 0, high = 1_000_000, shape = [2]))]
    path: ObsLeaf<u32>,
}

/// One agent, `actions` choices per step, `length` steps. Step `k`
/// (1-based) with action `a` rewards `(0.25 (a + 1) k - 0.5) * scale`,
/// and the last step adds `32 (a + 1) * scale`, outside `[-1, 1]`.
struct Walk;

fn reward(cfg: &WalkConfig, k: u32, a: u32) -> f32 {
    let small = |x: u32| f32::from(u16::try_from(x).unwrap());
    let (kf, a1, s) = (small(k), small(a + 1), small(cfg.scale));
    let base = 0.25 * a1 * kf - 0.5;
    let bonus = if k == cfg.length { 32.0 * a1 } else { 0.0 };
    (base + bonus) * s
}

impl Environment for Walk {
    type State = Path;
    type Config = WalkConfig;
    type Obs = PathObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &WalkConfig) -> Result<Space, SpaceError> {
        PathObs::space(cfg)
    }

    fn action_space(cfg: &WalkConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: cfg.actions })
    }

    fn reset(_: &WalkConfig, _: &mut StepRng, state: &mut Path) {
        *state = Path::default();
    }

    fn step(
        cfg: &WalkConfig,
        state: &mut Path,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let me = AgentSlot::FIRST;
        let a = u32::try_from(actions.discrete(me, 0).unwrap_or(0)).unwrap_or(0);
        state.steps += 1;
        state.code = state.code * 4 + a + 1;
        out.set_reward(me, reward(cfg, state.steps, a));
        if state.steps >= cfg.length {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &WalkConfig,
        state: &Path,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, PathObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(PathObs::PATH) {
            leaf.set(0, state.steps);
            leaf.set(1, state.code);
        }
    }
}

impl tessera_rl_core::Validate for Walk {
    fn validate(_: &WalkConfig, _: &Path) -> Result<(), tessera_rl_core::InvariantViolation> {
        Ok(())
    }
}

/// Priors and values from the state hash (dyadic, so a power-of-two
/// `scale` scales values exactly), or a fixed value.
struct Canned {
    actions: usize,
    scale: f64,
    fixed: Option<f64>,
}

impl Canned {
    fn new(cfg: &WalkConfig) -> Self {
        Self {
            actions: usize::try_from(cfg.actions).unwrap(),
            scale: f64::from(cfg.scale),
            fixed: None,
        }
    }
}

impl Evaluator<Walk> for Canned {
    fn evaluate(
        &mut self,
        leaves: &[Record<Walk>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        let n = self.actions;
        for (i, r) in leaves.iter().enumerate() {
            // From the state alone: the record header carries the episode
            // return, which `scale` changes.
            let s = r.state();
            let h = u64::from(s.code)
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(u64::from(s.steps))
                .rotate_left(17);
            let w: Vec<f64> = (0..n)
                .map(|a| f64::from(1 + u32::try_from((h >> (8 * a)) & 0xff).unwrap()))
                .collect();
            let sum: f64 = w.iter().sum();
            for (a, x) in w.iter().enumerate() {
                priors[i * n + a] = x / sum;
            }
            let v = f64::from(u32::try_from((h >> 40) & 0xff).unwrap()) / 128.0 - 1.0;
            values[i] = self.fixed.unwrap_or(v) * self.scale;
        }
        Ok(())
    }
}

fn root(cfg: &WalkConfig) -> Record<Walk> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(9))
    };
    let mut p = EnvPool::<Walk, NoMasks>::new(cfg.clone(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p.snapshot(0).unwrap()
}

fn search_with(
    cfg: &WalkConfig,
    puct: PuctConfig,
    workers: usize,
    schedule: Schedule,
) -> Result<Mcts<Walk, NoMasks>, SearchError> {
    let mut o = SearchOptions::new(NonZeroU32::new(4_096).unwrap(), NonZeroU32::new(8).unwrap());
    o.eviction = Eviction::Strict;
    let build = move |c, mut opts: PoolOptions| -> Result<EnvPool<Walk, NoMasks>, EngineError> {
        opts.schedule = schedule;
        let p = EnvPool::<Walk, NoMasks>::new(c, opts)?;
        if workers == 0 {
            Ok(p)
        } else {
            p.with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
        }
    };
    let pool = SearchPool::new(cfg.clone(), o, &root(cfg), build)?;
    Mcts::new(pool, puct)
}

fn search(cfg: &WalkConfig, puct: PuctConfig) -> Mcts<Walk, NoMasks> {
    search_with(cfg, puct, 0, Schedule::Static).unwrap()
}

fn returns(discount: f64, simulations: u32) -> PuctConfig {
    let mut c = PuctConfig::default();
    c.simulations = simulations;
    c.value = ValueMode::Returns { discount };
    c
}

/// Every edge's `(N, W bits)` in a depth-first walk, and the tree hash.
fn fingerprint(m: &Mcts<Walk, NoMasks>, actions: u32) -> (u64, Vec<(u32, u64)>) {
    fn walk(m: &Mcts<Walk, NoMasks>, h: NodeHandle, actions: u32, out: &mut Vec<(u32, u64)>) {
        for a in 0..actions {
            let e = m.pool().edge_stats(h, a).unwrap();
            out.push((e.visits, e.value.to_bits()));
            if let Some(c) = m.pool().child(h, a).unwrap() {
                walk(m, c, actions, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(m, m.pool().root(), actions, &mut out);
    (m.pool().tree_hash(), out)
}

#[test]
fn test_returns_backup_equals_the_closed_form_exactly() {
    let cfg = WalkConfig {
        actions: 1,
        ..WalkConfig::default()
    };
    let (d, v) = (0.5, 0.75);
    let mut puct = returns(d, 3);
    puct.batch_size = 1;
    let mut m = search(&cfg, puct);
    let mut eval = Canned::new(&cfg);
    eval.fixed = Some(v);
    m.run(&mut eval).unwrap();
    let r = |k| f64::from(reward(&cfg, k, 0));
    let (r1, r2, r3) = (r(1), r(2), r(3));
    assert!(r3 > 1.0, "the terminal reward is outside [-1, 1]");
    let p = m.pool();
    let c1 = p.child(p.root(), 0).unwrap().unwrap();
    let c2 = p.child(c1, 0).unwrap().unwrap();
    // Leaves at depth 1, 2 and 3 (terminal, V = 0), in that order.
    let e1 = p.edge_stats(p.root(), 0).unwrap();
    let e2 = p.edge_stats(c1, 0).unwrap();
    let e3 = p.edge_stats(c2, 0).unwrap();
    assert_eq!((e1.visits, e2.visits, e3.visits), (3, 2, 1));
    let w3 = 0.0 + r3;
    let w2 = (0.0 + (r2 + d * v)) + (r2 + d * r3);
    let w1 = ((0.0 + (r1 + d * v)) + (r1 + d * (r2 + d * v))) + (r1 + d * (r2 + d * r3));
    assert_eq!(e3.value.to_bits(), w3.to_bits());
    assert_eq!(e2.value.to_bits(), w2.to_bits());
    assert_eq!(e1.value.to_bits(), w1.to_bits());
}

#[test]
fn test_returns_accepts_unbounded_values_that_terminal_mode_refuses() {
    let cfg = WalkConfig {
        actions: 1,
        ..WalkConfig::default()
    };
    let mut eval = Canned::new(&cfg);
    eval.fixed = Some(0.75);
    let mut terminal = returns(0.5, 3);
    terminal.value = ValueMode::Terminal;
    terminal.batch_size = 1;
    let mut m = search(&cfg, terminal);
    assert_eq!(m.run(&mut eval), Err(SearchError::TerminalValue));
    // An evaluator value of 3: out of range for Terminal, finite for Returns.
    eval.fixed = Some(3.0);
    let mut m = search(&cfg, terminal);
    assert!(matches!(
        m.run(&mut eval),
        Err(SearchError::Evaluator(
            EvaluatorError::ValueOutOfRange { .. }
        ))
    ));
    let mut m = search(&cfg, returns(0.5, 3));
    m.run(&mut eval).unwrap();
}

#[test]
fn test_returns_refuses_bad_discounts_and_alternating_movers() {
    let cfg = WalkConfig::default();
    for bad in [-0.25, 1.5, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                search_with(&cfg, returns(bad, 4), 0, Schedule::Static),
                Err(SearchError::Options {
                    option: "value.discount",
                    ..
                })
            ),
            "discount {bad}"
        );
    }
    for good in [0.0, 1.0] {
        search(&cfg, returns(good, 4));
    }
    let o = SearchOptions::new(NonZeroU32::new(64).unwrap(), NonZeroU32::new(8).unwrap());
    let mut p = EnvPool::<Connect4, Masked>::with_masks(
        Connect4Config {},
        PoolOptions::new(NonZeroU32::MIN, Seed::new(1)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let c4 = SearchPool::new(
        Connect4Config {},
        o,
        &p.snapshot(0).unwrap(),
        EnvPool::with_masks,
    )
    .unwrap();
    assert!(matches!(
        Mcts::new(c4, returns(0.5, 4)),
        Err(SearchError::ReturnsNeedOneMover)
    ));
}

/// Decision 2a: finite evaluator values whose returns overflow are
/// refused before any edge of the path changes, and the failed call
/// leaves the tree, its statistics and the counter as before it.
#[test]
fn test_returns_overflow_is_refused_with_the_tree_unchanged() {
    let cfg = WalkConfig {
        actions: 1,
        ..WalkConfig::default()
    };
    let mut eval = Canned::new(&cfg);
    eval.fixed = Some(f64::MAX * 0.75);
    let mut puct = returns(1.0, 1);
    puct.batch_size = 1;
    let mut m = search(&cfg, puct);
    for _ in 0..8 {
        let (before, calls) = (fingerprint(&m, cfg.actions), m.calls());
        if let Err(err) = m.run(&mut eval) {
            assert!(
                matches!(
                    err,
                    SearchError::NonFiniteReturn {
                        origin: ReturnSource::Overflow,
                        ..
                    }
                ),
                "{err:?}"
            );
            assert_eq!(fingerprint(&m, cfg.actions), before);
            assert_eq!(m.calls(), calls);
            return;
        }
    }
    panic!("no call overflowed");
}

#[test]
fn test_returns_selection_is_scale_invariant_under_puct_and_gumbel() {
    let gumbel = {
        let mut g = GumbelConfig::default();
        g.m = 3;
        RootSelection::Gumbel(g)
    };
    for root_sel in [RootSelection::Puct, gumbel] {
        let run = |scale: u32| {
            let cfg = WalkConfig {
                length: 4,
                scale,
                ..WalkConfig::default()
            };
            let mut puct = returns(0.5, 48);
            puct.root = root_sel;
            let mut m = search(&cfg, puct);
            m.run(&mut Canned::new(&cfg)).unwrap();
            let mut visits = vec![0; 3];
            m.root_visits(&mut visits);
            let policy = m
                .improved_policy()
                .map(|p| p.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            let means: Vec<f64> = (0..3)
                .map(|a| {
                    let e = m.pool().edge_stats(m.pool().root(), a).unwrap();
                    e.value / f64::from(e.visits.max(1))
                })
                .collect();
            // Not the tree hash: the records carry the episode return.
            (visits, policy, means)
        };
        let (one, k) = (run(1), run(1024));
        assert_eq!(one.0, k.0, "{root_sel:?}");
        assert_eq!(one.1, k.1, "{root_sel:?}");
        for (a, b) in one.2.iter().zip(&k.2) {
            assert_eq!((a * 1024.0).to_bits(), b.to_bits());
        }
    }
}

#[test]
fn test_returns_is_bit_identical_across_workers_and_schedules() {
    let cfg = WalkConfig {
        length: 4,
        ..WalkConfig::default()
    };
    let golden = {
        let mut m = search(&cfg, returns(0.5, 64));
        m.run(&mut Canned::new(&cfg)).unwrap();
        fingerprint(&m, cfg.actions)
    };
    for workers in [1, 2, 7] {
        for schedule in [Schedule::Static, Schedule::Dynamic] {
            let mut m = search_with(&cfg, returns(0.5, 64), workers, schedule).unwrap();
            m.run(&mut Canned::new(&cfg)).unwrap();
            assert_eq!(
                fingerprint(&m, cfg.actions),
                golden,
                "workers {workers}, {schedule:?}"
            );
        }
    }
}

/// Uniform priors and a value in `[-1, 1)` from the state hash.
struct C4Canned;

impl Evaluator<Connect4> for C4Canned {
    fn evaluate(
        &mut self,
        leaves: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        priors.fill(1.0 / 7.0);
        for (i, r) in leaves.iter().enumerate() {
            let h = r.hash().get();
            values[i] = f64::from(u32::try_from((h >> 40) & 0xff).unwrap()) / 128.0 - 1.0;
        }
        Ok(())
    }
}

/// `ValueMode::Terminal` written explicitly is the default: the same
/// visits, edge values and tree on Connect4 (ADR-0096 decision 1).
#[test]
fn test_explicit_terminal_mode_equals_the_default_on_connect4() {
    let run = |puct: PuctConfig| {
        let o = SearchOptions::new(NonZeroU32::new(4_096).unwrap(), NonZeroU32::new(8).unwrap());
        let mut p = EnvPool::<Connect4, Masked>::with_masks(
            Connect4Config {},
            PoolOptions::new(NonZeroU32::MIN, Seed::new(21)),
        )
        .unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        let pool = SearchPool::new(
            Connect4Config {},
            o,
            &p.snapshot(0).unwrap(),
            EnvPool::with_masks,
        )
        .unwrap();
        let mut m = Mcts::new(pool, puct).unwrap();
        m.run(&mut C4Canned).unwrap();
        let r = m.pool().root();
        let edges: Vec<(u32, u64)> = (0..7)
            .map(|a| {
                let e = m.pool().edge_stats(r, a).unwrap();
                (e.visits, e.value.to_bits())
            })
            .collect();
        (m.pool().tree_hash(), edges)
    };
    let mut default = PuctConfig::default();
    default.simulations = 200;
    let mut explicit = default;
    explicit.value = ValueMode::Terminal;
    assert_eq!(run(default), run(explicit));
}

/// FNV-1a over a fingerprint's `(N, W bits)` pairs.
fn digest(edges: &[(u32, u64)]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for &(n, w) in edges {
        for b in u64::from(n)
            .to_le_bytes()
            .into_iter()
            .chain(w.to_le_bytes())
        {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// Pinned `Returns` goldens, pinned on Linux x86-64 (ADR-0043 section 4
/// makes them identical on every platform): PUCT with batches of 8
/// (virtual loss in play) over two calls, the second starting its own
/// min-max, and the Gumbel root's visits and `pi'`.
#[test]
fn test_returns_puct_and_gumbel_goldens_are_pinned() {
    let cfg = WalkConfig {
        length: 4,
        ..WalkConfig::default()
    };
    let mut m = search(&cfg, returns(0.5, 48));
    let mut eval = Canned::new(&cfg);
    m.run(&mut eval).unwrap();
    let first = digest(&fingerprint(&m, cfg.actions).1);
    m.run(&mut eval).unwrap();
    let second = digest(&fingerprint(&m, cfg.actions).1);
    let mut g = GumbelConfig::default();
    // Two of three root actions: the third is unvisited, so `pi'` uses
    // `v_mix` in normalized units.
    g.m = 2;
    let mut puct = returns(0.5, 48);
    puct.root = RootSelection::Gumbel(g);
    let mut gm = search(&cfg, puct);
    gm.run(&mut eval).unwrap();
    let mut visits = vec![0; 3];
    gm.root_visits(&mut visits);
    let policy: Vec<u64> = gm
        .improved_policy()
        .unwrap()
        .iter()
        .map(|p| p.to_bits())
        .collect();
    let got = (first, second, visits, policy);
    let pinned = (
        9_811_372_669_519_447_865,
        14_723_742_536_957_817_435,
        vec![0, 24, 24],
        vec![
            4_599_162_800_145_273_833,
            4_598_890_451_588_625_239,
            4_600_976_006_529_300_673,
        ],
    );
    assert_eq!(got, pinned);
}
