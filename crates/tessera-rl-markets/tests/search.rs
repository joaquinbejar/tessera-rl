// SPDX-License-Identifier: MIT OR Apache-2.0

//! Search smoke test (design section 9, row 3). Search takes this
//! environment's `MultiDiscrete([6, 9, 4, 8])` action as a canonical joint
//! head (ADR-0086, #454), and its `Set` observation leaves are ordinary
//! evaluator leaves (#490), so a determinized search pool builds. With the
//! default policy map its first expansion is refused for simultaneous
//! actors (`NotSingleActor`, ADR-0043 unchanged); with every other slot an
//! `InfoHeuristic` slot of the scratch pool, slot 1 searches against them
//! (ADR-0093, #491). Under the default `ValueMode::Terminal` search values
//! are terminal-only (ADR-0093 decision 6): the per-step profit and loss
//! does not enter the tree; under `ValueMode::Returns` (ADR-0096, #524)
//! it does.
//! Plain search is refused for imperfect information first. The
//! determinized root the search would build is checked by hand here, the
//! way its sampler does it: a sample from the sampler's identity passes
//! `Validate` and keeps the searching agent's information state.

#![expect(
    clippy::unwrap_used,
    reason = "test fixtures: a failure is the assertion"
)]

mod support;

use core::num::NonZeroU32;

use support::{Rng, config, input, legal_actions, pool};
use tessera_rl_core::PolicyGroup;
use tessera_rl_core::encoding::encode_to_vec;
use tessera_rl_core::{AgentSlot, Determinize, ImperfectInformation, Validate};
use tessera_rl_engine::{ActionInput, EngineError, EnvPool, Masked, PoolOptions, Record, Reseed};
use tessera_rl_markets::limits::MAX_AGENTS;
use tessera_rl_markets::{LimitOrderBook, LobState};
use tessera_rl_num::identity::{Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_search::{
    Determinization, Evaluator, EvaluatorError, Mcts, PuctConfig, SearchError, SearchOptions,
    SearchPool, ValueMode, sampler_identity,
};

type Search = SearchPool<LimitOrderBook, Masked>;

fn options() -> SearchOptions {
    SearchOptions::new(NonZeroU32::new(64).unwrap(), NonZeroU32::new(4).unwrap())
}

#[test]
fn test_search_smoke_builds_and_refuses_simultaneous_actors_at_expansion() {
    let cfg = config(4, 2);
    let mut p = pool(cfg.clone(), 1, 11, None);
    let mut masks = p.reset_all(Reseed::Keep).unwrap().masks().to_vec();
    let mut rng = Rng(0x5ea1);
    for _ in 0..12 {
        let a = legal_actions(&mut rng, &masks, 1);
        masks = p.step(input(&a)).unwrap().masks().to_vec();
    }
    let root = p.snapshot(0).unwrap();
    let agent = AgentSlot::new(1, MAX_AGENTS).unwrap();
    let how = Determinization {
        agent,
        sampler_seed: Seed::new(3),
        sample_index: 0,
    };

    let plain = Search::new(cfg.clone(), options(), &root, EnvPool::with_masks);
    assert!(matches!(plain, Err(SearchError::ImperfectInformation)));
    let det = Search::new_determinized(cfg.clone(), options(), &root, how, EnvPool::with_masks);
    let mut det = det.unwrap();
    assert_eq!(det.observation_sets().len(), 2);
    let r = det.root();
    let mut out = [r];
    assert!(matches!(
        det.expand(&[(r, 0)], &mut out),
        Err(SearchError::NotSingleActor { .. })
    ));

    // The sample the determinized root would take.
    let info = LimitOrderBook::info_state(&cfg, root.state(), agent);
    let id = sampler_identity(how.sampler_seed, how.sample_index);
    let mut r = StepRng::new(&RngAuthority::engine_only(), &id, StreamId::DYNAMICS);
    let mut sample = LobState::default();
    LimitOrderBook::sample(&cfg, &info, &mut r, &mut sample);
    assert_eq!(LimitOrderBook::validate(&cfg, &sample), Ok(()));
    assert_eq!(
        encode_to_vec(&LimitOrderBook::info_state(&cfg, &sample, agent)),
        encode_to_vec(&info)
    );
}

/// The scratch pool of a search by slot 1: the other alive slots (0, 2, 3
/// of a 4-agent config) play the default baseline mix from their
/// information states (ADR-0079 item 4.2); the never-alive seats 4 to 15
/// keep the default group.
fn opponents(
    cfg: tessera_rl_markets::LobConfig,
    o: PoolOptions,
) -> Result<EnvPool<LimitOrderBook, Masked>, EngineError> {
    EnvPool::with_masks(cfg, o)?.with_info_heuristic_slots(&[0, 2, 3])
}

/// A recorded root after 12 seeded legal steps, and the determinization
/// of slot 1.
fn recorded() -> (
    tessera_rl_markets::LobConfig,
    Record<LimitOrderBook>,
    Determinization,
) {
    let cfg = config(4, 2);
    let mut p = pool(cfg.clone(), 1, 11, None);
    let mut masks = p.reset_all(Reseed::Keep).unwrap().masks().to_vec();
    let mut rng = Rng(0x5ea1);
    for _ in 0..12 {
        let a = legal_actions(&mut rng, &masks, 1);
        masks = p.step(input(&a)).unwrap().masks().to_vec();
    }
    let how = Determinization {
        agent: AgentSlot::new(1, MAX_AGENTS).unwrap(),
        sampler_seed: Seed::new(3),
        sample_index: 0,
    };
    (cfg, p.snapshot(0).unwrap(), how)
}

struct Uniform;

impl Evaluator<LimitOrderBook> for Uniform {
    fn evaluate(
        &mut self,
        _: &[Record<LimitOrderBook>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        priors.fill(1.0 / 1728.0);
        values.fill(0.0);
        Ok(())
    }
}

#[test]
fn test_search_order_book_against_heuristic_opponents_is_reproducible() {
    let (cfg, root, how) = recorded();
    let run = || {
        let s = Search::new_determinized(cfg.clone(), options(), &root, how, opponents).unwrap();
        assert_eq!(s.searcher(), Some(1));
        let mut c = PuctConfig::default();
        c.simulations = 32;
        c.batch_size = 4;
        let mut m = Mcts::new(s, c).unwrap();
        m.run(&mut Uniform).unwrap();
        let mut visits = vec![0u32; 1728];
        m.root_visits(&mut visits);
        (m.pool().tree_hash(), visits)
    };
    let (a, visits) = run();
    assert_eq!(visits.iter().sum::<u32>(), 32);
    assert_eq!(run().0, a);
}

#[test]
fn test_search_order_book_child_equals_the_live_step_with_heuristic_slots() {
    let (cfg, root, how) = recorded();
    let mut s = Search::new_determinized(cfg.clone(), options(), &root, how, opponents).unwrap();
    let r = s.root();
    let sampled = *s.record(r).unwrap();
    let mut out = [r];
    // Joint 0: the no-op, legal and canonical in every state.
    s.expand(&[(r, 0)], &mut out).unwrap();
    let mut live = opponents(cfg, PoolOptions::new(NonZeroU32::MIN, Seed::new(1))).unwrap();
    live.reset_all(Reseed::Keep).unwrap();
    live.restore(0, &sampled).unwrap();
    let actions = vec![0i32; MAX_AGENTS * 4];
    live.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap();
    assert_eq!(
        s.record(out[0]).unwrap().hash(),
        live.snapshot(0).unwrap().hash()
    );
}

#[test]
fn test_search_order_book_opponent_checks_are_typed() {
    let (cfg, root, how) = recorded();
    // Alive slot 3 left a Learner.
    let partial = |c, o| -> Result<EnvPool<LimitOrderBook, Masked>, EngineError> {
        EnvPool::with_masks(c, o)?.with_info_heuristic_slots(&[0, 2])
    };
    assert_eq!(
        Search::new_determinized(cfg.clone(), options(), &root, how, partial).err(),
        Some(SearchError::NotSearchable {
            slot: 3,
            group: PolicyGroup::Learner
        })
    );
    // An explicit searcher other than the determinization agent.
    let mut o = options();
    o.searcher = Some(2);
    assert_eq!(
        Search::new_determinized(cfg, o, &root, how, opponents).err(),
        Some(SearchError::SearcherMismatch {
            searcher: 2,
            determinized: 1
        })
    );
}

/// ADR-0096 (#524): the same search in `Returns` mode backs up the
/// searcher's per-step profit and loss, so its root edge values differ from
/// `Terminal` mode's (where a zero evaluator and no terminal leaf leave
/// every value 0), and it is reproducible.
#[test]
fn test_search_order_book_returns_mode_backs_up_the_step_rewards() {
    let (cfg, root, how) = recorded();
    let run = |value: ValueMode| {
        let s = Search::new_determinized(cfg.clone(), options(), &root, how, opponents).unwrap();
        let mut c = PuctConfig::default();
        c.simulations = 32;
        c.batch_size = 4;
        c.value = value;
        let mut m = Mcts::new(s, c).unwrap();
        m.run(&mut Uniform).unwrap();
        let r = m.pool().root();
        let values: Vec<u64> = (0..1728)
            .map(|a| m.pool().edge_stats(r, a).unwrap().value.to_bits())
            .collect();
        (m.pool().tree_hash(), values)
    };
    let returns = ValueMode::Returns { discount: 0.5 };
    let (hash, values) = run(returns);
    let (_, terminal) = run(ValueMode::Terminal);
    assert!(terminal.iter().all(|&v| v == 0.0_f64.to_bits()));
    assert!(values.iter().any(|&v| v != 0.0_f64.to_bits()));
    assert_eq!(run(returns), (hash, values));
}
