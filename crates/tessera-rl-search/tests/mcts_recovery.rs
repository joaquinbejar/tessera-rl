// SPDX-License-Identifier: MIT OR Apache-2.0

//! A contained kernel fault on the first search call rolls the tree back,
//! root priors included, and leaves the scratch pool poisoned; the next
//! call must recover it and succeed (C3, Codex R39). An interruption
//! reported by [`Evaluator::check`] rolls the call back too (R46-01).

#![expect(
    clippy::expect_used,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, Ordering};

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, NoAdaptive, ObsLeaf, Observation, RecordHeader, SpaceError, StepOut,
    Tier, TurnStructure,
};
use tessera_rl_engine::{EngineError, EnvPool, NoMasks, PoolOptions, Record, Reseed};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};
use tessera_rl_search::{
    Evaluator, EvaluatorError, Mcts, PuctConfig, SearchError, SearchOptions, SearchPool,
};

/// Set to make the next step kernel panic once.
static FAULT: AtomicBool = AtomicBool::new(false);

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Count {
    total: u32,
    to_move: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct CountConfig {}

#[derive(Observation)]
struct CountObs {
    #[space(box(low = 0, high = 1000, shape = [1]))]
    total: ObsLeaf<i32>,
}

/// Two players add 0, 1 or 2 in turns; the game never ends. A step
/// panics once after `FAULT` is set.
struct OneShot;

impl Environment for OneShot {
    type State = Count;
    type Config = CountConfig;
    type Obs = CountObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = 2;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        CountObs::space(cfg)
    }

    fn action_space(_: &CountConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 3 })
    }

    fn acting(_: &CountConfig, state: &Count, agent: AgentSlot) -> bool {
        agent.index() == state.to_move as usize
    }

    fn reset(_: &CountConfig, _: &mut StepRng, state: &mut Count) {
        *state = Count::default();
    }

    fn step(
        _: &CountConfig,
        state: &mut Count,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
        assert!(
            !FAULT.swap(false, Ordering::Relaxed),
            "fixture: one-shot fault"
        );
        let me = AgentSlot::new(u16::try_from(state.to_move).unwrap_or(0), 2)
            .unwrap_or(AgentSlot::FIRST);
        let a = actions.discrete(me, 0).unwrap_or(0);
        state.total = state.total.saturating_add(u32::try_from(a).unwrap_or(0));
        state.to_move ^= 1;
    }

    fn observe(
        _: &CountConfig,
        state: &Count,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, CountObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(CountObs::TOTAL) {
            leaf.set(0, i32::try_from(state.total % 1000).unwrap_or(0));
        }
    }
}

impl tessera_rl_core::Validate for OneShot {
    fn validate(_: &CountConfig, _: &Count) -> Result<(), tessera_rl_core::InvariantViolation> {
        Ok(())
    }
}

struct Uniform;

impl Evaluator<OneShot> for Uniform {
    fn evaluate(
        &mut self,
        _: &[Record<OneShot>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        priors.fill(1.0 / 3.0);
        values.fill(0.0);
        Ok(())
    }
}

fn mcts() -> Mcts<OneShot, NoMasks> {
    let mut p = EnvPool::<OneShot>::new(
        CountConfig {},
        PoolOptions::new(NonZeroU32::MIN, Seed::new(1)),
    )
    .expect("pool");
    p.reset_all(Reseed::Keep).expect("reset");
    let pool = SearchPool::new(
        CountConfig {},
        SearchOptions::new(
            NonZeroU32::new(256).expect("capacity"),
            NonZeroU32::new(4).expect("batch"),
        ),
        &p.snapshot(0).expect("snapshot"),
        EnvPool::new,
    )
    .expect("search pool");
    let mut cfg = PuctConfig::default();
    cfg.batch_size = 4;
    cfg.simulations = 32;
    Mcts::new(pool, cfg).expect("mcts")
}

#[test]
fn test_mcts_recovers_the_scratch_pool_after_a_fault_on_the_first_call() {
    let mut m = mcts();
    let before = m.pool().tree_hash();
    FAULT.store(true, Ordering::Relaxed);
    let err = m.run(&mut Uniform).expect_err("the fault fails the call");
    assert_eq!(err, SearchError::Engine(EngineError::Panicked));
    assert_eq!(
        m.pool().tree_hash(),
        before,
        "rolled back, root priors included"
    );
    assert_eq!(m.calls(), 0);
    // The retry evaluates the root again on the poisoned scratch pool.
    m.run(&mut Uniform)
        .expect("the retry recovers the scratch pool");
    assert_eq!(m.calls(), 1);
    let mut v = [0; 3];
    m.root_visits(&mut v);
    assert_eq!(v.iter().sum::<u32>(), 32);
    // The retry is exactly a fresh first call: same tree, same stream.
    let mut fresh = mcts();
    fresh.run(&mut Uniform).expect("fresh");
    assert_eq!(m.pool().tree_hash(), fresh.pool().tree_hash());
    let draws = |x: &mut Mcts<OneShot, NoMasks>| {
        (0..16)
            .map(|_| x.select_action(1.0).expect("draw"))
            .collect::<Vec<_>>()
    };
    assert_eq!(draws(&mut m), draws(&mut fresh));
}

/// Uniform, and interrupted at the `at`-th check.
struct Interrupted {
    checks: u32,
    at: u32,
}

impl Evaluator<OneShot> for Interrupted {
    fn evaluate(
        &mut self,
        leaves: &[Record<OneShot>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        Uniform.evaluate(leaves, priors, values)
    }

    fn check(&mut self) -> Result<(), EvaluatorError> {
        self.checks += 1;
        if self.checks == self.at {
            return Err(EvaluatorError::Failed {
                reason: "interrupted".to_owned(),
            });
        }
        Ok(())
    }
}

#[test]
fn test_mcts_check_runs_before_every_batch_and_an_interruption_rolls_back() {
    let mut whole = Interrupted {
        checks: 0,
        at: u32::MAX,
    };
    let mut m = mcts();
    m.run(&mut whole).expect("never interrupted");
    // Once before the root, then once per batch: 32 simulations in
    // batches of at most 4.
    assert!(whole.checks > 8, "{} checks", whole.checks);
    let before = m.pool().tree_hash();
    let mut late = Interrupted { checks: 0, at: 6 };
    let err = m.run(&mut late).expect_err("interrupted");
    assert!(matches!(
        err,
        SearchError::Evaluator(EvaluatorError::Failed { .. })
    ));
    assert_eq!((m.pool().tree_hash(), m.calls()), (before, 1));
}
