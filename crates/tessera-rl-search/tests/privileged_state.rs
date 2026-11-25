// SPDX-License-Identifier: MIT OR Apache-2.0

//! Search never builds or reads the privileged state buffer (ADR-0079 item
//! 7.3): on Leduc hold'em, a root taken from a live pool with the buffer is
//! byte-identical to one taken without it, and the search over it (its
//! tree, its root visits and every observation and record handed to the
//! evaluator) is identical too; a scratch pool with the buffer is refused.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::AgentSlot;
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, EnvPool, Masked, PoolOptions, Record, Reseed,
};
use tessera_rl_games::leduc::CALL;
use tessera_rl_games::{Leduc, LeducConfig};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{
    Determinization, Evaluator, EvaluatorError, LeafBatch, Mcts, PuctConfig, SearchError,
    SearchOptions, SearchPool,
};

const CFG: LeducConfig = LeducConfig {};

fn options() -> PoolOptions {
    PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(41))
    }
}

/// Instance 0's record after one check, from a live pool with or without
/// the privileged state buffer.
fn root(privileged: bool) -> Record<Leduc> {
    let mut p = EnvPool::<Leduc, Masked>::with_masks(CFG, options()).unwrap();
    if privileged {
        p = p.with_privileged_state().unwrap();
    }
    p.reset_all(Reseed::Keep).unwrap();
    let act = [i32::try_from(CALL).unwrap(); 4];
    p.step(ActionInput {
        discrete: &act,
        continuous: &[],
    })
    .unwrap();
    assert_eq!(p.has_privileged_state(), privileged);
    p.snapshot(0).unwrap()
}

fn how(index: u64) -> Determinization {
    Determinization {
        agent: AgentSlot::new(1, 2).unwrap(),
        sampler_seed: Seed::new(5),
        sample_index: index,
    }
}

fn search_options() -> SearchOptions {
    SearchOptions::new(NonZeroU32::new(256).unwrap(), NonZeroU32::new(8).unwrap())
}

/// Every evaluator input: the leaf records and observation rows.
#[derive(Default)]
struct Recorder {
    records: Vec<Vec<u8>>,
    observations: Vec<u8>,
}

impl Evaluator<Leduc> for Recorder {
    fn evaluate(
        &mut self,
        leaves: &[Record<Leduc>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        priors.fill(1.0 / 3.0);
        for (v, r) in values.iter_mut().zip(leaves) {
            *v = f64::from(u32::try_from(r.hash().get() >> 48).unwrap_or(0)) / 65_535.0 - 0.5;
        }
        Ok(())
    }

    fn evaluate_batch(
        &mut self,
        batch: &mut LeafBatch<'_, Leduc>,
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        self.observations.extend_from_slice(batch.observations());
        self.records
            .extend(batch.records().iter().map(Record::to_bytes));
        self.evaluate(batch.records(), priors, values)
    }
}

/// The search of `root`: its tree hash, root visits and evaluator inputs.
fn searched(root: &Record<Leduc>) -> (u64, Vec<u32>, Recorder) {
    let mut c = PuctConfig::default();
    c.simulations = 64;
    c.batch_size = 4;
    let mut pool =
        SearchPool::new_determinized(CFG, search_options(), root, how(1), EnvPool::with_masks)
            .unwrap();
    pool.set_root_determinized(root, how(2)).unwrap();
    let mut m = Mcts::new(pool, c).unwrap();
    let mut recorder = Recorder::default();
    m.run(&mut recorder).unwrap();
    let mut visits = vec![0; 3];
    m.root_visits(&mut visits);
    (m.pool().tree_hash(), visits, recorder)
}

#[test]
fn test_search_privileged_state_on_or_off_gives_identical_search_and_evaluator_inputs() {
    let (off, on) = (root(false), root(true));
    assert_eq!(
        off.to_bytes(),
        on.to_bytes(),
        "the record never holds the buffer"
    );
    let (hash_off, visits_off, inputs_off) = searched(&off);
    let (hash_on, visits_on, inputs_on) = searched(&on);
    assert!(visits_off.iter().sum::<u32>() > 0);
    assert_ne!(inputs_off.observations.len(), 0);
    assert_eq!(hash_off, hash_on);
    assert_eq!(visits_off, visits_on);
    assert_eq!(inputs_off.records, inputs_on.records);
    assert_eq!(inputs_off.observations, inputs_on.observations);
}

#[test]
fn test_search_scratch_pool_with_privileged_state_is_refused() {
    let root = root(false);
    let built = SearchPool::new_determinized(CFG, search_options(), &root, how(1), |c, o| {
        EnvPool::with_masks(c, o)?.with_privileged_state()
    });
    assert!(matches!(
        built.err(),
        Some(SearchError::Engine(EngineError::PrivilegedState { .. }))
    ));
}
