// SPDX-License-Identifier: MIT OR Apache-2.0

//! PUCT search objects for Python (ADR-0043): a type-erased [`Mcts`] over
//! one environment, and the bridge that lets a Python callable evaluate
//! its batches.
//!
//! The search runs attached to the interpreter: every batch calls the
//! Python evaluator, through a callout (ADR-0041), so the callable runs
//! under the caller's floating-point controls and the search arithmetic
//! under the canonical ones. Zero-allocation guarantees do not extend
//! across this boundary: building the arguments and reading the result
//! allocate.

use core::num::NonZeroU32;

use tessera_rl_core::encoding::decode_from_slice;
use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout, Validate};
use tessera_rl_engine::{
    AutoReset, DecodedRecord, EnvPool, MaskProvider, Masked, PoolOptions, Record, Reseed,
};
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{
    Evaluator, EvaluatorError, Eviction, LeafBatch, Mcts, PuctConfig, RootNoise, SearchError,
    SearchOptions, SearchPool,
};

use crate::env::BindError;

/// One evaluator batch, type-erased: what the Python callable receives.
pub(crate) struct RawBatch<'a> {
    /// Leaves.
    pub(crate) n: usize,
    /// `n * obs_bytes` observation bytes of each leaf's mover.
    pub(crate) observations: &'a [u8],
    /// `n * actions` legal flags.
    pub(crate) legal: &'a [bool],
    /// Each leaf's record hash (`StateHash`), for canned evaluators.
    pub(crate) hashes: Vec<u64>,
}

/// An evaluator over type-erased batches.
pub(crate) trait BatchEval {
    /// Writes `n * actions` priors and `n` values.
    fn eval(
        &mut self,
        batch: &RawBatch<'_>,
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError>;

    /// Polls for an interruption before every batch.
    fn check(&mut self) -> Result<(), EvaluatorError>;
}

/// Adapts a [`BatchEval`] to the search's evaluator trait.
struct Adapter<'a>(&'a mut dyn BatchEval);

impl<E: Environment> Evaluator<E> for Adapter<'_> {
    fn evaluate(
        &mut self,
        _: &[Record<E>],
        _: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        Err(EvaluatorError::Failed {
            reason: "the Python evaluator needs observations (evaluate_batch)".to_owned(),
        })
    }

    fn evaluate_batch(
        &mut self,
        batch: &mut LeafBatch<'_, E>,
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        let (n, legal) = (batch.len(), batch.legal());
        let hashes = batch.records().iter().map(|r| r.hash().get()).collect();
        let raw = RawBatch {
            n,
            observations: batch.observations(),
            legal,
            hashes,
        };
        self.0.eval(&raw, priors, values)
    }

    fn check(&mut self) -> Result<(), EvaluatorError> {
        self.0.check()
    }
}

/// What a search object can do, whatever the environment.
pub(crate) trait SearchOps: Send {
    /// Actions per node.
    fn actions(&self) -> u32;
    /// Bytes of one leaf's observation.
    fn obs_bytes(&self) -> usize;
    /// Replaces the tree by the root `record` (canonical record bytes).
    fn set_root(&mut self, record: &[u8]) -> Result<(), BindError>;
    /// One search call.
    fn run(&mut self, eval: &mut dyn BatchEval) -> Result<(), BindError>;
    /// The root's visit counts.
    fn root_visits(&self) -> Vec<u32>;
    /// The action to play (greedy at 0).
    fn select_action(&mut self, temperature: f64) -> Result<u32, BindError>;
    /// Completed calls.
    fn calls(&self) -> u64;
    /// The tree digest.
    fn tree_hash(&self) -> u64;
}

struct Searcher<E: Environment, M: MaskProvider<E>> {
    mcts: Mcts<E, M>,
    /// A one-instance pool that validates imported roots.
    importer: EnvPool<E, M>,
}

impl<E, M> SearchOps for Searcher<E, M>
where
    E: Environment<
            Config: Sync + Clone,
            State: Send,
            Header: Send + RecordLayout<Adaptive: AdaptiveValues<E::Config>>,
        > + Validate,
    M: MaskProvider<E> + Send,
    E::Config: Send,
{
    fn actions(&self) -> u32 {
        self.mcts.pool().actions()
    }

    fn obs_bytes(&self) -> usize {
        self.mcts.obs_bytes()
    }

    fn set_root(&mut self, record: &[u8]) -> Result<(), BindError> {
        let decoded = decode_from_slice::<DecodedRecord<E>>(record).map_err(BindError::Decode)?;
        self.importer.import(&[0], &[decoded])?;
        let root = self.importer.snapshot(0)?;
        self.mcts.pool_mut().set_root(&root)?;
        Ok(())
    }

    fn run(&mut self, eval: &mut dyn BatchEval) -> Result<(), BindError> {
        self.mcts.run(&mut Adapter(eval))?;
        Ok(())
    }

    fn root_visits(&self) -> Vec<u32> {
        let mut v = vec![0; usize::try_from(self.actions()).unwrap_or(0)];
        self.mcts.root_visits(&mut v);
        v
    }

    fn select_action(&mut self, temperature: f64) -> Result<u32, BindError> {
        Ok(self.mcts.select_action(temperature)?)
    }

    fn calls(&self) -> u64 {
        self.mcts.calls()
    }

    fn tree_hash(&self) -> u64 {
        self.mcts.pool().tree_hash()
    }
}

/// Construction arguments of a search, as Python passes them.
#[derive(Debug, Clone)]
pub(crate) struct SearchArgs {
    pub(crate) capacity: u32,
    pub(crate) batch: u32,
    pub(crate) eviction: Eviction,
    pub(crate) puct: PuctConfig,
    pub(crate) seed: u64,
}

impl SearchArgs {
    /// Defaults: 4,096 nodes, batch 8, strict, the default PUCT config.
    pub(crate) fn new() -> Self {
        Self {
            capacity: 4_096,
            batch: 8,
            eviction: Eviction::Strict,
            puct: PuctConfig::default(),
            seed: 0,
        }
    }

    /// Sets root noise.
    pub(crate) fn with_noise(mut self, alpha: f64, epsilon: f64) -> Self {
        self.puct.noise = Some(RootNoise { alpha, epsilon });
        self
    }
}

fn nonzero(v: u32, what: &str) -> Result<NonZeroU32, BindError> {
    NonZeroU32::new(v).ok_or_else(|| BindError::Argument(format!("{what} must be at least 1")))
}

/// A Connect Four search (`tessera/Connect4-v1`), rooted at a fresh game.
pub(crate) fn connect4(args: &SearchArgs) -> Result<Box<dyn SearchOps>, BindError> {
    let cfg = Connect4Config {};
    let pool_options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(args.seed))
    };
    let mut importer = EnvPool::<Connect4, Masked>::with_masks(cfg.clone(), pool_options)?;
    importer.reset_all(Reseed::Keep)?;
    let root = importer.snapshot(0)?;
    let mut options = SearchOptions::new(
        nonzero(args.capacity, "capacity")?,
        nonzero(args.batch, "batch")?,
    );
    options.eviction = args.eviction;
    let pool = SearchPool::new(cfg.clone(), options, &root, EnvPool::with_masks)?;
    let mcts = Mcts::new(pool, args.puct)?;
    Ok(Box::new(Searcher { mcts, importer }))
}

impl From<SearchError> for BindError {
    fn from(e: SearchError) -> Self {
        match e {
            SearchError::Engine(e) => Self::Engine(e),
            e => Self::Search(e),
        }
    }
}
