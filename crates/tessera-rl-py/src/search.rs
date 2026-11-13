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
    DEFAULT_GROUP_BUDGET, Evaluator, EvaluatorError, Eviction, GumbelConfig, LeafBatch, Mcts,
    PuctConfig, RootNoise, RootSelection, SearchError, SearchGroup, SearchOptions, SearchPool,
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

/// How Python asks for an ONNX evaluator (ADR-0057).
#[derive(Debug, Clone)]
#[cfg_attr(
    not(feature = "onnx"),
    expect(dead_code, reason = "read only by the onnx feature's evaluator")
)]
pub(crate) struct OnnxArgs {
    /// The model file.
    pub(crate) model: std::path::PathBuf,
    /// The ONNX Runtime library, by explicit path.
    pub(crate) library: std::path::PathBuf,
    /// `accept_telemetry=True`.
    pub(crate) accept_telemetry: bool,
    /// The model size cap; the default when `None`.
    pub(crate) max_model_bytes: Option<u64>,
}

/// An ONNX evaluator's provenance, as Python sees it (ADR-0057 section 3).
#[derive(Debug, Clone)]
pub(crate) struct OnnxInfo {
    pub(crate) model_sha256: String,
    pub(crate) library_path: String,
    pub(crate) library_sha256: String,
    pub(crate) library_version: String,
    pub(crate) provider: &'static str,
    pub(crate) intra_threads: usize,
    pub(crate) inter_threads: usize,
    pub(crate) optimization_level: &'static str,
    pub(crate) telemetry_enabled: bool,
    pub(crate) arch: &'static str,
    pub(crate) cpu_model: Option<String>,
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
    /// Builds this search's ONNX evaluator (one session per search),
    /// replacing any earlier one, and returns its provenance.
    fn attach_onnx(&mut self, args: &OnnxArgs) -> Result<OnnxInfo, BindError>;
    /// One search call with the attached ONNX evaluator; `check` polls for
    /// interruptions before every batch, and every inference runs inside
    /// `detach` (the binding releases the interpreter there).
    fn run_onnx(
        &mut self,
        check: &mut dyn FnMut() -> Result<(), EvaluatorError>,
        detach: &mut dyn FnMut(&mut dyn FnMut()),
    ) -> Result<(), BindError>;
    /// The root's visit counts.
    fn root_visits(&self) -> Vec<u32>;
    /// The action to play (greedy at 0).
    fn select_action(&mut self, temperature: f64) -> Result<u32, BindError>;
    /// The last Gumbel call's improved policy, `None` under PUCT, before
    /// any call or once the root changed.
    fn improved_policy(&self) -> Option<Vec<f64>>;
    /// The last Gumbel call's considered actions `S_0`, `None` as for
    /// [`Self::improved_policy`].
    fn considered(&self) -> Option<Vec<u32>>;
    /// Completed calls.
    fn calls(&self) -> u64;
    /// The tree digest.
    fn tree_hash(&self) -> u64;
    /// Whether the search reads the full state of an imperfect-information
    /// environment (ADR-0046).
    fn is_privileged(&self) -> bool;
}

struct Searcher<E: Environment, M: MaskProvider<E>> {
    mcts: Mcts<E, M>,
    /// A one-instance pool that validates imported roots.
    importer: EnvPool<E, M>,
    /// The attached ONNX evaluator, if any.
    #[cfg(feature = "onnx")]
    onnx: Option<tessera_rl_onnx::OnnxEvaluator<E>>,
    /// The environment's config, for building the ONNX evaluator.
    #[cfg(feature = "onnx")]
    cfg: E::Config,
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

    #[cfg(feature = "onnx")]
    fn attach_onnx(&mut self, args: &OnnxArgs) -> Result<OnnxInfo, BindError> {
        let mut options = tessera_rl_onnx::OnnxOptions::new(&args.library);
        if args.accept_telemetry {
            options.telemetry = tessera_rl_onnx::TelemetryPolicy::Accept;
        }
        if let Some(cap) = args.max_model_bytes {
            options.max_model_bytes = cap;
        }
        let eval = tessera_rl_onnx::OnnxEvaluator::<E>::new(&self.cfg, &args.model, &options)
            .map_err(|e| BindError::Onnx(e.to_string()))?;
        let p = eval.provenance();
        let info = OnnxInfo {
            model_sha256: p.model_sha256.clone(),
            library_path: p.library_path.display().to_string(),
            library_sha256: p.library_sha256.clone(),
            library_version: p.library_version.clone(),
            provider: p.provider,
            intra_threads: p.intra_threads,
            inter_threads: p.inter_threads,
            optimization_level: p.optimization_level,
            telemetry_enabled: p.telemetry_enabled,
            arch: p.arch,
            cpu_model: p.cpu_model.clone(),
        };
        self.onnx = Some(eval);
        Ok(info)
    }

    #[cfg(not(feature = "onnx"))]
    fn attach_onnx(&mut self, _: &OnnxArgs) -> Result<OnnxInfo, BindError> {
        Err(not_built())
    }

    #[cfg(feature = "onnx")]
    fn run_onnx(
        &mut self,
        check: &mut dyn FnMut() -> Result<(), EvaluatorError>,
        detach: &mut dyn FnMut(&mut dyn FnMut()),
    ) -> Result<(), BindError> {
        let inner = self
            .onnx
            .as_mut()
            .ok_or_else(|| BindError::Onnx("no ONNX evaluator attached".to_owned()))?;
        self.mcts.run(&mut OnnxRun {
            inner,
            check,
            detach,
        })?;
        Ok(())
    }

    #[cfg(not(feature = "onnx"))]
    fn run_onnx(
        &mut self,
        _: &mut dyn FnMut() -> Result<(), EvaluatorError>,
        _: &mut dyn FnMut(&mut dyn FnMut()),
    ) -> Result<(), BindError> {
        Err(not_built())
    }

    fn root_visits(&self) -> Vec<u32> {
        let mut v = vec![0; usize::try_from(self.actions()).unwrap_or(0)];
        self.mcts.root_visits(&mut v);
        v
    }

    fn select_action(&mut self, temperature: f64) -> Result<u32, BindError> {
        Ok(self.mcts.select_action(temperature)?)
    }

    fn improved_policy(&self) -> Option<Vec<f64>> {
        self.mcts.improved_policy().map(<[f64]>::to_vec)
    }

    fn considered(&self) -> Option<Vec<u32>> {
        self.mcts.considered().map(<[u32]>::to_vec)
    }

    fn calls(&self) -> u64 {
        self.mcts.calls()
    }

    fn tree_hash(&self) -> u64 {
        self.mcts.pool().tree_hash()
    }

    fn is_privileged(&self) -> bool {
        self.mcts.is_privileged()
    }
}

/// Construction arguments of a search, as Python passes them.
#[derive(Debug, Clone)]
pub(crate) struct SearchArgs {
    pub(crate) capacity: u32,
    pub(crate) batch: u32,
    pub(crate) eviction: Eviction,
    /// The PUCT options; its `root` is set from the two fields below by
    /// [`Self::config`].
    pub(crate) puct: PuctConfig,
    /// Whether the root is the Gumbel root (`root="gumbel"`).
    pub(crate) gumbel_root: bool,
    /// The Gumbel root's options, used under `root="gumbel"` only.
    pub(crate) gumbel: GumbelConfig,
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
            gumbel_root: false,
            gumbel: GumbelConfig::default(),
            seed: 0,
        }
    }

    /// The search configuration, with the root selection applied.
    pub(crate) fn config(&self) -> PuctConfig {
        let mut cfg = self.puct;
        cfg.root = if self.gumbel_root {
            RootSelection::Gumbel(self.gumbel)
        } else {
            RootSelection::Puct
        };
        cfg
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

/// A Connect Four tree and its root importer.
type Connect4Tree = (Mcts<Connect4, Masked>, EnvPool<Connect4, Masked>);

/// A one-instance Connect Four pool that validates imported roots, and
/// its fresh-game root record.
fn connect4_importer(
    seed: u64,
) -> Result<(EnvPool<Connect4, Masked>, Record<Connect4>), BindError> {
    let pool_options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(seed))
    };
    let mut importer = EnvPool::<Connect4, Masked>::with_masks(Connect4Config {}, pool_options)?;
    importer.reset_all(Reseed::Keep)?;
    let root = importer.snapshot(0)?;
    Ok((importer, root))
}

/// A Connect Four tree rooted at a fresh game, and its root importer.
fn connect4_tree(args: &SearchArgs) -> Result<Connect4Tree, BindError> {
    let (importer, root) = connect4_importer(args.seed)?;
    let mut options = SearchOptions::new(
        nonzero(args.capacity, "capacity")?,
        nonzero(args.batch, "batch")?,
    );
    options.eviction = args.eviction;
    let pool = SearchPool::new(Connect4Config {}, options, &root, EnvPool::with_masks)?;
    let mcts = Mcts::new(pool, args.config())?;
    Ok((mcts, importer))
}

/// A Connect Four search (`tessera/Connect4-v1`), rooted at a fresh game.
pub(crate) fn connect4(args: &SearchArgs) -> Result<Box<dyn SearchOps>, BindError> {
    let (mcts, importer) = connect4_tree(args)?;
    Ok(Box::new(Searcher {
        mcts,
        importer,
        #[cfg(feature = "onnx")]
        onnx: None,
        #[cfg(feature = "onnx")]
        cfg: Connect4Config {},
    }))
}

/// The ONNX evaluator with the Python interruption poll, its inference run
/// with the interpreter released.
#[cfg(feature = "onnx")]
struct OnnxRun<'a, E: Environment> {
    inner: &'a mut tessera_rl_onnx::OnnxEvaluator<E>,
    check: &'a mut dyn FnMut() -> Result<(), EvaluatorError>,
    detach: &'a mut dyn FnMut(&mut dyn FnMut()),
}

#[cfg(feature = "onnx")]
impl<E: Environment> Evaluator<E> for OnnxRun<'_, E> {
    fn evaluate(
        &mut self,
        records: &[Record<E>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        self.inner.evaluate(records, priors, values)
    }

    fn evaluate_batch(
        &mut self,
        batch: &mut LeafBatch<'_, E>,
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        let n = batch.len();
        if n == 0 {
            return Ok(());
        }
        // Copied out of the scratch pool while attached; the inference
        // itself touches only the evaluator and these slices.
        let observations = batch.observations();
        let inner = &mut *self.inner;
        let mut result = Ok(());
        (self.detach)(&mut || result = inner.evaluate_rows(observations, n, priors, values));
        result
    }

    fn check(&mut self) -> Result<(), EvaluatorError> {
        (self.check)()
    }
}

/// The error of a build without the `onnx` feature.
#[cfg(not(feature = "onnx"))]
fn not_built() -> BindError {
    BindError::Onnx("this build of tessera-rl has no ONNX evaluator (the onnx feature)".to_owned())
}

/// The last failed group call, as Python sees it: its slot, its message,
/// and whether the exception raised was the user's own (the evaluator's,
/// or a signal handler's), which the bindings never touch (ADR-0045 §3).
#[derive(Debug, Clone)]
pub(crate) struct Failure {
    pub(crate) slot: Option<usize>,
    pub(crate) message: String,
    pub(crate) user_raised: bool,
}

/// What a search group can do, whatever the environment (ADR-0045).
pub(crate) trait GroupOps: Send {
    /// Slots.
    fn trees(&self) -> usize;
    /// Actions per node.
    fn actions(&self) -> u32;
    /// Bytes of one leaf's observation.
    fn obs_bytes(&self) -> usize;
    /// Replaces slot `i`'s tree by the root `record`.
    fn set_root(&mut self, i: usize, record: &[u8]) -> Result<(), BindError>;
    /// Sets whether slot `i` takes part in the next call.
    fn set_active(&mut self, i: usize, flag: bool) -> Result<(), BindError>;
    /// Whether slot `i` takes part in the next call.
    fn active(&self, i: usize) -> Result<bool, BindError>;
    /// One group call; on failure, the slot it belongs to.
    fn run(&mut self, eval: &mut dyn BatchEval) -> Result<(), (Option<usize>, BindError)>;
    /// Records whether the last failure's exception was the user's own.
    fn set_user_raised(&mut self, user: bool);
    /// The last failed call, cleared at the start of every call.
    fn last_failure(&self) -> Option<Failure>;
    /// Slot `i`'s root visit counts.
    fn root_visits(&self, i: usize) -> Result<Vec<u32>, BindError>;
    /// Slot `i`'s action to play.
    fn select_action(&mut self, i: usize, temperature: f64) -> Result<u32, BindError>;
    /// Slot `i`'s last Gumbel improved policy (see [`SearchOps`]).
    fn improved_policy(&self, i: usize) -> Result<Option<Vec<f64>>, BindError>;
    /// Slot `i`'s last Gumbel considered actions (see [`SearchOps`]).
    fn considered(&self, i: usize) -> Result<Option<Vec<u32>>, BindError>;
    /// Slot `i`'s completed calls.
    fn calls(&self, i: usize) -> Result<u64, BindError>;
    /// Slot `i`'s tree digest.
    fn tree_hash(&self, i: usize) -> Result<u64, BindError>;
    /// The aggregate memory bound, in bytes.
    fn memory_bytes(&self) -> u64;
}

struct GroupSearcher<E: Environment, M: MaskProvider<E>> {
    group: SearchGroup<E, M>,
    importer: EnvPool<E, M>,
    user_raised: bool,
}

fn slot_error(i: usize, trees: usize) -> BindError {
    BindError::Argument(format!("slot {i} out of range for {trees} trees"))
}

impl<E, M> GroupSearcher<E, M>
where
    E: Environment<
            Config: Sync + Clone,
            State: Send,
            Header: Send + RecordLayout<Adaptive: AdaptiveValues<E::Config>>,
        > + Validate,
    M: MaskProvider<E> + Send,
    E::Config: Send,
{
    fn tree(&self, i: usize) -> Result<&Mcts<E, M>, BindError> {
        self.group
            .slot(i)
            .ok_or_else(|| slot_error(i, self.group.len()))
    }

    fn tree_mut(&mut self, i: usize) -> Result<&mut Mcts<E, M>, BindError> {
        let n = self.group.len();
        self.group.slot_mut(i).ok_or_else(|| slot_error(i, n))
    }
}

impl<E, M> GroupOps for GroupSearcher<E, M>
where
    E: Environment<
            Config: Sync + Clone,
            State: Send,
            Header: Send + RecordLayout<Adaptive: AdaptiveValues<E::Config>>,
        > + Validate,
    M: MaskProvider<E> + Send,
    E::Config: Send,
{
    fn trees(&self) -> usize {
        self.group.len()
    }

    fn actions(&self) -> u32 {
        self.group.slot(0).map_or(0, |m| m.pool().actions())
    }

    fn obs_bytes(&self) -> usize {
        self.group.slot(0).map_or(0, Mcts::obs_bytes)
    }

    fn set_root(&mut self, i: usize, record: &[u8]) -> Result<(), BindError> {
        self.tree(i)?;
        let decoded = decode_from_slice::<DecodedRecord<E>>(record).map_err(BindError::Decode)?;
        self.importer.import(&[0], &[decoded])?;
        let root = self.importer.snapshot(0)?;
        self.tree_mut(i)?.pool_mut().set_root(&root)?;
        Ok(())
    }

    fn set_active(&mut self, i: usize, flag: bool) -> Result<(), BindError> {
        if self.group.set_active(i, flag) {
            Ok(())
        } else {
            Err(slot_error(i, self.group.len()))
        }
    }

    fn active(&self, i: usize) -> Result<bool, BindError> {
        self.group
            .active(i)
            .ok_or_else(|| slot_error(i, self.group.len()))
    }

    fn run(&mut self, eval: &mut dyn BatchEval) -> Result<(), (Option<usize>, BindError)> {
        self.user_raised = false;
        self.group
            .run(&mut Adapter(eval))
            .map_err(|e| (e.slot, e.error.into()))
    }

    fn set_user_raised(&mut self, user: bool) {
        self.user_raised = user;
    }

    fn last_failure(&self) -> Option<Failure> {
        self.group.last_failure().map(|f| Failure {
            slot: f.slot,
            message: f.message.clone(),
            user_raised: self.user_raised,
        })
    }

    fn root_visits(&self, i: usize) -> Result<Vec<u32>, BindError> {
        let m = self.tree(i)?;
        let mut v = vec![0; usize::try_from(m.pool().actions()).unwrap_or(0)];
        m.root_visits(&mut v);
        Ok(v)
    }

    fn select_action(&mut self, i: usize, temperature: f64) -> Result<u32, BindError> {
        Ok(self.tree_mut(i)?.select_action(temperature)?)
    }

    fn improved_policy(&self, i: usize) -> Result<Option<Vec<f64>>, BindError> {
        Ok(self.tree(i)?.improved_policy().map(<[f64]>::to_vec))
    }

    fn considered(&self, i: usize) -> Result<Option<Vec<u32>>, BindError> {
        Ok(self.tree(i)?.considered().map(<[u32]>::to_vec))
    }

    fn calls(&self, i: usize) -> Result<u64, BindError> {
        Ok(self.tree(i)?.calls())
    }

    fn tree_hash(&self, i: usize) -> Result<u64, BindError> {
        Ok(self.tree(i)?.pool().tree_hash())
    }

    fn memory_bytes(&self) -> u64 {
        self.group.memory_bytes()
    }
}

/// A Connect Four search group: one tree per entry of `slots`, each from
/// its own arguments, every root a fresh game.
pub(crate) fn connect4_group(slots: &[SearchArgs]) -> Result<Box<dyn GroupOps>, BindError> {
    let first = slots
        .first()
        .ok_or_else(|| BindError::Argument("trees must be at least 1".to_owned()))?;
    // The aggregate bound is checked from the arguments alone, before any
    // tree is allocated (ADR-0045 section 1).
    let shapes = slots
        .iter()
        .map(|a| {
            let mut o =
                SearchOptions::new(nonzero(a.capacity, "capacity")?, nonzero(a.batch, "batch")?);
            o.eviction = a.eviction;
            Ok((o, a.config()))
        })
        .collect::<Result<Vec<_>, BindError>>()?;
    let bound = SearchGroup::<Connect4, Masked>::bound_bytes(&Connect4Config {}, &shapes);
    match bound {
        Some(bytes) if bytes <= DEFAULT_GROUP_BUDGET => {}
        bytes => {
            return Err(BindError::Search(SearchError::Budget {
                bytes: bytes.unwrap_or(u64::MAX),
                per_node: SearchPool::<Connect4, Masked>::bytes_per_node(7),
                budget: DEFAULT_GROUP_BUDGET,
            }));
        }
    }
    // Only the small root importer: no tree is built and dropped.
    let (importer, _) = connect4_importer(first.seed)?;
    let trees = slots
        .iter()
        .map(|a| connect4_tree(a).map(|(m, _)| m))
        .collect::<Result<Vec<_>, _>>()?;
    let group = SearchGroup::new(&Connect4Config {}, trees, DEFAULT_GROUP_BUDGET)?;
    Ok(Box::new(GroupSearcher {
        group,
        importer,
        user_raised: false,
    }))
}

impl From<SearchError> for BindError {
    fn from(e: SearchError) -> Self {
        match e {
            SearchError::Engine(e) => Self::Engine(e),
            e => Self::Search(e),
        }
    }
}
