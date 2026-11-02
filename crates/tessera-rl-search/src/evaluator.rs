// SPDX-License-Identifier: MIT OR Apache-2.0

//! The batched evaluator contract (ADR-0043 section 3).

use tessera_rl_core::Environment;
use tessera_rl_engine::Record;

use crate::pool::ObsSource;
use thiserror::Error;

/// Tolerance on the sum of an evaluator's priors over all actions.
pub const PRIOR_SUM_TOLERANCE: f64 = 1e-6;

/// One batch of leaves for an evaluator, in batch order: each leaf's
/// record, the observation of the player to move at the leaf (the
/// observation leaves of the environment concatenated in order, as the
/// engine writes them), and its legal actions.
///
/// The observations are copied out of the scratch pool on the first call
/// to [`Self::observations`], into a preallocated buffer: an evaluator
/// that never asks for them does not pay for them.
#[derive(Debug)]
pub struct LeafBatch<'a, E: Environment> {
    records: &'a [Record<E>],
    source: ObsSource<'a>,
    /// Each leaf's scratch row and player to move.
    picks: &'a [(usize, usize)],
    buf: &'a mut [u8],
    filled: bool,
    obs_bytes: usize,
    legal: &'a [bool],
    actions: usize,
}

impl<'a, E: Environment> LeafBatch<'a, E> {
    /// `buf` holds `records.len() * obs_bytes` bytes.
    pub(crate) fn new(
        records: &'a [Record<E>],
        source: ObsSource<'a>,
        picks: &'a [(usize, usize)],
        buf: &'a mut [u8],
        obs_bytes: usize,
        legal: &'a [bool],
        actions: usize,
    ) -> Self {
        debug_assert_eq!(picks.len(), records.len());
        debug_assert_eq!(buf.len(), records.len() * obs_bytes);
        Self {
            records,
            source,
            picks,
            buf,
            filled: false,
            obs_bytes,
            legal,
            actions,
        }
    }

    /// Leaves in the batch.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the batch is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The leaves' records.
    #[must_use]
    pub const fn records(&self) -> &'a [Record<E>] {
        self.records
    }

    /// Bytes of one leaf's observation.
    #[must_use]
    pub const fn obs_bytes(&self) -> usize {
        self.obs_bytes
    }

    /// Every leaf's observation, `len() * obs_bytes()` bytes, row-major:
    /// the observation of the player to move at the leaf. Copied on the
    /// first call, without allocating.
    pub fn observations(&mut self) -> &[u8] {
        if !self.filled {
            let ob = self.obs_bytes;
            for (leaf, &(row, mover)) in self.picks.iter().enumerate() {
                if let Some(out) = self.buf.get_mut(leaf * ob..leaf * ob + ob) {
                    self.source.copy(row, mover, out);
                }
            }
            self.filled = true;
        }
        self.buf
    }

    /// The legal actions of every leaf, `len() * actions`, row-major (the
    /// mask of the player to move; all true without masks).
    #[must_use]
    pub const fn legal(&self) -> &'a [bool] {
        self.legal
    }

    /// Actions per leaf.
    #[must_use]
    pub const fn actions(&self) -> usize {
        self.actions
    }
}

/// A batched evaluator: priors and a value for every leaf of a batch.
///
/// `leaves` holds the leaf records in batch order. The evaluator writes
/// `priors`, `leaves.len() * actions` values, row-major and over **all**
/// actions, and `values`, one per leaf, each from the view of the player
/// to move at the leaf, in `[-1, 1]`. The search validates both before
/// using them (see [`EvaluatorError`]).
pub trait Evaluator<E: Environment> {
    /// Evaluates one batch.
    ///
    /// # Errors
    ///
    /// [`EvaluatorError::Failed`] (or any other variant) aborts the search
    /// call, which then leaves the tree as it was.
    fn evaluate(
        &mut self,
        leaves: &[Record<E>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError>;

    /// Evaluates one batch with its observations and legal actions. The
    /// search calls this; the default delegates to [`Self::evaluate`] with
    /// the records, so an evaluator that needs no observation implements
    /// only that.
    ///
    /// # Errors
    ///
    /// As [`Self::evaluate`].
    fn evaluate_batch(
        &mut self,
        batch: &mut LeafBatch<'_, E>,
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        self.evaluate(batch.records(), priors, values)
    }

    /// Called before every batch, whether or not it has leaves to
    /// evaluate: an error aborts the search call and rolls it back, like
    /// an evaluator error. A host polls for interruptions here (a pending
    /// Ctrl-C); the default never aborts.
    ///
    /// # Errors
    ///
    /// Any [`EvaluatorError`], which aborts the call.
    fn check(&mut self) -> Result<(), EvaluatorError> {
        Ok(())
    }
}

/// Why an evaluator's response was rejected (ADR-0043 section 3). The
/// search call that received it leaves the tree as before the call.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum EvaluatorError {
    /// The evaluator itself failed (for example a Python exception).
    #[error("the evaluator failed: {reason}")]
    Failed {
        /// What failed.
        reason: String,
    },
    /// The response has the wrong shape or dtype.
    #[error("the evaluator returned a wrong shape or dtype: {what}")]
    Shape {
        /// What is wrong.
        what: String,
    },
    /// A prior or value is NaN or infinite.
    #[error("leaf {leaf}: a prior or value is not finite")]
    NonFinite {
        /// Leaf in batch order.
        leaf: usize,
    },
    /// A value outside `[-1, 1]`.
    #[error("leaf {leaf}: value outside [-1, 1]")]
    ValueOutOfRange {
        /// Leaf in batch order.
        leaf: usize,
    },
    /// A negative prior.
    #[error("leaf {leaf}: negative prior")]
    NegativePrior {
        /// Leaf in batch order.
        leaf: usize,
    },
    /// Priors over all actions do not sum to `1 ± 1e-6`.
    #[error("leaf {leaf}: priors do not sum to 1 over all actions")]
    PriorSum {
        /// Leaf in batch order.
        leaf: usize,
    },
    /// The legal actions have no prior mass.
    #[error("leaf {leaf}: no prior mass on the legal actions")]
    NoLegalMass {
        /// Leaf in batch order.
        leaf: usize,
    },
}

/// Validates one leaf's response (before masking), then writes the
/// priors renormalized over the legal actions into `out` (illegal
/// actions get 0).
pub(crate) fn validate_leaf(
    leaf: usize,
    priors: &[f64],
    value: f64,
    legal: &[bool],
    out: &mut [f64],
) -> Result<(), EvaluatorError> {
    if !value.is_finite() || priors.iter().any(|p| !p.is_finite()) {
        return Err(EvaluatorError::NonFinite { leaf });
    }
    if !(-1.0..=1.0).contains(&value) {
        return Err(EvaluatorError::ValueOutOfRange { leaf });
    }
    if priors.iter().any(|&p| p < 0.0) {
        return Err(EvaluatorError::NegativePrior { leaf });
    }
    let sum: f64 = priors.iter().sum();
    if (sum - 1.0).abs() > PRIOR_SUM_TOLERANCE {
        return Err(EvaluatorError::PriorSum { leaf });
    }
    let mass: f64 = priors
        .iter()
        .zip(legal)
        .filter(|&(_, &l)| l)
        .map(|(p, _)| p)
        .sum();
    if mass <= 0.0 {
        return Err(EvaluatorError::NoLegalMass { leaf });
    }
    for ((o, &p), &l) in out.iter_mut().zip(priors).zip(legal) {
        *o = if l { p / mass } else { 0.0 };
    }
    Ok(())
}
