// SPDX-License-Identifier: MIT OR Apache-2.0

//! The batched evaluator contract (ADR-0043 section 3).

use tessera_rl_core::Environment;
use tessera_rl_engine::Record;
use thiserror::Error;

/// Tolerance on the sum of an evaluator's priors over all actions.
pub const PRIOR_SUM_TOLERANCE: f64 = 1e-6;

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
