// SPDX-License-Identifier: MIT OR Apache-2.0

//! The value mode of a search (ADR-0096): `Terminal`, ADR-0043's bounded
//! values bit for bit, or `Returns`, discounted accumulated rewards with
//! min-max normalization for selection and the Gumbel root.

/// What a backed-up value is (ADR-0096 decision 1).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum ValueMode {
    /// ADR-0043: a leaf's value is the evaluator's in `[-1, 1]`, a
    /// terminal node's the final reward in `[-1, 1]`, step rewards are not
    /// used. The default.
    #[default]
    Terminal,
    /// ADR-0096 decision 2: the value of edge `(s, a)` into child `c` is
    /// `r(c) + discount * V(c)`, with `V` the evaluator's value at an
    /// evaluated leaf, `0` at a terminal node, or the next edge's value.
    /// Needs one mover on every path (decision 3).
    Returns {
        /// Finite and in `[0, 1]`.
        discount: f64,
    },
}

impl ValueMode {
    /// The discount under `Returns`, `None` under `Terminal`.
    #[must_use]
    pub(crate) const fn discount(self) -> Option<f64> {
        match self {
            Self::Terminal => None,
            Self::Returns { discount } => Some(discount),
        }
    }
}

/// Where a non-finite value came from (ADR-0096 decision 2a).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReturnSource {
    /// The evaluator's value at the leaf.
    Evaluator,
    /// A step reward on the path, or the return it made overflow.
    Reward,
}

/// The min and max of every backed-up edge mean in the call so far
/// (ADR-0096 decision 4), updated in backup order with exact `f64`
/// comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct MinMax {
    bounds: Option<(f64, f64)>,
}

impl MinMax {
    /// Widens the bounds to `q` (finite).
    pub(crate) fn update(&mut self, q: f64) {
        self.bounds = Some(match self.bounds {
            None => (q, q),
            Some((lo, hi)) => (if q < lo { q } else { lo }, if q > hi { q } else { hi }),
        });
    }

    /// `(q - min) / (max - min)` when `max > min`, else `0.5`.
    #[must_use]
    pub(crate) fn normalize(self, q: f64) -> f64 {
        match self.bounds {
            Some((lo, hi)) if hi > lo => (q - lo) / (hi - lo),
            _ => 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MinMax;

    #[test]
    fn test_minmax_empty_and_single_value_normalize_to_one_half() {
        let mut m = MinMax::default();
        assert_eq!(m.normalize(3.0).to_bits(), 0.5_f64.to_bits());
        m.update(3.0);
        assert_eq!(m.normalize(3.0).to_bits(), 0.5_f64.to_bits());
    }

    #[test]
    fn test_minmax_bounds_widen_in_both_directions() {
        let mut m = MinMax::default();
        for q in [2.0, -1.0, 5.0, 0.0] {
            m.update(q);
        }
        assert_eq!(m.normalize(-1.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(m.normalize(5.0).to_bits(), 1.0_f64.to_bits());
        assert_eq!(m.normalize(2.0).to_bits(), 0.5_f64.to_bits());
    }

    #[test]
    fn test_minmax_is_scale_invariant_for_a_power_of_two() {
        let (mut a, mut b) = (MinMax::default(), MinMax::default());
        for q in [0.3, -0.7, 0.125, 0.9] {
            a.update(q);
            b.update(q * 1024.0);
        }
        for q in [0.3, -0.7, 0.125, 0.9, 0.0] {
            assert_eq!(a.normalize(q).to_bits(), b.normalize(q * 1024.0).to_bits());
        }
    }
}
