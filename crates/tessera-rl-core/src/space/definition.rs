// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Space` description and its validation.

use super::DType;
use crate::error::SpaceError;

/// Maximum nesting depth of `Dict` and `Tuple` (ADR-0014).
pub const MAX_DEPTH: usize = 4;

/// An observation or action space. `Text`, `Graph` and `Sequence` are not
/// supported (ADR-0014).
#[derive(Debug, Clone, PartialEq)]
pub enum Space {
    /// `n` choices, `0..n`.
    Discrete {
        /// Number of choices.
        n: u32,
    },
    /// One discrete choice per entry of `nvec`.
    MultiDiscrete {
        /// Choices per dimension.
        nvec: Vec<u32>,
    },
    /// Independent binary values of the given shape.
    MultiBinary {
        /// Shape.
        shape: Vec<u32>,
    },
    /// A box of the given shape and dtype with per-element bounds, stored
    /// as `f64` (exact for every supported dtype except `i64` beyond 2^53).
    Box {
        /// Shape.
        shape: Vec<u32>,
        /// Element type.
        dtype: DType,
        /// Lower bounds, one per element, or one for all.
        low: Vec<f64>,
        /// Upper bounds, one per element, or one for all.
        high: Vec<f64>,
    },
    /// Named sub-spaces, in declaration order.
    Dict(Vec<(String, Space)>),
    /// Positional sub-spaces.
    Tuple(Vec<Space>),
    /// Up to `max` elements of `inner`, with a presence mask.
    Set {
        /// Element space.
        inner: Box<Space>,
        /// Capacity.
        max: u32,
    },
}

fn check_shape(shape: &[u32]) -> Result<u64, SpaceError> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(SpaceError::InvalidShape {
            shape: shape.to_vec(),
        });
    }
    shape
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(u64::from(d)))
        .ok_or_else(|| SpaceError::InvalidShape {
            shape: shape.to_vec(),
        })
}

impl Space {
    /// Validates the space: non-zero choices, valid shapes and bounds,
    /// non-empty composites, unique dict keys and depth at most
    /// [`MAX_DEPTH`].
    ///
    /// # Errors
    ///
    /// Returns the first [`SpaceError`] found.
    pub fn validate(&self) -> Result<(), SpaceError> {
        self.validate_at(0)
    }

    fn validate_at(&self, depth: usize) -> Result<(), SpaceError> {
        match self {
            Self::Discrete { n } => {
                if *n == 0 {
                    return Err(SpaceError::ZeroChoices);
                }
            }
            Self::MultiDiscrete { nvec } => {
                if nvec.is_empty() {
                    return Err(SpaceError::Empty);
                }
                if nvec.contains(&0) {
                    return Err(SpaceError::ZeroChoices);
                }
            }
            Self::MultiBinary { shape } => {
                check_shape(shape)?;
            }
            Self::Box {
                shape, low, high, ..
            } => {
                let elements = check_shape(shape)?;
                let fits = |v: &[f64]| v.len() == 1 || u64::try_from(v.len()) == Ok(elements);
                let ordered = low.iter().zip(high.iter()).all(|(l, h)| l <= h);
                if !fits(low) || !fits(high) || low.len() != high.len() || !ordered {
                    return Err(SpaceError::InvalidBounds);
                }
            }
            Self::Dict(entries) => {
                let child = Self::enter(depth)?;
                if entries.is_empty() {
                    return Err(SpaceError::Empty);
                }
                for (i, (key, space)) in entries.iter().enumerate() {
                    if entries.iter().take(i).any(|(k, _)| k == key) {
                        return Err(SpaceError::DuplicateKey { key: key.clone() });
                    }
                    space.validate_at(child)?;
                }
            }
            Self::Tuple(items) => {
                let child = Self::enter(depth)?;
                if items.is_empty() {
                    return Err(SpaceError::Empty);
                }
                for space in items {
                    space.validate_at(child)?;
                }
            }
            Self::Set { inner, max } => {
                if *max == 0 {
                    return Err(SpaceError::Empty);
                }
                inner.validate_at(depth)?;
            }
        }
        Ok(())
    }

    fn enter(depth: usize) -> Result<usize, SpaceError> {
        let child = depth.saturating_add(1);
        if child > MAX_DEPTH {
            Err(SpaceError::TooDeep {
                depth: child,
                max: MAX_DEPTH,
            })
        } else {
            Ok(child)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_DEPTH, Space};
    use crate::error::SpaceError;
    use crate::space::DType;

    fn leaf() -> Space {
        Space::Discrete { n: 4 }
    }

    #[test]
    fn test_space_validate_accepts_valid_box() {
        let s = Space::Box {
            shape: vec![2, 3],
            dtype: DType::F32,
            low: vec![-1.0],
            high: vec![1.0],
        };
        assert_eq!(s.validate(), Ok(()));
    }

    #[test]
    fn test_space_validate_rejects_bad_shapes_and_bounds() {
        assert_eq!(
            Space::Discrete { n: 0 }.validate(),
            Err(SpaceError::ZeroChoices)
        );
        assert!(matches!(
            Space::MultiBinary { shape: vec![3, 0] }.validate(),
            Err(SpaceError::InvalidShape { .. })
        ));
        let inverted = Space::Box {
            shape: vec![1],
            dtype: DType::U8,
            low: vec![5.0],
            high: vec![1.0],
        };
        assert_eq!(inverted.validate(), Err(SpaceError::InvalidBounds));
    }

    #[test]
    fn test_space_validate_depth_limit() {
        let mut s = leaf();
        for _ in 0..MAX_DEPTH {
            s = Space::Tuple(vec![s]);
        }
        assert_eq!(s.validate(), Ok(()));
        let too_deep = Space::Tuple(vec![s]);
        assert_eq!(
            too_deep.validate(),
            Err(SpaceError::TooDeep {
                depth: MAX_DEPTH + 1,
                max: MAX_DEPTH
            })
        );
    }

    #[test]
    fn test_space_validate_rejects_duplicate_dict_keys() {
        let d = Space::Dict(vec![("a".into(), leaf()), ("a".into(), leaf())]);
        assert_eq!(
            d.validate(),
            Err(SpaceError::DuplicateKey { key: "a".into() })
        );
    }
}
