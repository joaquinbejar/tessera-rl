// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Space` description and its validation.

use super::{BoxBounds, DType};
use crate::error::SpaceError;

/// Maximum nesting depth of `Dict`, `Tuple` and `Set` (ADR-0014).
pub const MAX_DEPTH: usize = 4;

/// Maximum number of choices of a discrete dimension: actions travel as
/// `i32` (SPEC 5.2), so values `0..n` must fit.
pub const MAX_CHOICES: u64 = 1 << 31;

/// An observation or action space. `Text`, `Graph` and `Sequence` are not
/// supported (ADR-0014).
#[derive(Debug, Clone, PartialEq)]
pub enum Space {
    /// `n` choices, `0..n`, with `1 <= n <= 2^31`.
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
    /// A box of the given shape and dtype with typed bounds.
    Box {
        /// Shape.
        shape: Vec<u32>,
        /// Element type.
        dtype: DType,
        /// Bounds, typed to match `dtype`.
        bounds: BoxBounds,
    },
    /// Named sub-spaces, in declaration order.
    Dict(Vec<(String, Space)>),
    /// Positional sub-spaces.
    Tuple(Vec<Space>),
    /// Up to `max` elements of `inner`, with a presence mask. Counts as one
    /// level of nesting.
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

fn check_choices(n: u32) -> Result<(), SpaceError> {
    if n == 0 || u64::from(n) > MAX_CHOICES {
        Err(SpaceError::InvalidChoices { n })
    } else {
        Ok(())
    }
}

impl Space {
    /// Validates the space: choices in `1..=2^31`, valid shapes, bounds
    /// typed and in range for the dtype, non-empty composites, unique dict
    /// keys and depth at most [`MAX_DEPTH`].
    ///
    /// # Errors
    ///
    /// Returns the first [`SpaceError`] found.
    pub fn validate(&self) -> Result<(), SpaceError> {
        self.validate_at(0)
    }

    fn validate_at(&self, depth: usize) -> Result<(), SpaceError> {
        match self {
            Self::Discrete { n } => check_choices(*n)?,
            Self::MultiDiscrete { nvec } => {
                if nvec.is_empty() {
                    return Err(SpaceError::Empty);
                }
                for &n in nvec {
                    check_choices(n)?;
                }
            }
            Self::MultiBinary { shape } => {
                check_shape(shape)?;
            }
            Self::Box {
                shape,
                dtype,
                bounds,
            } => {
                let elements = check_shape(shape)?;
                bounds.validate(*dtype, elements)?;
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
                let child = Self::enter(depth)?;
                if *max == 0 {
                    return Err(SpaceError::Empty);
                }
                inner.validate_at(child)?;
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
    use crate::space::{BoxBounds, DType};

    fn leaf() -> Space {
        Space::Discrete { n: 4 }
    }

    fn int_box(dtype: DType, low: i64, high: i64) -> Space {
        Space::Box {
            shape: vec![1],
            dtype,
            bounds: BoxBounds::Int {
                low: vec![low],
                high: vec![high],
            },
        }
    }

    fn float_box(dtype: DType, low: f64, high: f64) -> Space {
        Space::Box {
            shape: vec![1],
            dtype,
            bounds: BoxBounds::Float {
                low: vec![low],
                high: vec![high],
            },
        }
    }

    fn bad_bounds(s: &Space) -> bool {
        matches!(s.validate(), Err(SpaceError::InvalidBounds { .. }))
    }

    #[test]
    fn test_space_validate_accepts_valid_box() {
        let s = Space::Box {
            shape: vec![2, 3],
            dtype: DType::F32,
            bounds: BoxBounds::Float {
                low: vec![-1.0],
                high: vec![1.0],
            },
        };
        assert_eq!(s.validate(), Ok(()));
    }

    #[test]
    fn test_box_bounds_reject_values_outside_dtype() {
        // U8 [-1, 300] cannot even be written as UInt; written as Int it is a
        // variant mismatch, and 300 is out of range as UInt.
        assert!(bad_bounds(&int_box(DType::U8, -1, 300)));
        let u8_high = Space::Box {
            shape: vec![1],
            dtype: DType::U8,
            bounds: BoxBounds::UInt {
                low: vec![0],
                high: vec![300],
            },
        };
        assert!(bad_bounds(&u8_high));
        assert!(bad_bounds(&int_box(DType::Bool, 2, 3)));
        assert_eq!(int_box(DType::Bool, 0, 1).validate(), Ok(()));
        assert!(bad_bounds(&int_box(DType::I8, -129, 0)));
    }

    #[test]
    fn test_box_bounds_reject_variant_mismatch() {
        assert!(bad_bounds(&float_box(DType::I32, 0.5, 0.5)));
        assert!(bad_bounds(&int_box(DType::F32, 0, 1)));
    }

    #[test]
    fn test_box_bounds_i64_exact_beyond_2_pow_53() {
        let big = int_box(DType::I64, 1 << 53, (1 << 53) + 1);
        assert_eq!(big.validate(), Ok(()));
        let Space::Box {
            bounds: BoxBounds::Int { high, .. },
            ..
        } = &big
        else {
            panic!("int bounds");
        };
        assert_eq!(high[0], (1 << 53) + 1);
        assert_eq!(int_box(DType::I64, i64::MAX, i64::MAX).validate(), Ok(()));
    }

    #[test]
    fn test_box_bounds_float_rules() {
        assert_eq!(
            float_box(DType::F64, f64::NEG_INFINITY, f64::INFINITY).validate(),
            Ok(())
        );
        assert!(bad_bounds(&float_box(DType::F64, f64::NAN, 1.0)));
        assert!(bad_bounds(&float_box(
            DType::F64,
            f64::INFINITY,
            f64::INFINITY
        )));
        assert!(bad_bounds(&float_box(
            DType::F64,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY
        )));
        assert!(bad_bounds(&float_box(DType::F64, 2.0, 1.0)));
        assert_eq!(
            float_box(DType::F32, 0.0, f64::from(f32::MAX)).validate(),
            Ok(())
        );
        assert_eq!(float_box(DType::F64, 0.1, 0.1).validate(), Ok(()));
    }

    #[test]
    fn test_f32_bounds_must_be_exact() {
        let inexact = |s: &Space| matches!(s.validate(), Err(SpaceError::InexactF32Bound { .. }));
        assert!(inexact(&float_box(DType::F32, 0.1, 0.1)));
        assert_eq!(
            float_box(DType::F32, f64::from(0.1f32), f64::from(0.1f32)).validate(),
            Ok(())
        );
        let next = f64::from(f32::from_bits(0.1f32.to_bits() + 1));
        assert_eq!(
            float_box(DType::F32, f64::from(0.1f32), next).validate(),
            Ok(())
        );
        assert!(inexact(&float_box(DType::F32, 0.0, 1e-46)));
        assert!(inexact(&float_box(DType::F32, 0.0, 1e39)));
        assert_eq!(float_box(DType::F32, -0.0, 0.0).validate(), Ok(()));
        assert_eq!(
            float_box(DType::F32, f64::NEG_INFINITY, f64::INFINITY).validate(),
            Ok(())
        );
    }

    #[test]
    fn test_box_bounds_per_element_lengths() {
        let s = Space::Box {
            shape: vec![2],
            dtype: DType::I16,
            bounds: BoxBounds::Int {
                low: vec![0, 1],
                high: vec![5],
            },
        };
        assert!(bad_bounds(&s));
    }

    #[test]
    fn test_discrete_choices_limit_is_2_pow_31() {
        let max = 1u32 << 31;
        assert_eq!(Space::Discrete { n: max }.validate(), Ok(()));
        assert_eq!(
            Space::Discrete { n: max + 1 }.validate(),
            Err(SpaceError::InvalidChoices { n: max + 1 })
        );
        assert_eq!(
            Space::Discrete { n: 0 }.validate(),
            Err(SpaceError::InvalidChoices { n: 0 })
        );
        assert_eq!(
            Space::MultiDiscrete { nvec: vec![3, max] }.validate(),
            Ok(())
        );
        assert_eq!(
            Space::MultiDiscrete {
                nvec: vec![3, max + 1]
            }
            .validate(),
            Err(SpaceError::InvalidChoices { n: max + 1 })
        );
    }

    #[test]
    fn test_space_validate_rejects_bad_shapes() {
        assert!(matches!(
            Space::MultiBinary { shape: vec![3, 0] }.validate(),
            Err(SpaceError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_space_validate_depth_limit_counts_sets() {
        let mut s = leaf();
        for _ in 0..MAX_DEPTH {
            s = Space::Tuple(vec![s]);
        }
        assert_eq!(s.validate(), Ok(()));
        let too_deep = Space::Set {
            inner: Box::new(s),
            max: 2,
        };
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
