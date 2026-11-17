// SPDX-License-Identifier: MIT OR Apache-2.0

//! The action heads of a space (SPEC 5.2, ADR-0062 item 6): one source of
//! truth for the head layout, used by the engine and by host environments.

use thiserror::Error;

use super::{BoxBounds, DType, Space};

/// The action heads of one agent, in space order.
///
/// Discrete heads travel as `i32` and continuous heads as `f32` (SPEC 5.2):
/// a `Discrete` or `MultiDiscrete` dimension is one discrete head, each
/// element of a `MultiBinary` space is a discrete head with two choices,
/// and each element of an `F32` box is a continuous head with inclusive
/// bounds. `Dict` and `Tuple` contribute their entries in order.
///
/// There is one entry per element: a `MultiBinary` or `F32` box space
/// gives as many heads as it has elements, unbounded here. A pool bounds
/// what it accepts when it builds its buffers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ActionHeads {
    discrete: Vec<u32>,
    continuous: Vec<(f64, f64)>,
}

impl ActionHeads {
    /// Choices of each discrete head, in order.
    #[must_use]
    pub fn discrete(&self) -> &[u32] {
        &self.discrete
    }

    /// Inclusive `(low, high)` bounds of each continuous head, in order.
    #[must_use]
    pub fn continuous(&self) -> &[(f64, f64)] {
        &self.continuous
    }

    /// The discrete and continuous heads, by value.
    #[must_use]
    pub fn into_parts(self) -> (Vec<u32>, Vec<(f64, f64)>) {
        (self.discrete, self.continuous)
    }
}

/// A space that cannot be an action space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("unsupported action space: {what}")]
pub struct HeadsError {
    what: &'static str,
}

impl HeadsError {
    /// What is unsupported, for example `"f64 box actions"`.
    #[must_use]
    pub const fn what(&self) -> &'static str {
        self.what
    }
}

impl Space {
    /// The action heads of one agent for this space.
    ///
    /// # Errors
    ///
    /// [`HeadsError`] for a space that cannot carry actions: `Set`, an
    /// integer or `F64` box, `F32` box bounds that do not cover every
    /// element, or a shape whose element count overflows `usize`.
    pub fn heads(&self) -> Result<ActionHeads, HeadsError> {
        let mut heads = ActionHeads::default();
        add(self, &mut heads)?;
        Ok(heads)
    }
}

fn add(space: &Space, heads: &mut ActionHeads) -> Result<(), HeadsError> {
    match space {
        Space::Discrete { n } => heads.discrete.push(*n),
        Space::MultiDiscrete { nvec } => heads.discrete.extend_from_slice(nvec),
        Space::MultiBinary { shape } => {
            for _ in 0..elements(shape)? {
                heads.discrete.push(2);
            }
        }
        Space::Box {
            shape,
            dtype: DType::F32,
            bounds: BoxBounds::Float { low, high },
        } => {
            for i in 0..elements(shape)? {
                match (pick(low, i), pick(high, i)) {
                    (Some(&low), Some(&high)) => heads.continuous.push((low, high)),
                    _ => return Err(HeadsError { what: "box bounds" }),
                }
            }
        }
        // Continuous actions travel as f32 (SPEC 5.2); f64 action heads
        // come later.
        Space::Box {
            dtype: DType::F64, ..
        } => {
            return Err(HeadsError {
                what: "f64 box actions",
            });
        }
        Space::Box { .. } => {
            return Err(HeadsError {
                what: "integer box actions",
            });
        }
        Space::Dict(entries) => {
            for (_, inner) in entries {
                add(inner, heads)?;
            }
        }
        Space::Tuple(items) => {
            for inner in items {
                add(inner, heads)?;
            }
        }
        Space::Set { .. } => {
            return Err(HeadsError {
                what: "set actions",
            });
        }
    }
    Ok(())
}

/// One bound per element, or one bound broadcast to every element.
fn pick(v: &[f64], i: usize) -> Option<&f64> {
    if v.len() == 1 { v.first() } else { v.get(i) }
}

fn elements(shape: &[u32]) -> Result<usize, HeadsError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(usize::try_from(d).ok()?))
        .ok_or(HeadsError {
            what: "shape too large",
        })
}

#[cfg(test)]
mod tests {
    use super::{ActionHeads, HeadsError};
    use crate::space::{BoxBounds, DType, Space};

    fn f32_box(shape: Vec<u32>, low: Vec<f64>, high: Vec<f64>) -> Space {
        Space::Box {
            shape,
            dtype: DType::F32,
            bounds: BoxBounds::Float { low, high },
        }
    }

    type Parts = (Vec<u32>, Vec<(f64, f64)>);

    fn parts(space: &Space) -> Result<Parts, HeadsError> {
        space.heads().map(ActionHeads::into_parts)
    }

    #[test]
    fn test_heads_discrete_one_head() {
        assert_eq!(parts(&Space::Discrete { n: 7 }), Ok((vec![7], vec![])));
    }

    #[test]
    fn test_heads_multidiscrete_one_head_per_dimension() {
        let s = Space::MultiDiscrete {
            nvec: vec![3, 5, 2],
        };
        assert_eq!(parts(&s), Ok((vec![3, 5, 2], vec![])));
    }

    #[test]
    fn test_heads_multibinary_two_choices_per_element() {
        let s = Space::MultiBinary { shape: vec![2, 2] };
        assert_eq!(parts(&s), Ok((vec![2, 2, 2, 2], vec![])));
    }

    #[test]
    fn test_heads_f32_box_per_element_and_broadcast_bounds() {
        let per = f32_box(vec![2], vec![-1.0, 0.0], vec![1.0, 2.0]);
        assert_eq!(parts(&per), Ok((vec![], vec![(-1.0, 1.0), (0.0, 2.0)])));
        let broadcast = f32_box(vec![3], vec![-2.0], vec![2.0]);
        assert_eq!(parts(&broadcast), Ok((vec![], vec![(-2.0, 2.0); 3])));
    }

    #[test]
    fn test_heads_dict_and_tuple_in_entry_order() {
        let s = Space::Dict(vec![
            ("a".to_owned(), Space::Discrete { n: 4 }),
            (
                "b".to_owned(),
                Space::Tuple(vec![
                    f32_box(vec![1], vec![-1.0], vec![1.0]),
                    Space::Discrete { n: 9 },
                ]),
            ),
        ]);
        assert_eq!(parts(&s), Ok((vec![4, 9], vec![(-1.0, 1.0)])));
    }

    #[test]
    fn test_heads_unsupported_spaces_error_with_engine_wording() {
        let what = |s: &Space| s.heads().map_err(|e| e.what());
        let int_box = Space::Box {
            shape: vec![1],
            dtype: DType::I32,
            bounds: BoxBounds::Int {
                low: vec![0],
                high: vec![1],
            },
        };
        let f64_box = Space::Box {
            shape: vec![1],
            dtype: DType::F64,
            bounds: BoxBounds::Float {
                low: vec![0.0],
                high: vec![1.0],
            },
        };
        let set = Space::Set {
            inner: Box::new(Space::Discrete { n: 2 }),
            max: 2,
        };
        let short = f32_box(vec![3], vec![0.0, 1.0], vec![1.0, 2.0]);
        assert_eq!(what(&int_box), Err("integer box actions"));
        assert_eq!(what(&f64_box), Err("f64 box actions"));
        assert_eq!(what(&set), Err("set actions"));
        assert_eq!(what(&short), Err("box bounds"));
    }
}
