// SPDX-License-Identifier: MIT OR Apache-2.0

//! Buffer layouts derived from the spaces at construction (SPEC 5.2).

use tessera_rl_core::space::{BoxBounds, DType, Space};

use crate::error::EngineError;

/// Most observation leaves a pool supports; per-agent leaf views live on
/// the stack, so the hot path never allocates.
pub const MAX_LEAVES: usize = 16;

/// One action head.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Head {
    /// A discrete head with `n` choices, as `i32`.
    Discrete { n: u32 },
    /// A continuous head, as `f32`, inclusive bounds.
    Continuous { low: f64, high: f64 },
}

/// Action heads of one agent, in space order.
#[derive(Debug, Clone, Default)]
pub(crate) struct ActionLayout {
    pub(crate) discrete: Vec<u32>,
    pub(crate) continuous: Vec<(f64, f64)>,
}

impl ActionLayout {
    pub(crate) fn from_space(space: &Space) -> Result<Self, EngineError> {
        let mut layout = Self::default();
        layout.add(space)?;
        Ok(layout)
    }

    fn push(&mut self, head: Head) {
        match head {
            Head::Discrete { n } => self.discrete.push(n),
            Head::Continuous { low, high } => self.continuous.push((low, high)),
        }
    }

    fn add(&mut self, space: &Space) -> Result<(), EngineError> {
        match space {
            Space::Discrete { n } => self.push(Head::Discrete { n: *n }),
            Space::MultiDiscrete { nvec } => {
                for &n in nvec {
                    self.push(Head::Discrete { n });
                }
            }
            Space::MultiBinary { shape } => {
                for _ in 0..elements(shape)? {
                    self.push(Head::Discrete { n: 2 });
                }
            }
            Space::Box {
                shape,
                dtype: DType::F32,
                bounds: BoxBounds::Float { low, high },
            } => {
                let count = elements(shape)?;
                for i in 0..count {
                    match (pick(low, i), pick(high, i)) {
                        (Some(&low), Some(&high)) => self.push(Head::Continuous { low, high }),
                        _ => return Err(EngineError::UnsupportedSpace { what: "box bounds" }),
                    }
                }
            }
            Space::Box {
                dtype: DType::F64, ..
            } => {
                // Continuous actions travel as f32 (SPEC 5.2); f64 action
                // heads come later.
                return Err(EngineError::UnsupportedSpace {
                    what: "f64 box actions",
                });
            }
            Space::Box { .. } => {
                return Err(EngineError::UnsupportedSpace {
                    what: "integer box actions",
                });
            }
            Space::Dict(entries) => {
                for (_, inner) in entries {
                    self.add(inner)?;
                }
            }
            Space::Tuple(items) => {
                for inner in items {
                    self.add(inner)?;
                }
            }
            Space::Set { .. } => {
                return Err(EngineError::UnsupportedSpace {
                    what: "set actions",
                });
            }
        }
        Ok(())
    }
}

/// One observation leaf of one agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Leaf {
    /// Bytes per agent.
    pub(crate) bytes: usize,
}

/// Observation leaves in declaration order.
pub(crate) fn obs_leaves(space: &Space) -> Result<Vec<Leaf>, EngineError> {
    let mut leaves = Vec::new();
    add_leaves(space, &mut leaves)?;
    if leaves.len() > MAX_LEAVES {
        return Err(EngineError::UnsupportedSpace {
            what: "more than 16 observation leaves",
        });
    }
    Ok(leaves)
}

fn add_leaves(space: &Space, leaves: &mut Vec<Leaf>) -> Result<(), EngineError> {
    let leaf = |count: usize, size: usize| {
        count
            .checked_mul(size)
            .map(|bytes| Leaf { bytes })
            .ok_or(EngineError::UnsupportedSpace {
                what: "leaf too large",
            })
    };
    match space {
        Space::Discrete { .. } => leaves.push(leaf(1, DType::I32.size())?),
        Space::MultiDiscrete { nvec } => leaves.push(leaf(nvec.len(), DType::I32.size())?),
        Space::MultiBinary { shape } => leaves.push(leaf(elements(shape)?, DType::Bool.size())?),
        Space::Box { shape, dtype, .. } => leaves.push(leaf(elements(shape)?, dtype.size())?),
        Space::Dict(entries) => {
            for (_, inner) in entries {
                add_leaves(inner, leaves)?;
            }
        }
        Space::Tuple(items) => {
            for inner in items {
                add_leaves(inner, leaves)?;
            }
        }
        Space::Set { .. } => {
            return Err(EngineError::UnsupportedSpace {
                what: "set observations",
            });
        }
    }
    Ok(())
}

/// Bound `i`, broadcasting a single value.
fn pick(v: &[f64], i: usize) -> Option<&f64> {
    if v.len() == 1 { v.first() } else { v.get(i) }
}

fn elements(shape: &[u32]) -> Result<usize, EngineError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(usize::try_from(d).ok()?))
        .ok_or(EngineError::UnsupportedSpace {
            what: "shape too large",
        })
}
