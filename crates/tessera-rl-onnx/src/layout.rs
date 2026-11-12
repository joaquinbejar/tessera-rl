// SPDX-License-Identifier: MIT OR Apache-2.0

//! The model contract derived from the environment (ADR-0057 section 5):
//! one model input per observation leaf, named and typed as Python sees
//! the leaves, and the discrete action count.

use tessera_rl_core::space::{DType, Space};

use crate::error::OnnxError;

/// One observation leaf of one agent, as a model input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LeafInput {
    /// The model input's name: the leaf's key, or `obs` for an unnamed box.
    pub(crate) name: String,
    /// Element type.
    pub(crate) dtype: DType,
    /// Shape of one agent's observation (without the batch dimension).
    pub(crate) shape: Vec<u32>,
    /// Elements per leaf row.
    pub(crate) elements: usize,
}

impl LeafInput {
    /// Bytes per leaf row.
    pub(crate) fn bytes(&self) -> usize {
        self.elements.saturating_mul(self.dtype.size())
    }
}

/// The observation leaves in the engine's order: a box, or a dict of
/// boxes (the spaces Python exposes; anything else is refused).
pub(crate) fn leaf_inputs(space: &Space) -> Result<Vec<LeafInput>, OnnxError> {
    let mut out = Vec::new();
    match space {
        Space::Dict(entries) => {
            for (key, inner) in entries {
                out.push(leaf(key, inner)?);
            }
        }
        other => out.push(leaf("obs", other)?),
    }
    Ok(out)
}

fn leaf(name: &str, space: &Space) -> Result<LeafInput, OnnxError> {
    let Space::Box { shape, dtype, .. } = space else {
        return Err(OnnxError::Contract {
            what: format!("observation leaf {name:?} is not a box"),
        });
    };
    let elements = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(usize::try_from(d).ok()?))
        .ok_or_else(|| OnnxError::Contract {
            what: format!("observation leaf {name:?} is too large"),
        })?;
    Ok(LeafInput {
        name: name.to_owned(),
        dtype: *dtype,
        shape: shape.clone(),
        elements,
    })
}

/// The number of actions of a `Discrete` action space.
pub(crate) fn action_count(space: &Space) -> Result<usize, OnnxError> {
    match space {
        Space::Discrete { n } => usize::try_from(*n).map_err(|_| OnnxError::Contract {
            what: "too many actions".to_owned(),
        }),
        _ => Err(OnnxError::Contract {
            what: "the ONNX evaluator needs a Discrete action space".to_owned(),
        }),
    }
}
