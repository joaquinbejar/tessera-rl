// SPDX-License-Identifier: MIT OR Apache-2.0

//! The model contract derived from the environment (ADR-0057 section 5):
//! one model input per observation leaf, named and typed as Python sees
//! the leaves, and the discrete action count.

use tessera_rl_core::space::{DType, Space};
use tessera_rl_search::observation_layout;

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

/// The observation leaves in the engine's order, one model input each:
/// boxes, dicts of them, and `Set`s of those (#505), from search's
/// [`observation_layout`], so the inputs are exactly the leaves of a search
/// row. A set `p` gives `p` (or `p/<key>`), shape `[max, ...]`, then
/// `p_mask`, `bool` `[max]`; absent slots arrive zeroed (ADR-0082 point 3).
/// A single unnamed box is the input `obs`.
pub(crate) fn leaf_inputs(space: &Space) -> Result<Vec<LeafInput>, OnnxError> {
    let (leaves, _) = observation_layout(space).ok_or_else(|| OnnxError::Contract {
        what: "the observation is not boxes, dicts of boxes and sets of those".to_owned(),
    })?;
    leaves
        .into_iter()
        .map(|l| {
            let elements = l.bytes.checked_div(l.dtype.size()).unwrap_or(0);
            Ok(LeafInput {
                name: if l.name.is_empty() {
                    "obs".to_owned()
                } else {
                    l.name
                },
                dtype: l.dtype,
                shape: l.shape,
                elements,
            })
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use tessera_rl_core::space::{BoxBounds, DType, Space};

    use super::leaf_inputs;

    #[test]
    fn test_layout_set_leaves_are_inputs_in_row_order() {
        let item = Space::Box {
            shape: vec![2],
            dtype: DType::I32,
            bounds: BoxBounds::Int {
                low: vec![0],
                high: vec![7],
            },
        };
        let agent = item.clone();
        let space = Space::Dict(vec![
            ("agent".to_owned(), agent),
            (
                "items".to_owned(),
                Space::Set {
                    inner: Box::new(item),
                    max: 8,
                },
            ),
        ]);
        let got: Vec<_> = leaf_inputs(&space)
            .unwrap()
            .into_iter()
            .map(|l| (l.name, l.dtype, l.shape, l.elements))
            .collect();
        assert_eq!(
            got,
            [
                ("agent".to_owned(), DType::I32, vec![2], 2),
                ("items".to_owned(), DType::I32, vec![8, 2], 16),
                ("items_mask".to_owned(), DType::Bool, vec![8], 8),
            ]
        );
    }

    #[test]
    fn test_layout_set_with_a_dict_inner_names_its_keys() {
        let b = |n: u32| Space::Box {
            shape: vec![n],
            dtype: DType::F32,
            bounds: BoxBounds::Float {
                low: vec![0.0],
                high: vec![1.0],
            },
        };
        let space = Space::Dict(vec![(
            "units".to_owned(),
            Space::Set {
                inner: Box::new(Space::Dict(vec![
                    ("pos".to_owned(), b(2)),
                    ("hp".to_owned(), b(1)),
                ])),
                max: 4,
            },
        )]);
        let names: Vec<_> = leaf_inputs(&space)
            .unwrap()
            .into_iter()
            .map(|l| l.name)
            .collect();
        assert_eq!(names, ["units/pos", "units/hp", "units_mask"]);
    }

    #[test]
    fn test_layout_unnamed_box_is_obs() {
        let space = Space::Box {
            shape: vec![3],
            dtype: DType::U8,
            bounds: BoxBounds::UInt {
                low: vec![0],
                high: vec![1],
            },
        };
        assert_eq!(leaf_inputs(&space).unwrap()[0].name, "obs");
    }
}
