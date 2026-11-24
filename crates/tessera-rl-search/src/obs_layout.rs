// SPDX-License-Identifier: MIT OR Apache-2.0

//! The observation leaves of an evaluator row, and the `Set`s among them
//! (ADR-0082 points 2 and 6, #490), read once from the scratch pool's
//! exported observation space.
//!
//! A row of [`crate::LeafBatch::observations`] is one agent's exported
//! leaves concatenated in layout order: a `Box` is one leaf, a `Dict`
//! contributes its entries in order, and a `Set` named `p` contributes its
//! item leaves (`p`, or `p/<key>`, shape `[max, ...]`) and then `p_mask`
//! (`bool`, `[max]`), whose absent slots the engine zeroes.

use tessera_rl_core::space::{DType, Space};

/// One exported leaf of an evaluator row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafLayout {
    /// The exported name (`""` for a single unnamed leaf).
    pub name: String,
    /// Element type.
    pub dtype: DType,
    /// Shape of one agent's leaf.
    pub shape: Vec<u32>,
    /// First byte in the row.
    pub offset: usize,
    /// Bytes in the row.
    pub bytes: usize,
}

/// One `Set` of the observation: its item leaves and its mask leaf, as
/// indices into the leaves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetLayout {
    /// The set's name.
    pub path: String,
    /// Its capacity `MAX`.
    pub max: u32,
    /// Its item leaves.
    pub items: Vec<usize>,
    /// Its `bool` mask leaf (`path_mask`).
    pub mask: usize,
}

/// The leaves and sets of `space` (an exported observation space), in the
/// order of an evaluator row, or `None` for a space evaluators do not
/// receive as leaves (anything but boxes, dicts of them, and sets of
/// those) or a size that overflows. The one walk that search, the ONNX
/// evaluator (#505) and evaluators of other crates share, so leaf names,
/// order and byte widths cannot drift between them.
#[must_use]
pub fn observation_layout(space: &Space) -> Option<(Vec<LeafLayout>, Vec<SetLayout>)> {
    let mut leaves = Vec::new();
    let mut sets = Vec::new();
    walk(space, "", &mut leaves, &mut sets)?;
    let mut offset = 0usize;
    for leaf in &mut leaves {
        leaf.offset = offset;
        offset = offset.checked_add(leaf.bytes)?;
    }
    Some((leaves, sets))
}

fn walk(
    space: &Space,
    name: &str,
    out: &mut Vec<LeafLayout>,
    sets: &mut Vec<SetLayout>,
) -> Option<()> {
    match space {
        Space::Box { shape, dtype, .. } => {
            let elements = shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(usize::try_from(d).ok()?))?;
            out.push(LeafLayout {
                name: name.to_owned(),
                dtype: *dtype,
                shape: shape.clone(),
                offset: 0,
                bytes: elements.checked_mul(dtype.size())?,
            });
        }
        Space::Dict(entries) => {
            for (key, inner) in entries {
                walk(inner, key, out, sets)?;
            }
        }
        Space::Set { inner, max } => {
            let mut items = Vec::new();
            let mut nested = Vec::new();
            walk(inner, "", &mut items, &mut nested)?;
            if !nested.is_empty() {
                return None;
            }
            let copies = usize::try_from(*max).ok()?;
            let first = out.len();
            for item in items {
                let mut shape = vec![*max];
                shape.extend(&item.shape);
                out.push(LeafLayout {
                    name: if item.name.is_empty() {
                        name.to_owned()
                    } else {
                        format!("{name}/{}", item.name)
                    },
                    dtype: item.dtype,
                    shape,
                    offset: 0,
                    bytes: item.bytes.checked_mul(copies)?,
                });
            }
            let mask = out.len();
            out.push(LeafLayout {
                name: format!("{name}_mask"),
                dtype: DType::Bool,
                shape: vec![*max],
                offset: 0,
                bytes: copies,
            });
            sets.push(SetLayout {
                path: name.to_owned(),
                max: *max,
                items: (first..mask).collect(),
                mask,
            });
        }
        _ => return None,
    }
    Some(())
}
