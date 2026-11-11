// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a borrowed output view describes: which region of the shared
//! output storage, with which element type and shape (ADR-0003). No
//! `unsafe` here; the view objects live in `ffi::view_type`.

use core::ffi::CStr;
use core::ops::Range;

use tessera_rl_engine::{OutputDtype, OutputField};

use crate::env::Layout;

/// An element type as both export protocols name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Elem {
    /// Buffer protocol format (struct module syntax, little-endian).
    pub(crate) format: &'static CStr,
    /// Bytes per element.
    pub(crate) itemsize: usize,
    /// DLPack type code (`kDLInt` 0, `kDLUInt` 1, `kDLFloat` 2, `kDLBool` 6).
    pub(crate) dl_code: u8,
}

const fn elem(format: &'static CStr, itemsize: usize, dl_code: u8) -> Elem {
    Elem {
        format,
        itemsize,
        dl_code,
    }
}

/// The element type of an observation leaf, from its NumPy dtype name.
pub(crate) fn leaf_elem(dtype: &str) -> Option<Elem> {
    Some(match dtype {
        "bool" => elem(c"?", 1, 6),
        "int8" => elem(c"<b", 1, 0),
        "int16" => elem(c"<h", 2, 0),
        "int32" => elem(c"<i", 4, 0),
        "int64" => elem(c"<q", 8, 0),
        "uint8" => elem(c"<B", 1, 1),
        "uint16" => elem(c"<H", 2, 1),
        "uint32" => elem(c"<I", 4, 1),
        "float32" => elem(c"<f", 4, 2),
        "float64" => elem(c"<d", 8, 2),
        _ => return None,
    })
}

/// The element type of a non-observation region.
const fn region_elem(dtype: OutputDtype) -> Option<Elem> {
    match dtype {
        OutputDtype::Bool8 => Some(elem(c"?", 1, 6)),
        OutputDtype::F32 => Some(elem(c"<f", 4, 2)),
        OutputDtype::U32 => Some(elem(c"<I", 4, 1)),
        OutputDtype::U8 => Some(elem(c"<B", 1, 1)),
        _ => None,
    }
}

/// One view to export: a region, its element type and its shape.
#[derive(Debug, Clone)]
pub(crate) struct ViewSpec {
    pub(crate) field: OutputField,
    pub(crate) elem: Elem,
    /// The exported rows: `shape[0]` of them from `first_row`, out of the
    /// region's `total_rows` (a half's rows in split-phase modes).
    pub(crate) shape: Vec<usize>,
    pub(crate) first_row: usize,
    pub(crate) total_rows: usize,
}

/// The views of one call's outputs, keyed as the owned dict is.
#[derive(Debug, Clone)]
pub(crate) struct ViewSet {
    pub(crate) observations: Vec<ViewSpec>,
    /// Terminal observation leaves (`same_step` mode only, else empty).
    pub(crate) terminal_observations: Vec<ViewSpec>,
    pub(crate) others: Vec<(&'static str, ViewSpec)>,
}

/// The view set of `layout`; `None` for a leaf dtype the protocols cannot
/// name.
pub(crate) fn view_set(layout: &Layout) -> Option<ViewSet> {
    let (n, a) = (layout.num_envs, layout.agents);
    let leaf_views = |field: fn(usize) -> OutputField| {
        layout
            .leaves
            .iter()
            .enumerate()
            .map(|(i, leaf)| {
                let mut shape = vec![n, a];
                shape.extend(
                    leaf.shape
                        .iter()
                        .map(|&d| usize::try_from(d).unwrap_or(usize::MAX)),
                );
                Some(ViewSpec {
                    field: field(i),
                    elem: leaf_elem(leaf.dtype)?,
                    shape,
                    first_row: 0,
                    total_rows: n,
                })
            })
            .collect::<Option<Vec<_>>>()
    };
    let observations = leaf_views(OutputField::Observation)?;
    let terminal_observations = if layout.terminal {
        leaf_views(OutputField::TerminalObservation)?
    } else {
        Vec::new()
    };
    let spec = |field, dtype, shape: &[usize]| {
        Some(ViewSpec {
            field,
            elem: region_elem(dtype)?,
            shape: shape.to_vec(),
            first_row: 0,
            total_rows: n,
        })
    };
    let per_agent = [n, a];
    let others = vec![
        (
            "rewards",
            spec(OutputField::Rewards, OutputDtype::F32, &per_agent)?,
        ),
        (
            "agent_terminated",
            spec(OutputField::AgentTerminated, OutputDtype::Bool8, &per_agent)?,
        ),
        (
            "agent_truncated",
            spec(OutputField::AgentTruncated, OutputDtype::Bool8, &per_agent)?,
        ),
        (
            "alive",
            spec(OutputField::Alive, OutputDtype::Bool8, &per_agent)?,
        ),
        (
            "acting",
            spec(OutputField::Acting, OutputDtype::Bool8, &per_agent)?,
        ),
        (
            "episode_terminated",
            spec(OutputField::EpisodeTerminated, OutputDtype::Bool8, &[n])?,
        ),
        (
            "episode_truncated",
            spec(OutputField::EpisodeTruncated, OutputDtype::Bool8, &[n])?,
        ),
        (
            "masks",
            spec(
                OutputField::Masks,
                OutputDtype::Bool8,
                &[n, a, layout.mask_len],
            )?,
        ),
    ];
    Some(ViewSet {
        observations,
        terminal_observations,
        others,
    })
}

/// The byte offset and length of `rows` rows from `first_row`, in a region
/// of exactly `total_rows` rows of `row_bytes` bytes each and `region_bytes`
/// bytes in all; `None` if they do not fit or overflow.
#[must_use]
pub(crate) fn row_window(
    region_bytes: usize,
    row_bytes: usize,
    total_rows: usize,
    first_row: usize,
    rows: usize,
) -> Option<(usize, usize)> {
    if row_bytes.checked_mul(total_rows)? != region_bytes
        || first_row.checked_add(rows)? > total_rows
    {
        return None;
    }
    Some((
        row_bytes.checked_mul(first_row)?,
        row_bytes.checked_mul(rows)?,
    ))
}

/// The view set of the rows `rows` of `layout`; `None` as [`view_set`].
pub(crate) fn half_view_set(layout: &Layout, rows: &Range<usize>) -> Option<ViewSet> {
    let mut set = view_set(layout)?;
    let specs = set
        .observations
        .iter_mut()
        .chain(set.terminal_observations.iter_mut())
        .chain(set.others.iter_mut().map(|(_, spec)| spec));
    for spec in specs {
        spec.first_row = rows.start;
        *spec.shape.first_mut()? = rows.len();
    }
    Some(set)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::Leaf;

    #[test]
    fn test_row_window_stays_inside_the_region() {
        let region = [0u8; 24];
        for first in 0..=4 {
            for rows in 0..=4 - first {
                let (offset, len) = row_window(24, 6, 4, first, rows).unwrap();
                assert_eq!(
                    region.get(offset..offset + len).map(<[u8]>::len),
                    Some(rows * 6)
                );
            }
        }
        assert_eq!(row_window(24, 6, 4, 3, 2), None);
        assert_eq!(row_window(24, 6, 4, usize::MAX, 1), None);
        assert_eq!(row_window(25, 6, 4, 0, 1), None);
        assert_eq!(row_window(usize::MAX, usize::MAX, 2, 0, 1), None);
        assert_eq!(row_window(0, 0, 4, 1, 2), Some((0, 0)));
    }

    #[test]
    fn test_view_set_shapes_follow_the_layout() {
        let layout = Layout {
            num_envs: 3,
            agents: 2,
            initial_agents: 2,
            leaves: vec![Leaf {
                name: String::new(),
                dtype: "float32",
                shape: vec![4],
                low: vec![-1.0; 4],
                high: vec![1.0; 4],
            }],
            action_choices: 2,
            action_nvec: vec![2],
            action_bounds: None,
            mask_len: 5,
            mask_heads: vec![(0, vec![], 2)],
            entity_group_heads: 0,
            turns: "simultaneous",
            terminal: true,
        };
        let set = view_set(&layout).unwrap();
        assert_eq!(set.observations[0].shape, vec![3, 2, 4]);
        assert_eq!(set.observations[0].elem.itemsize, 4);
        let masks = &set.others.iter().find(|(k, _)| *k == "masks").unwrap().1;
        assert_eq!(masks.shape, vec![3, 2, 5]);
        assert_eq!(set.terminal_observations[0].shape, vec![3, 2, 4]);
        assert_eq!(leaf_elem("complex64"), None);
        let half = half_view_set(&layout, &(1..3)).unwrap();
        assert_eq!(half.observations[0].shape, vec![2, 2, 4]);
        assert_eq!(half.observations[0].first_row, 1);
        assert_eq!(half.observations[0].total_rows, 3);
    }
}
