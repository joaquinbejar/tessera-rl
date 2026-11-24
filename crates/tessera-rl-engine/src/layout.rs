// SPDX-License-Identifier: MIT OR Apache-2.0

//! Buffer layouts derived from the spaces at construction (SPEC 5.2).

use tessera_rl_core::space::{DType, Space};

use crate::error::EngineError;

/// Most observation leaves a pool supports; per-agent leaf views live on
/// the stack, so the hot path never allocates.
pub const MAX_LEAVES: usize = 16;

/// Action heads of one agent, in space order.
#[derive(Debug, Clone, Default)]
pub(crate) struct ActionLayout {
    pub(crate) discrete: Vec<u32>,
    pub(crate) continuous: Vec<(f64, f64)>,
    /// The mask row's layout; set at construction for masked pools.
    pub(crate) mask: MaskLayout,
}

/// Most discrete heads of an agent that declares entity groups: the size of
/// the zero-filled stack array the no-op is written into (ADR-0051).
pub const MAX_GROUPED_HEADS: usize = 1024;

/// One discrete head's place in the mask row.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MaskHead {
    /// First byte of the head's block.
    pub(crate) offset: usize,
    /// Choices.
    pub(crate) size: usize,
    /// `(parent head, stride)` for up to two parents.
    pub(crate) parents: [Option<(usize, usize)>; 2],
}

/// The shape of one head's block in the mask row, for consumers (SPEC 11.3):
/// `[parent 1?, parent 2?, choices]` starting at `offset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskHeadShape {
    /// First byte of the block in an agent's mask row.
    pub offset: usize,
    /// Sizes of the parents, outermost first.
    pub parent_sizes: [Option<usize>; 2],
    /// Choices of this head.
    pub size: usize,
}

/// The layout of one agent's mask row (SPEC 11.3, ADR-0014, ADR-0051).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct MaskLayout {
    pub(crate) heads: Vec<MaskHead>,
    /// Bytes per agent.
    pub(crate) len: usize,
    /// Heads per entity group; 0 when none are declared.
    pub(crate) group_heads: usize,
    /// Groups: 1 when none are declared.
    pub(crate) group_count: usize,
    /// Each head's choice count, as the action space gives them.
    pub(crate) sizes: Vec<u32>,
    /// No head has a parent: the row is the heads' blocks end to end, and
    /// the reader takes the flat path that predates conditional masks.
    pub(crate) flat: bool,
}

impl MaskLayout {
    /// Every head a root: offsets as today's concatenated row.
    #[cfg(test)]
    pub(crate) fn flat(sizes: &[u32]) -> Option<Self> {
        Self::build(sizes, &vec![[None, None]; sizes.len()], None, false).ok()?
    }

    /// `Ok(None)` when the row's length overflows; `Err` for a bad declaration.
    pub(crate) fn build(
        sizes: &[u32],
        parents: &[[Option<u16>; 2]],
        group_heads: Option<usize>,
        has_continuous: bool,
    ) -> Result<Option<Self>, EngineError> {
        let bad = |what| Err(EngineError::InvalidMaskLayout { what });
        let (group_heads, group_count) = match group_heads {
            None => (0, 1),
            Some(g) => {
                if has_continuous {
                    return bad("entity groups beside continuous heads");
                }
                if sizes.len() > MAX_GROUPED_HEADS {
                    return bad("more entity-group heads than MAX_GROUPED_HEADS");
                }
                if g == 0 || !sizes.len().is_multiple_of(g) {
                    return bad("the group size does not divide the discrete heads");
                }
                (g, sizes.len() / g)
            }
        };
        let size_of = |h: usize| sizes.get(h).and_then(|&n| usize::try_from(n).ok());
        let mut heads = Vec::with_capacity(sizes.len());
        let mut offset = 0usize;
        for (head, (&n, declared)) in sizes.iter().zip(parents).enumerate() {
            let Ok(size) = usize::try_from(n) else {
                return Ok(None);
            };
            if declared[0].is_none() && declared[1].is_some() {
                return bad("a head declares a second parent without a first");
            }
            if declared[0].is_some() && declared[0] == declared[1] {
                return bad("a head declares the same parent twice");
            }
            let mut list = [None, None];
            let mut stride = size;
            // Right to left: the last parent varies fastest.
            for (slot, parent) in declared.iter().enumerate().rev() {
                let Some(parent) = parent else { continue };
                let parent = usize::from(*parent);
                if parent >= head {
                    return bad("a head's parent is not an earlier head");
                }
                if group_heads != 0 && parent / group_heads != head / group_heads {
                    return bad("a head's parent lies outside its entity group");
                }
                let Some(parent_size) = size_of(parent) else {
                    return Ok(None);
                };
                if let Some(entry) = list.get_mut(slot) {
                    *entry = Some((parent, stride));
                }
                let Some(next) = stride.checked_mul(parent_size) else {
                    return Ok(None);
                };
                stride = next;
            }
            heads.push(MaskHead {
                offset,
                size,
                parents: list,
            });
            let Some(next) = offset.checked_add(stride) else {
                return Ok(None);
            };
            offset = next;
        }
        let flat = heads.iter().all(|h| h.parents == [None, None]);
        Ok(Some(Self {
            heads,
            len: offset,
            group_heads,
            group_count,
            sizes: sizes.to_vec(),
            flat,
        }))
    }

    /// Groups as a snapshot bound factor: 1 when none are declared.
    pub(crate) fn count(&self) -> u32 {
        u32::try_from(self.group_count.max(1)).unwrap_or(u32::MAX)
    }

    /// The public shape of `head`.
    pub(crate) fn shape(&self, head: usize, sizes: &[u32]) -> Option<MaskHeadShape> {
        let entry = self.heads.get(head)?;
        let parent = |p: Option<(usize, usize)>| {
            p.and_then(|(h, _)| sizes.get(h).and_then(|&n| usize::try_from(n).ok()))
        };
        Some(MaskHeadShape {
            offset: entry.offset,
            parent_sizes: [parent(entry.parents[0]), parent(entry.parents[1])],
            size: entry.size,
        })
    }
}

impl ActionLayout {
    /// The heads of `space`, from [`Space::heads`] (ADR-0062 item 6), with
    /// an empty mask layout.
    pub(crate) fn from_space(space: &Space) -> Result<Self, EngineError> {
        let (discrete, continuous) = space
            .heads()
            .map_err(|e| EngineError::UnsupportedSpace { what: e.what() })?
            .into_parts();
        Ok(Self {
            discrete,
            continuous,
            mask: MaskLayout::default(),
        })
    }
}

/// One observation leaf of one agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Leaf {
    /// Bytes per agent.
    pub(crate) bytes: usize,
}

/// Most `Set`s an observation can hold: each exports at least two leaves.
pub(crate) const MAX_SETS: usize = MAX_LEAVES / 2;

/// One `Set`'s leaves (ADR-0082): item leaves `first..mask`, then the mask.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SetSpan {
    /// First item leaf.
    pub(crate) first: usize,
    /// The mask leaf, right after the last item leaf.
    pub(crate) mask: usize,
    /// Capacity `MAX`.
    pub(crate) max: usize,
}

/// The `Set`s of an observation, in leaf order: a fixed array, so the
/// observation pass reads it without allocating.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SetSpans {
    spans: [SetSpan; MAX_SETS],
    len: usize,
    /// Bytes of one slot's row, per leaf index: nonzero only for item
    /// leaves, computed at pool build so the observation pass needs no
    /// division.
    rows: [usize; MAX_LEAVES],
}

impl SetSpans {
    /// The sets, in leaf order.
    #[inline]
    pub(crate) fn as_slice(&self) -> &[SetSpan] {
        self.spans.get(..self.len).unwrap_or(&[])
    }

    /// Test helper: sets and `(leaf, row bytes)` pairs.
    #[cfg(test)]
    pub(crate) fn from_parts(spans: &[SetSpan], rows: &[(usize, usize)]) -> Self {
        let mut out = Self::default();
        for &span in spans {
            let _ = out.push(span);
        }
        for &(leaf, row) in rows {
            if let Some(slot) = out.rows.get_mut(leaf) {
                *slot = row;
            }
        }
        out
    }

    /// Row bytes per leaf index (zero for leaves outside any set).
    #[inline]
    pub(crate) fn rows(&self) -> &[usize; MAX_LEAVES] {
        &self.rows
    }

    fn push(&mut self, span: SetSpan) -> Result<(), EngineError> {
        let slot = self
            .spans
            .get_mut(self.len)
            .ok_or(EngineError::UnsupportedSpace {
                what: "more than 16 observation leaves",
            })?;
        *slot = span;
        self.len += 1;
        Ok(())
    }
}

/// The observation leaves of a space, their exported names and its sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ObsLayout {
    pub(crate) leaves: Vec<Leaf>,
    /// Exported names, unique (ADR-0082 point 2): `Dict` keys and `Tuple`
    /// positions joined by `.`, empty for a space that is a single leaf; a
    /// set `p` exports `p` (single-leaf items) or `p/<inner path>`, then
    /// `p_mask`.
    pub(crate) names: Vec<String>,
    pub(crate) sets: SetSpans,
}

/// Observation leaves in declaration order, with their names and sets.
pub(crate) fn obs_leaves(space: &Space) -> Result<ObsLayout, EngineError> {
    let mut out = ObsLayout {
        leaves: Vec::new(),
        names: Vec::new(),
        sets: SetSpans::default(),
    };
    add_leaves(space, String::new(), false, &mut out)?;
    if out.leaves.len() > MAX_LEAVES {
        return Err(EngineError::UnsupportedSpace {
            what: "more than 16 observation leaves",
        });
    }
    if let Some(name) = duplicate(&out.names) {
        return Err(EngineError::DuplicateLeafName {
            name: name.to_owned(),
        });
    }
    Ok(out)
}

/// The first name that appears twice, if any.
fn duplicate(names: &[String]) -> Option<&str> {
    names
        .iter()
        .enumerate()
        .find(|(i, name)| names.iter().take(*i).any(|seen| seen == *name))
        .map(|(_, name)| name.as_str())
}

fn join(name: &str, part: &str) -> String {
    if name.is_empty() {
        part.to_owned()
    } else {
        format!("{name}.{part}")
    }
}

fn add_leaves(
    space: &Space,
    name: String,
    in_set: bool,
    out: &mut ObsLayout,
) -> Result<(), EngineError> {
    let leaf = |count: usize, size: usize| {
        count
            .checked_mul(size)
            .map(|bytes| Leaf { bytes })
            .ok_or(EngineError::UnsupportedSpace {
                what: "leaf too large",
            })
    };
    let one = match space {
        Space::Discrete { .. } => leaf(1, DType::I32.size())?,
        Space::MultiDiscrete { nvec } => leaf(nvec.len(), DType::I32.size())?,
        Space::MultiBinary { shape } => leaf(elements(shape)?, DType::Bool.size())?,
        Space::Box { shape, dtype, .. } => leaf(elements(shape)?, dtype.size())?,
        Space::Dict(entries) => {
            for (key, inner) in entries {
                add_leaves(inner, join(&name, key), in_set, out)?;
            }
            return Ok(());
        }
        Space::Tuple(items) => {
            for (i, inner) in items.iter().enumerate() {
                add_leaves(inner, join(&name, &i.to_string()), in_set, out)?;
            }
            return Ok(());
        }
        Space::Set { inner, max } => {
            if in_set {
                return Err(EngineError::UnsupportedSpace {
                    what: "a set inside a set",
                });
            }
            return add_set(inner, *max, &name, out);
        }
    };
    out.leaves.push(one);
    out.names.push(name);
    Ok(())
}

/// A set `p`: its item leaves, each `MAX` times its inner bytes, at `p` or
/// `p/<inner path>`, then `p_mask` (ADR-0082 point 2).
fn add_set(inner: &Space, max: u32, name: &str, out: &mut ObsLayout) -> Result<(), EngineError> {
    let too_large = || EngineError::UnsupportedSpace {
        what: "leaf too large",
    };
    let max = usize::try_from(max).map_err(|_| too_large())?;
    let first = out.leaves.len();
    add_leaves(inner, String::new(), true, out)?;
    let (leaves, names) = (
        out.leaves.get_mut(first..).unwrap_or(&mut []),
        out.names.get_mut(first..).unwrap_or(&mut []),
    );
    for (index, (leaf, path)) in leaves.iter_mut().zip(names.iter_mut()).enumerate() {
        // The inner bytes are one slot's row (ADR-0082 point 3).
        if let Some(row) = out.sets.rows.get_mut(first + index) {
            *row = leaf.bytes;
        }
        leaf.bytes = leaf.bytes.checked_mul(max).ok_or_else(too_large)?;
        *path = if path.is_empty() {
            name.to_owned()
        } else {
            format!("{name}/{path}")
        };
    }
    let mask = out.leaves.len();
    out.leaves.push(Leaf {
        bytes: max.checked_mul(DType::Bool.size()).ok_or_else(too_large)?,
    });
    out.names.push(format!("{name}_mask"));
    out.sets.push(SetSpan { first, mask, max })
}

fn elements(shape: &[u32]) -> Result<usize, EngineError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(usize::try_from(d).ok()?))
        .ok_or(EngineError::UnsupportedSpace {
            what: "shape too large",
        })
}

#[cfg(test)]
mod mask_tests {
    use super::{MAX_GROUPED_HEADS, MaskLayout};
    use crate::EngineError;
    use crate::pool::{first_illegal, first_illegal_conditional, first_illegal_flat};
    use tessera_rl_core::Bool8;

    /// The flat reader predates conditional masks: this is a cross-check
    /// between the two production readers, not an independent reference.
    fn reference(heads: &[i32], sizes: &[u32], mask: &[Bool8]) -> Option<usize> {
        first_illegal_flat(heads, sizes, mask)
    }

    fn refused(built: Result<Option<MaskLayout>, EngineError>, what: &str) {
        match built {
            Err(EngineError::InvalidMaskLayout { what: got }) => assert_eq!(got, what),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn test_reader_on_a_flat_grouped_layout_reads_only_its_range() {
        // Two groups of two root heads; head 0 illegal, group 1 legal.
        let l = MaskLayout::build(&[2; 4], &[[None, None]; 4], Some(2), false)
            .unwrap()
            .unwrap();
        assert!(l.flat);
        let mask: Vec<Bool8> = [false, true, true, true, true, true, true, true]
            .into_iter()
            .map(Bool8::new)
            .collect();
        let heads = [0, 0, 1, 1];
        assert_eq!(first_illegal(&heads, &l, 0..4, &mask), Some(0));
        assert_eq!(first_illegal(&heads, &l, 2..4, &mask), None);
    }

    #[test]
    fn test_reader_matches_todays_on_flat_layouts() {
        let mut s = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..50_000 {
            let n = usize::try_from(next() % 6 + 1).unwrap();
            let sizes: Vec<u32> = (0..n)
                .map(|_| u32::try_from(next() % 7 + 1).unwrap())
                .collect();
            let heads: Vec<i32> = sizes
                .iter()
                .map(|&z| i32::try_from(next() % u64::from(z)).unwrap())
                .collect();
            let len: u32 = sizes.iter().sum();
            let mask: Vec<Bool8> = (0..len).map(|_| Bool8::new(next() & 3 != 0)).collect();
            let flat = MaskLayout::flat(&sizes).unwrap();
            assert_eq!(flat.len, usize::try_from(len).unwrap());
            assert!(flat.flat);
            let want = reference(&heads, &sizes, &mask);
            // Both the general reader and the dispatch agree with it.
            assert_eq!(first_illegal_conditional(&heads, &flat, 0..n, &mask), want);
            assert_eq!(first_illegal(&heads, &flat, 0..n, &mask), want);
        }
    }

    #[test]
    fn test_two_parent_row_is_addressed_by_both_values() {
        let parents = [[None, None], [None, None], [Some(0), Some(1)]];
        let l = MaskLayout::build(&[2, 3, 4], &parents, None, false)
            .unwrap()
            .unwrap();
        assert_eq!(l.len, 2 + 3 + 2 * 3 * 4);
        assert_eq!(l.heads[2].offset, 5);
        assert_eq!(l.heads[2].parents, [Some((0, 12)), Some((1, 4))]);
        let mut mask = vec![Bool8::FALSE; l.len];
        mask[1] = Bool8::TRUE;
        mask[2 + 2] = Bool8::TRUE;
        // Row (1, 2), value 3: 5 + 1 * 12 + 2 * 4 + 3.
        mask[5 + 12 + 8 + 3] = Bool8::TRUE;
        assert_eq!(first_illegal(&[1, 2, 3], &l, 0..3, &mask), None);
        assert_eq!(first_illegal(&[1, 2, 2], &l, 0..3, &mask), Some(2));
    }

    #[test]
    fn test_illegal_parent_stops_before_its_child() {
        let parents = [[None, None], [Some(0), None]];
        let l = MaskLayout::build(&[2, 2], &parents, Some(2), false)
            .unwrap()
            .unwrap();
        // The child rows are all false; the parent's illegality is reported.
        let mask = [
            Bool8::TRUE,
            Bool8::FALSE,
            Bool8::FALSE,
            Bool8::FALSE,
            Bool8::FALSE,
            Bool8::FALSE,
        ];
        assert_eq!(first_illegal(&[1, 0], &l, 0..2, &mask), Some(0));
    }

    #[test]
    fn test_construction_refuses_bad_declarations() {
        let none = [[None, None]; 4];
        refused(
            MaskLayout::build(&[2; 4], &none, Some(2), true),
            "entity groups beside continuous heads",
        );
        refused(
            MaskLayout::build(&[2; 4], &none, Some(3), false),
            "the group size does not divide the discrete heads",
        );
        let forward = [[Some(1), None], [None, None], [None, None], [None, None]];
        refused(
            MaskLayout::build(&[2; 4], &forward, None, false),
            "a head's parent is not an earlier head",
        );
        let selfish = [[Some(0), None], [None, None], [None, None], [None, None]];
        refused(
            MaskLayout::build(&[2; 4], &selfish, None, false),
            "a head's parent is not an earlier head",
        );
        let across = [[None, None], [None, None], [Some(1), None], [None, None]];
        refused(
            MaskLayout::build(&[2; 4], &across, Some(2), false),
            "a head's parent lies outside its entity group",
        );
        let twice = [[None, None], [Some(0), Some(0)], [None, None], [None, None]];
        refused(
            MaskLayout::build(&[2; 4], &twice, None, false),
            "a head declares the same parent twice",
        );
        let gap = [[None, None], [None, Some(0)], [None, None], [None, None]];
        refused(
            MaskLayout::build(&[2; 4], &gap, None, false),
            "a head declares a second parent without a first",
        );
        let big = vec![2u32; MAX_GROUPED_HEADS + 1];
        let parents = vec![[None, None]; big.len()];
        refused(
            MaskLayout::build(&big, &parents, Some(1), false),
            "more entity-group heads than MAX_GROUPED_HEADS",
        );
        assert!(
            MaskLayout::build(
                &vec![2u32; MAX_GROUPED_HEADS],
                &vec![[None, None]; MAX_GROUPED_HEADS],
                Some(1),
                false
            )
            .is_ok()
        );
    }

    #[test]
    fn test_overflowing_row_is_reported_as_too_large() {
        let parents = [[None, None], [None, None], [Some(0), Some(1)]];
        assert_eq!(
            MaskLayout::build(&[u32::MAX, u32::MAX, u32::MAX], &parents, None, false).unwrap(),
            None
        );
    }
}

#[cfg(test)]
mod obs_tests {
    use super::{SetSpan, obs_leaves};
    use crate::EngineError;
    use tessera_rl_core::space::{BoxBounds, DType, Space};

    fn boxed(shape: Vec<u32>, dtype: DType) -> Space {
        let bounds = match dtype {
            DType::F32 | DType::F64 => BoxBounds::Float {
                low: vec![0.0],
                high: vec![1.0],
            },
            _ => BoxBounds::Int {
                low: vec![0],
                high: vec![1],
            },
        };
        Space::Box {
            shape,
            dtype,
            bounds,
        }
    }

    fn set(inner: Space, max: u32) -> Space {
        Space::Set {
            inner: Box::new(inner),
            max,
        }
    }

    fn dict(entries: Vec<(&str, Space)>) -> Space {
        Space::Dict(
            entries
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect(),
        )
    }

    #[test]
    fn test_obs_leaves_single_leaf_set_exports_items_then_mask() {
        let space = dict(vec![
            ("agent", boxed(vec![2], DType::I32)),
            ("items", set(boxed(vec![2], DType::I32), 8)),
        ]);
        let l = obs_leaves(&space).unwrap();
        assert_eq!(l.names, ["agent", "items", "items_mask"]);
        let bytes: Vec<usize> = l.leaves.iter().map(|leaf| leaf.bytes).collect();
        assert_eq!(bytes, [8, 8 * 8, 8]);
        assert_eq!(
            l.sets.as_slice(),
            [SetSpan {
                first: 1,
                mask: 2,
                max: 8
            }]
        );
    }

    #[test]
    fn test_obs_leaves_dict_inner_set_exports_paths_then_mask() {
        let inner = dict(vec![
            ("pos", boxed(vec![2], DType::I32)),
            ("hp", boxed(vec![1], DType::F32)),
        ]);
        let l = obs_leaves(&dict(vec![("units", set(inner, 16))])).unwrap();
        assert_eq!(l.names, ["units/pos", "units/hp", "units_mask"]);
        let bytes: Vec<usize> = l.leaves.iter().map(|leaf| leaf.bytes).collect();
        assert_eq!(bytes, [16 * 8, 16 * 4, 16]);
        assert_eq!(
            l.sets.as_slice(),
            [SetSpan {
                first: 0,
                mask: 2,
                max: 16
            }]
        );
    }

    #[test]
    fn test_obs_leaves_without_sets_keep_their_paths() {
        let space = dict(vec![
            ("a", boxed(vec![1], DType::U8)),
            ("b", Space::Tuple(vec![Space::Discrete { n: 2 }])),
        ]);
        let l = obs_leaves(&space).unwrap();
        assert_eq!(l.names, ["a", "b.0"]);
        assert_eq!(l.sets.as_slice(), []);
        assert_eq!(obs_leaves(&Space::Discrete { n: 2 }).unwrap().names, [""]);
    }

    #[test]
    fn test_obs_leaves_mask_name_beside_a_set_is_refused_as_duplicate() {
        let space = dict(vec![
            ("units", set(boxed(vec![1], DType::I32), 4)),
            ("units_mask", boxed(vec![4], DType::U8)),
        ]);
        assert_eq!(
            obs_leaves(&space).err(),
            Some(EngineError::DuplicateLeafName {
                name: "units_mask".to_owned()
            })
        );
        let inner = dict(vec![("x", boxed(vec![1], DType::I32))]);
        let space = dict(vec![
            ("a", set(inner, 2)),
            ("a/x", boxed(vec![1], DType::I32)),
        ]);
        assert_eq!(
            obs_leaves(&space).err(),
            Some(EngineError::DuplicateLeafName {
                name: "a/x".to_owned()
            })
        );
    }

    #[test]
    fn test_obs_leaves_set_inside_a_set_is_refused() {
        let nested = set(dict(vec![("s", set(Space::Discrete { n: 2 }, 2))]), 2);
        assert_eq!(
            obs_leaves(&nested).err(),
            Some(EngineError::UnsupportedSpace {
                what: "a set inside a set"
            })
        );
    }

    #[test]
    fn test_obs_leaves_set_leaves_count_toward_the_limit() {
        let names = ["a", "b", "c", "d", "e", "f", "g", "h"];
        let sets: Vec<(&str, Space)> = names
            .iter()
            .map(|n| (*n, set(Space::Discrete { n: 2 }, 2)))
            .collect();
        assert_eq!(obs_leaves(&dict(sets.clone())).unwrap().leaves.len(), 16);
        let mut over = sets;
        over.push(("z", Space::Discrete { n: 2 }));
        assert_eq!(
            obs_leaves(&dict(over)).err(),
            Some(EngineError::UnsupportedSpace {
                what: "more than 16 observation leaves"
            })
        );
    }
}
