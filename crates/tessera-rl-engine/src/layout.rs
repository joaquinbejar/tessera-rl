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
