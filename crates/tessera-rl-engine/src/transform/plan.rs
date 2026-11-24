// SPDX-License-Identifier: MIT OR Apache-2.0

//! A transform list checked against an observation space (ADR-0078 items 2
//! and 3, ADR-0092): the exported space, and per leaf the operations, their
//! byte sizes and their offsets into the frame-stack arena and the
//! statistics.

use tessera_rl_core::space::{BoxBounds, DType, Space};

use super::error::TransformError;
use super::spec::{FRAME_STACK_DEPTH, MAX_TRANSFORMED_ROW_BYTES, Transform, Transforms};
use crate::layout::{MAX_LEAVES, ObsLayout, SetSpan, SetSpans};

/// Bytes of a frame stack's header per agent: the ring position and the
/// generation of the agent the frames belong to (both `u32`).
pub(crate) const STACK_HEADER: usize = 8;

/// The key of a stacked set's generated age leaf (ADR-0092 point 5).
pub(crate) const FRAME_AGE: &str = "frame_age";
/// The key a stacked set gives a single-leaf (or `Tuple`) inner space.
pub(crate) const ITEM: &str = "item";

/// One operation on one leaf, in list order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Op {
    /// The ring of the last `k` frames of `frame` bytes, at `offset` in the
    /// agent's arena slice.
    Stack {
        frame: usize,
        k: usize,
        offset: usize,
    },
    /// `rows` rows of `elements` integers of `dtype` to `elements * n`
    /// booleans each.
    OneHot {
        dtype: DType,
        elements: usize,
        n: usize,
        rows: usize,
    },
    /// `rows` rows of `elements` values of `dtype` to `f32`, with the
    /// statistics from element `first` of the pool's normalized elements,
    /// shared by every row (ADR-0092 point 3).
    Normalize {
        dtype: DType,
        elements: usize,
        clip: f32,
        first: usize,
        rows: usize,
    },
}

impl Op {
    /// Bytes this operation writes for one agent.
    pub(crate) fn out_bytes(&self) -> usize {
        match *self {
            Self::Stack { frame, k, .. } => frame * k,
            Self::OneHot {
                elements, n, rows, ..
            } => rows * elements * n,
            Self::Normalize { elements, rows, .. } => rows * elements * 4,
        }
    }
}

/// The operations of one transformed leaf.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LeafPlan {
    /// The leaf's index in the raw layout (what `E::observe` writes).
    pub(crate) leaf: usize,
    /// The leaf's index in the exported layout.
    pub(crate) out: usize,
    /// Bytes `E::observe` writes.
    pub(crate) raw: usize,
    /// The raw index of the set mask whose absent rows the row ops re-zero
    /// (ADR-0092 point 2); `None` outside a set.
    pub(crate) mask: Option<usize>,
    /// The operations; empty for a set mask copied for its items' ops.
    pub(crate) ops: Vec<Op>,
}

/// The generated `frame_age` leaf of a stacked set (ADR-0092 point 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrameAge {
    /// The age leaf, exported index.
    pub(crate) out: usize,
    /// The stacked mask, exported index.
    pub(crate) mask: usize,
    /// Frames.
    pub(crate) k: usize,
    /// Slots per frame.
    pub(crate) max: usize,
}

/// A transform list made concrete for one observation space.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Plan {
    /// The exported observation space.
    pub(crate) space: Space,
    /// The transformed leaves, in raw leaf order.
    pub(crate) leaves: Vec<LeafPlan>,
    /// The generated age leaves, in leaf order.
    pub(crate) ages: Vec<FrameAge>,
    /// The exported index of each raw leaf.
    pub(crate) out_of_raw: [usize; MAX_LEAVES],
    /// Raw leaves (what `E::observe` writes).
    pub(crate) raw_count: usize,
    /// The sets of the raw layout, zeroed before the pipeline (ADR-0082
    /// point 3).
    pub(crate) raw_sets: SetSpans,
    /// Frame-stack bytes per agent (`0` without a `FrameStack`).
    pub(crate) agent_state: usize,
    /// Normalized elements per agent row (`0` without a `Normalize`).
    pub(crate) normalized: usize,
    /// The largest intermediate row of any leaf, for the scratch buffers.
    pub(crate) widest: usize,
    /// Raw bytes of all transformed leaves of one agent.
    pub(crate) raw_bytes: usize,
}

/// The set a raw leaf belongs to, with its path `p` (`p_mask` less the
/// suffix).
fn set_of(layout: &ObsLayout, index: usize) -> Option<(&SetSpan, &str)> {
    let span = layout
        .sets
        .as_slice()
        .iter()
        .find(|s| (s.first..=s.mask).contains(&index))?;
    let path = layout.names.get(span.mask)?.strip_suffix("_mask")?;
    Some((span, path))
}

/// The set whose path is `name`, if any.
fn set_named<'a>(layout: &'a ObsLayout, name: &str) -> Option<&'a SetSpan> {
    layout.sets.as_slice().iter().find(|s| {
        layout
            .names
            .get(s.mask)
            .and_then(|m| m.strip_suffix("_mask"))
            == Some(name)
    })
}

impl Plan {
    /// Checks `transforms` against `space` and its leaf layout.
    pub(crate) fn new(
        transforms: &Transforms,
        space: &Space,
        layout: &ObsLayout,
    ) -> Result<Self, TransformError> {
        let count = layout.names.len();
        let mut builder = Builder {
            agent_state: 0,
            elements_normalized: 0,
            normalized: vec![false; count],
            stacked: vec![false; count],
        };
        let (mut current, ops, set_stacks) = builder.ops(transforms, space, layout)?;
        let masked = masked_leaves(layout, &ops);
        let (out_of_raw, ages) = exported_indices(layout, &set_stacks);
        let mut leaves = Vec::new();
        let mut widest = 0usize;
        let mut raw_bytes = 0usize;
        let mut row = 0usize;
        for (index, (list, leaf)) in ops.into_iter().zip(&layout.leaves).enumerate() {
            let out = list.last().map_or(leaf.bytes, Op::out_bytes);
            row = row
                .checked_add(out)
                .ok_or(TransformError::RowTooLarge { bytes: usize::MAX })?;
            let in_set_plan = masked.get(index).copied().unwrap_or(false);
            if list.is_empty() && !in_set_plan {
                continue;
            }
            widest = list
                .iter()
                .map(Op::out_bytes)
                .fold(widest.max(leaf.bytes), usize::max);
            raw_bytes = raw_bytes
                .checked_add(leaf.bytes)
                .ok_or(TransformError::RowTooLarge { bytes: usize::MAX })?;
            let mask = set_of(layout, index)
                .filter(|(s, _)| s.mask != index)
                .map(|(s, _)| s.mask);
            leaves.push(LeafPlan {
                leaf: index,
                out: out_of_raw.get(index).copied().unwrap_or(index),
                raw: leaf.bytes,
                mask,
                ops: list,
            });
        }
        for age in &ages {
            row = row
                .checked_add(age.k * age.max)
                .ok_or(TransformError::RowTooLarge { bytes: usize::MAX })?;
        }
        if row > MAX_TRANSFORMED_ROW_BYTES {
            return Err(TransformError::RowTooLarge { bytes: row });
        }
        let mut exported = space.clone();
        let mut index = 0usize;
        replace_leaves(&mut exported, &mut current, &mut index, &set_stacks)?;
        let check =
            crate::layout::obs_leaves(&exported).map_err(|_| TransformError::Unsupported {
                transform: "FrameStack",
                leaf: String::new(),
                reason: "the stacked set exceeds 16 leaves",
            })?;
        if check.names.len() != count + ages.len() {
            return Err(TransformError::Internal);
        }
        Ok(Self {
            space: exported,
            leaves,
            ages,
            out_of_raw,
            raw_count: count,
            raw_sets: layout.sets,
            agent_state: builder.agent_state,
            normalized: builder.elements_normalized,
            widest,
            raw_bytes,
        })
    }

    /// Whether any operation keeps per-instance state.
    pub(crate) fn stateful(&self) -> bool {
        self.agent_state > 0
    }
}

/// The masks of every set whose leaves have ops: copied raw for the
/// re-zero (ADR-0092 point 2).
fn masked_leaves(layout: &ObsLayout, ops: &[Vec<Op>]) -> Vec<bool> {
    // A set whose items have ops gets its mask in the raw copy set
    // (ADR-0092 point 2), with no ops of its own when unstacked.
    let mut masked: Vec<bool> = vec![false; layout.names.len()];
    for span in layout.sets.as_slice() {
        let items = ops.get(span.first..span.mask).unwrap_or(&[]);
        let any = items.iter().any(|l| !l.is_empty())
            || ops.get(span.mask).is_some_and(|l| !l.is_empty());
        if any && let Some(flag) = masked.get_mut(span.mask) {
            *flag = true;
        }
    }
    masked
}

/// The exported index of each raw leaf and the generated age leaves:
/// each stacked set (`stacks`: mask index, depth) adds its age leaf right
/// before its mask (ADR-0092 point 5).
fn exported_indices(
    layout: &ObsLayout,
    stacks: &[(usize, usize)],
) -> ([usize; MAX_LEAVES], Vec<FrameAge>) {
    // Exported indices: each stacked set adds its age leaf before its
    // mask.
    let mut out_of_raw = [0usize; MAX_LEAVES];
    let mut shift = 0usize;
    let mut ages = Vec::new();
    for (index, out) in out_of_raw.iter_mut().enumerate().take(layout.names.len()) {
        if let Some(&(_, k)) = stacks.iter().find(|&&(m, _)| m == index) {
            let max = layout
                .sets
                .as_slice()
                .iter()
                .find(|s| s.mask == index)
                .map_or(0, |s| s.max);
            ages.push(FrameAge {
                out: index + shift,
                mask: index + shift + 1,
                k,
                max,
            });
            shift += 1;
        }
        *out = index + shift;
    }
    (out_of_raw, ages)
}

/// The running totals of a plan under construction.
struct Builder {
    agent_state: usize,
    elements_normalized: usize,
    normalized: Vec<bool>,
    stacked: Vec<bool>,
}

impl Builder {
    /// The ops of every leaf in list order, the leaves' final spaces and
    /// the stacked sets (mask index, depth).
    #[expect(clippy::type_complexity, reason = "three parts of one pass")]
    fn ops(
        &mut self,
        transforms: &Transforms,
        space: &Space,
        layout: &ObsLayout,
    ) -> Result<(Vec<Option<Space>>, Vec<Vec<Op>>, Vec<(usize, usize)>), TransformError> {
        let count = layout.names.len();
        // The current space of each leaf, as transforms change it: the
        // inner (one-slot) space for a set item.
        let mut current: Vec<Option<Space>> = vec![None; count];
        let mut ops: Vec<Vec<Op>> = vec![Vec::new(); count];
        // The depth of each stacked set, by mask index.
        let mut set_stacks: Vec<(usize, usize)> = Vec::new();
        for t in transforms.as_slice() {
            if let Transform::FrameStack { k, .. } = *t
                && let Some(span) = set_named(layout, t.leaf())
            {
                if set_stacks.iter().any(|&(m, _)| m == span.mask) {
                    return Err(TransformError::AfterSetStack {
                        leaf: t.leaf().to_owned(),
                    });
                }
                let k_usize = self.set_stack(t, k, span, space, &mut current, &mut ops)?;
                set_stacks.push((span.mask, k_usize));
                continue;
            }
            let index = layout
                .names
                .iter()
                .position(|n| n == t.leaf())
                .ok_or_else(|| TransformError::UnknownLeaf {
                    leaf: t.leaf().to_owned(),
                })?;
            let mut rows = 1usize;
            if let Some((span, path)) = set_of(layout, index) {
                if set_stacks.iter().any(|&(m, _)| m == span.mask) {
                    return Err(TransformError::AfterSetStack {
                        leaf: t.leaf().to_owned(),
                    });
                }
                if matches!(t, Transform::FrameStack { .. }) {
                    return Err(TransformError::PartialSetStack {
                        set: path.to_owned(),
                        leaf: t.leaf().to_owned(),
                    });
                }
                if index != span.mask {
                    rows = span.max;
                }
            }
            let slot = current.get_mut(index).ok_or(TransformError::Internal)?;
            let input = match slot.take() {
                Some(s) => s,
                None => leaf_space(space, index).ok_or(TransformError::Internal)?,
            };
            let (op, output) = self.op(t, &input, index, rows)?;
            if op.out_bytes() > MAX_TRANSFORMED_ROW_BYTES {
                return Err(too_large(t));
            }
            *slot = Some(output);
            if let Some(list) = ops.get_mut(index) {
                list.push(op);
            }
        }
        Ok((current, ops, set_stacks))
    }

    /// A ring of `k` frames of `bytes`, at the next arena offset.
    fn ring(&mut self, t: &Transform, bytes: usize, k: u32) -> Result<Op, TransformError> {
        if !FRAME_STACK_DEPTH.contains(&k) {
            return Err(TransformError::StackDepth { k });
        }
        let k_usize = usize::try_from(k).map_err(|_| TransformError::StackDepth { k })?;
        let offset = self.agent_state;
        self.agent_state = bytes
            .checked_mul(k_usize)
            .and_then(|b| b.checked_add(STACK_HEADER))
            .and_then(|b| self.agent_state.checked_add(b))
            .ok_or_else(|| too_large(t))?;
        Ok(Op::Stack {
            frame: bytes,
            k: k_usize,
            offset,
        })
    }

    /// A set-wide `FrameStack` (ADR-0092 point 5): one ring of depth `k`
    /// per leaf of `span`, items and mask; the item spaces stay their
    /// one-slot spaces (the set's `max` grows instead). Returns `k`.
    fn set_stack(
        &mut self,
        t: &Transform,
        k: u32,
        span: &SetSpan,
        space: &Space,
        current: &mut [Option<Space>],
        ops: &mut [Vec<Op>],
    ) -> Result<usize, TransformError> {
        for index in span.first..=span.mask {
            let input = match current.get(index).cloned().flatten() {
                Some(s) => s,
                None => leaf_space(space, index).ok_or(TransformError::Internal)?,
            };
            let rows = if index == span.mask { 1 } else { span.max };
            let bytes = space_bytes(&input)
                .and_then(|b| b.checked_mul(rows))
                .ok_or_else(|| too_large(t))?;
            let op = self.ring(t, bytes, k)?;
            if op.out_bytes() > MAX_TRANSFORMED_ROW_BYTES {
                return Err(too_large(t));
            }
            if let Some(flag) = self.stacked.get_mut(index) {
                *flag = true;
            }
            if let Some(slot) = current.get_mut(index) {
                *slot = Some(input);
            }
            if let Some(list) = ops.get_mut(index) {
                list.push(op);
            }
        }
        usize::try_from(k).map_err(|_| TransformError::StackDepth { k })
    }

    /// The operation of `t` on leaf `index`, whose current space is
    /// `input` (one row of `rows`), and the leaf's space after it.
    fn op(
        &mut self,
        t: &Transform,
        input: &Space,
        index: usize,
        rows: usize,
    ) -> Result<(Op, Space), TransformError> {
        let bytes = space_bytes(input).ok_or_else(|| too_large(t))?;
        let elements = elements(input).ok_or_else(|| too_large(t))?;
        let unsupported = |reason: &'static str| TransformError::Unsupported {
            transform: t.name(),
            leaf: t.leaf().to_owned(),
            reason,
        };
        let pair = match *t {
            Transform::FrameStack { k, .. } => {
                let op = self.ring(t, bytes, k)?;
                if let Some(flag) = self.stacked.get_mut(index) {
                    *flag = true;
                }
                (
                    op,
                    stack_space(input, k).ok_or_else(|| unsupported("a Dict, Tuple or Set"))?,
                )
            }
            Transform::OneHot { n, .. } => {
                if n == 0 {
                    return Err(unsupported("n must be at least 1"));
                }
                let dtype =
                    integer_dtype(input).ok_or_else(|| unsupported("not an integer leaf"))?;
                let n_usize = usize::try_from(n).map_err(|_| too_large(t))?;
                let mut shape =
                    flat_shape(input).ok_or_else(|| unsupported("not an integer leaf"))?;
                shape.push(n);
                elements
                    .checked_mul(n_usize)
                    .and_then(|e| e.checked_mul(rows))
                    .ok_or_else(|| too_large(t))?;
                (
                    Op::OneHot {
                        dtype,
                        elements,
                        n: n_usize,
                        rows,
                    },
                    bool_box(shape),
                )
            }
            Transform::Normalize { clip, .. } => {
                if !(clip.is_finite() && clip > 0.0) {
                    return Err(TransformError::Clip { clip });
                }
                if self.normalized.get(index).copied().unwrap_or(false) {
                    return Err(TransformError::NormalizeTwice {
                        leaf: t.leaf().to_owned(),
                    });
                }
                if self.stacked.get(index).copied().unwrap_or(false) {
                    return Err(TransformError::NormalizeAfterStack {
                        leaf: t.leaf().to_owned(),
                    });
                }
                let dtype =
                    numeric_dtype(input).ok_or_else(|| unsupported("not a numeric leaf"))?;
                let shape = flat_shape(input).ok_or_else(|| unsupported("not a numeric leaf"))?;
                elements
                    .checked_mul(rows)
                    .and_then(|e| e.checked_mul(4))
                    .ok_or_else(|| too_large(t))?;
                if let Some(flag) = self.normalized.get_mut(index) {
                    *flag = true;
                }
                let first = self.elements_normalized;
                self.elements_normalized = self
                    .elements_normalized
                    .checked_add(elements)
                    .ok_or_else(|| too_large(t))?;
                (
                    Op::Normalize {
                        dtype,
                        elements,
                        clip,
                        first,
                        rows,
                    },
                    Space::Box {
                        shape,
                        dtype: DType::F32,
                        bounds: BoxBounds::Float {
                            low: vec![f64::from(-clip)],
                            high: vec![f64::from(clip)],
                        },
                    },
                )
            }
        };
        Ok(pair)
    }
}

impl Transforms {
    /// The exported observation space and `S`, the frame-stack bytes per
    /// instance, of these transforms on observation `space` with
    /// `max_agents` agent slots, without building a pool (for memory
    /// bounds such as `SearchGroup::bound_bytes`). An empty list exports
    /// `space` with `S = 0`.
    ///
    /// # Errors
    ///
    /// As `EnvPool::with_transforms` for a list the space does not allow.
    pub fn exported(
        &self,
        space: &Space,
        max_agents: usize,
    ) -> Result<(Space, usize), TransformError> {
        if self.is_empty() {
            return Ok((space.clone(), 0));
        }
        let layout = crate::layout::obs_leaves(space).map_err(|_| TransformError::Unsupported {
            transform: "Transforms",
            leaf: String::new(),
            reason: "the observation space has no valid leaf layout",
        })?;
        let plan = Plan::new(self, space, &layout)?;
        let state = plan
            .agent_state
            .checked_mul(max_agents)
            .ok_or(TransformError::RowTooLarge { bytes: usize::MAX })?;
        Ok((plan.space, state))
    }
}

fn too_large(t: &Transform) -> TransformError {
    TransformError::Unsupported {
        transform: t.name(),
        leaf: t.leaf().to_owned(),
        reason: "the transformed leaf is too large",
    }
}

/// The space of leaf `index`, counting leaves as the layout does: a set
/// item is its one-slot inner space, a set mask a `bool` box `[MAX]`.
fn leaf_space(space: &Space, index: usize) -> Option<Space> {
    fn walk(space: &Space, index: usize, seen: &mut usize) -> Option<Space> {
        match space {
            Space::Dict(entries) => entries.iter().find_map(|(_, s)| walk(s, index, seen)),
            Space::Tuple(items) => items.iter().find_map(|s| walk(s, index, seen)),
            Space::Set { inner, max } => {
                // Item leaves, then the mask leaf.
                let found = walk(inner, index, seen);
                let here = *seen == index;
                *seen += 1;
                found.or_else(|| here.then(|| bool_box(vec![*max])))
            }
            leaf => {
                let here = *seen == index;
                *seen += 1;
                here.then(|| leaf.clone())
            }
        }
    }
    walk(space, index, &mut 0)
}

/// Replaces each transformed leaf by its final space, in layout order; a
/// stacked set (`stacks`: mask index and depth) becomes a set of
/// `k * MAX` slots whose inner `Dict` ends with `frame_age` (ADR-0092
/// point 5).
fn replace_leaves(
    space: &mut Space,
    finals: &mut [Option<Space>],
    seen: &mut usize,
    stacks: &[(usize, usize)],
) -> Result<(), TransformError> {
    match space {
        Space::Dict(entries) => {
            for (_, s) in entries {
                replace_leaves(s, finals, seen, stacks)?;
            }
        }
        Space::Tuple(items) => {
            for s in items {
                replace_leaves(s, finals, seen, stacks)?;
            }
        }
        Space::Set { inner, max } => {
            replace_leaves(inner, finals, seen, stacks)?;
            if let Some(&(_, k)) = stacks.iter().find(|&&(m, _)| m == *seen) {
                let k32 = u32::try_from(k).map_err(|_| TransformError::Internal)?;
                *max = max
                    .checked_mul(k32)
                    .ok_or(TransformError::RowTooLarge { bytes: usize::MAX })?;
                let age = Space::Box {
                    shape: Vec::new(),
                    dtype: DType::U8,
                    bounds: BoxBounds::UInt {
                        low: vec![0],
                        high: vec![u64::from(k32 - 1)],
                    },
                };
                match &mut **inner {
                    Space::Dict(entries) => {
                        if entries.iter().any(|(key, _)| key == FRAME_AGE) {
                            return Err(TransformError::Unsupported {
                                transform: "FrameStack",
                                leaf: FRAME_AGE.to_owned(),
                                reason: "frame_age collides with an item leaf",
                            });
                        }
                        entries.push((FRAME_AGE.to_owned(), age));
                    }
                    other => {
                        let item = core::mem::replace(other, Space::Dict(Vec::new()));
                        *other =
                            Space::Dict(vec![(ITEM.to_owned(), item), (FRAME_AGE.to_owned(), age)]);
                    }
                }
            }
            *seen += 1;
        }
        leaf => {
            if let Some(done) = finals.get_mut(*seen).and_then(Option::take) {
                *leaf = done;
            }
            *seen += 1;
        }
    }
    Ok(())
}

/// A `bool` box of `shape`: the one-hot output, in the form every adapter
/// exposes.
fn bool_box(shape: Vec<u32>) -> Space {
    Space::Box {
        shape,
        dtype: DType::Bool,
        bounds: BoxBounds::Int {
            low: vec![0],
            high: vec![1],
        },
    }
}

/// The element count of a leaf space.
fn elements(space: &Space) -> Option<usize> {
    match space {
        Space::Discrete { .. } => Some(1),
        Space::MultiDiscrete { nvec } => Some(nvec.len()),
        Space::MultiBinary { shape } | Space::Box { shape, .. } => shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(usize::try_from(d).ok()?)),
        _ => None,
    }
}

/// The bytes of one agent's leaf.
fn space_bytes(space: &Space) -> Option<usize> {
    let size = match space {
        Space::Discrete { .. } | Space::MultiDiscrete { .. } => DType::I32.size(),
        Space::MultiBinary { .. } => DType::Bool.size(),
        Space::Box { dtype, .. } => dtype.size(),
        _ => return None,
    };
    elements(space)?.checked_mul(size)
}

/// A leaf's shape as an array shape (`Discrete` is `[1]`).
fn flat_shape(space: &Space) -> Option<Vec<u32>> {
    match space {
        Space::Discrete { .. } => Some(vec![1]),
        Space::MultiDiscrete { nvec } => Some(vec![u32::try_from(nvec.len()).ok()?]),
        Space::MultiBinary { shape } | Space::Box { shape, .. } => Some(shape.clone()),
        _ => None,
    }
}

/// The stored integer type of an integer leaf.
fn integer_dtype(space: &Space) -> Option<DType> {
    match space {
        Space::Discrete { .. } | Space::MultiDiscrete { .. } => Some(DType::I32),
        Space::Box { dtype, .. } if !dtype.is_float() && *dtype != DType::Bool => Some(*dtype),
        _ => None,
    }
}

/// The stored type of a numeric leaf (integer or float, not `bool`).
fn numeric_dtype(space: &Space) -> Option<DType> {
    match space {
        Space::Box { dtype, .. } if *dtype != DType::Bool => Some(*dtype),
        other => integer_dtype(other),
    }
}

/// The space of `k` stacked frames of a leaf.
fn stack_space(space: &Space, k: u32) -> Option<Space> {
    let repeat = |len: usize| usize::try_from(k).ok().and_then(|k| k.checked_mul(len));
    Some(match space {
        Space::Discrete { n } => Space::MultiDiscrete {
            nvec: vec![*n; usize::try_from(k).ok()?],
        },
        Space::MultiDiscrete { nvec } => {
            let mut all = Vec::with_capacity(repeat(nvec.len())?);
            for _ in 0..k {
                all.extend_from_slice(nvec);
            }
            Space::MultiDiscrete { nvec: all }
        }
        Space::MultiBinary { shape } => Space::MultiBinary {
            shape: [k].iter().chain(shape).copied().collect(),
        },
        Space::Box {
            shape,
            dtype,
            bounds,
        } => Space::Box {
            shape: [k].iter().chain(shape).copied().collect(),
            dtype: *dtype,
            bounds: stack_bounds(bounds, k)?,
        },
        _ => return None,
    })
}

/// Bounds of `k` stacked frames: one value stays one value, per-element
/// bounds repeat `k` times.
fn stack_bounds(bounds: &BoxBounds, k: u32) -> Option<BoxBounds> {
    fn rep<T: Copy>(v: &[T], k: u32) -> Option<Vec<T>> {
        if v.len() == 1 {
            return Some(v.to_vec());
        }
        let mut out = Vec::with_capacity(v.len().checked_mul(usize::try_from(k).ok()?)?);
        for _ in 0..k {
            out.extend_from_slice(v);
        }
        Some(out)
    }
    Some(match bounds {
        BoxBounds::Int { low, high } => BoxBounds::Int {
            low: rep(low, k)?,
            high: rep(high, k)?,
        },
        BoxBounds::UInt { low, high } => BoxBounds::UInt {
            low: rep(low, k)?,
            high: rep(high, k)?,
        },
        BoxBounds::Float { low, high } => BoxBounds::Float {
            low: rep(low, k)?,
            high: rep(high, k)?,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::{Op, Plan, STACK_HEADER};
    use crate::layout::obs_leaves;
    use crate::transform::{TransformError, Transforms};
    use tessera_rl_core::space::{BoxBounds, DType, Space};

    fn boxed(shape: Vec<u32>, dtype: DType) -> Space {
        let bounds = if dtype.is_float() {
            BoxBounds::Float {
                low: vec![-1.0],
                high: vec![1.0],
            }
        } else {
            BoxBounds::Int {
                low: vec![0],
                high: vec![9],
            }
        };
        Space::Box {
            shape,
            dtype,
            bounds,
        }
    }

    fn space() -> Space {
        Space::Dict(vec![
            ("pos".to_owned(), boxed(vec![2], DType::F32)),
            ("cell".to_owned(), Space::Discrete { n: 5 }),
            (
                "items".to_owned(),
                Space::Set {
                    inner: Box::new(boxed(vec![2], DType::F32)),
                    max: 3,
                },
            ),
        ])
    }

    fn plan(t: &Transforms) -> Result<Plan, TransformError> {
        let s = space();
        let layout = obs_leaves(&s).map_err(|_| TransformError::Internal)?;
        Plan::new(t, &s, &layout)
    }

    fn names(p: &Plan) -> Vec<String> {
        obs_leaves(&p.space).unwrap().names
    }

    /// ADR-0092 point 5: a set is stacked whole, by its path.
    #[test]
    fn test_plan_partial_set_stack_is_refused() {
        assert!(matches!(
            plan(&Transforms::new().frame_stack("items_mask", 2)),
            Err(TransformError::PartialSetStack { set, .. }) if set == "items"
        ));
    }

    #[test]
    fn test_plan_transform_after_set_stack_is_refused() {
        for t in [
            Transforms::new()
                .frame_stack("items", 2)
                .normalize("items", 1.0),
            Transforms::new()
                .frame_stack("items", 2)
                .frame_stack("items", 3),
        ] {
            assert!(
                matches!(plan(&t), Err(TransformError::AfterSetStack { .. })),
                "{t:?}"
            );
        }
    }

    #[test]
    fn test_plan_mask_normalize_and_one_hot_are_unsupported() {
        for t in [
            Transforms::new().normalize("items_mask", 1.0),
            Transforms::new().one_hot("items_mask", 2),
        ] {
            assert!(
                matches!(plan(&t), Err(TransformError::Unsupported { .. })),
                "{t:?}"
            );
        }
    }

    /// ADR-0092 point 5: `k * MAX` slots, `item` and `frame_age` before the
    /// mask, the leaves after the set shifted by one.
    #[test]
    fn test_plan_set_stack_exports_item_frame_age_then_mask() {
        let p = plan(
            &Transforms::new()
                .normalize("items", 3.0)
                .frame_stack("items", 4),
        )
        .unwrap();
        assert_eq!(
            names(&p),
            ["pos", "cell", "items/item", "items/frame_age", "items_mask"]
        );
        let Space::Dict(entries) = &p.space else {
            panic!("{:?}", p.space);
        };
        let Space::Set { inner, max } = &entries[2].1 else {
            panic!("{:?}", entries[2].1);
        };
        assert_eq!(*max, 12);
        let Space::Dict(inner) = &**inner else {
            panic!("{inner:?}");
        };
        assert!(
            matches!(&inner[0].1, Space::Box { dtype: DType::F32, shape, .. } if *shape == [2])
        );
        assert!(matches!(
            &inner[1].1,
            Space::Box {
                dtype: DType::U8,
                ..
            }
        ));
        assert_eq!(p.out_of_raw[..4], [0, 1, 2, 4]);
        assert_eq!(p.ages.len(), 1);
        assert_eq!(
            (p.ages[0].out, p.ages[0].mask, p.ages[0].k, p.ages[0].max),
            (3, 4, 4, 3)
        );
        // Items: normalize then a ring of 3 x 2 f32; mask: a ring of 3 bools.
        assert_eq!(p.agent_state, 2 * STACK_HEADER + 4 * 24 + 4 * 3);
        // Statistics pooled over the slots: 2 elements, not 6.
        assert_eq!(p.normalized, 2);
    }

    /// ADR-0092 point 2: items with ops bring their mask into the raw set.
    #[test]
    fn test_plan_set_item_op_copies_its_mask() {
        let p = plan(&Transforms::new().normalize("items", 3.0)).unwrap();
        let leaves: Vec<_> = p
            .leaves
            .iter()
            .map(|l| (l.leaf, l.mask, l.ops.len()))
            .collect();
        assert_eq!(leaves, [(2, Some(3), 1), (3, None, 0)]);
        assert_eq!(names(&p), ["pos", "cell", "items", "items_mask"]);
    }

    /// ADR-0092 point 5: a `Dict` inner keeps its keys and appends
    /// `frame_age`.
    #[test]
    fn test_plan_set_stack_dict_inner_keeps_keys() {
        let s = Space::Dict(vec![(
            "units".to_owned(),
            Space::Set {
                inner: Box::new(Space::Dict(vec![
                    ("pos".to_owned(), boxed(vec![2], DType::I32)),
                    ("hp".to_owned(), boxed(vec![1], DType::F32)),
                ])),
                max: 4,
            },
        )]);
        let layout = obs_leaves(&s).unwrap();
        let p = Plan::new(&Transforms::new().frame_stack("units", 2), &s, &layout).unwrap();
        assert_eq!(
            names(&p),
            ["units/pos", "units/hp", "units/frame_age", "units_mask"]
        );
    }

    #[test]
    fn test_plan_frame_age_collision_is_refused() {
        let s = Space::Set {
            inner: Box::new(Space::Dict(vec![(
                "frame_age".to_owned(),
                boxed(vec![1], DType::F32),
            )])),
            max: 2,
        };
        let layout = obs_leaves(&s).unwrap();
        assert!(matches!(
            Plan::new(&Transforms::new().frame_stack("", 2), &s, &layout),
            Err(TransformError::Unsupported { .. })
        ));
    }

    /// A stack exactly at the 16 MiB row limit passes; one slot more is
    /// refused.
    #[test]
    fn test_plan_set_stack_at_row_limit() {
        let set = |max: u32| Space::Set {
            inner: Box::new(boxed(vec![], DType::U8)),
            max,
        };
        // Items k*max bytes, age k*max, mask k*max: 3 * 2 * max.
        let limit = u32::try_from(super::MAX_TRANSFORMED_ROW_BYTES / 6).unwrap();
        let ok = set(limit);
        let layout = obs_leaves(&ok).unwrap();
        assert!(Plan::new(&Transforms::new().frame_stack("", 2), &ok, &layout).is_ok());
        let over = set(limit + 1);
        let layout = obs_leaves(&over).unwrap();
        assert!(matches!(
            Plan::new(&Transforms::new().frame_stack("", 2), &over, &layout),
            Err(TransformError::RowTooLarge { .. })
        ));
    }

    #[test]
    fn test_plan_exported_space_and_sizes() {
        let p = plan(&Transforms::new().frame_stack("pos", 4).one_hot("cell", 5)).unwrap();
        let Space::Dict(entries) = &p.space else {
            panic!("{:?}", p.space);
        };
        assert!(matches!(&entries[0].1, Space::Box { shape, .. } if *shape == [4, 2]));
        assert!(
            matches!(&entries[1].1, Space::Box { shape, dtype: DType::Bool, .. } if *shape == [1, 5])
        );
        // The set is untouched.
        assert_eq!(entries[2].1, space_items());
        assert_eq!(p.out_of_raw[..4], [0, 1, 2, 3]);
        assert_eq!(p.agent_state, STACK_HEADER + 4 * 8);
        assert_eq!(p.leaves.len(), 2);
        assert_eq!(
            p.leaves[0].ops,
            [Op::Stack {
                frame: 8,
                k: 4,
                offset: 0
            }]
        );
    }

    fn space_items() -> Space {
        match space() {
            Space::Dict(entries) => entries[2].1.clone(),
            other => other,
        }
    }

    #[test]
    fn test_plan_normalize_offsets_follow_list_order() {
        let p = plan(
            &Transforms::new()
                .normalize("cell", 3.0)
                .normalize("pos", 3.0),
        )
        .unwrap();
        assert_eq!(p.normalized, 3);
        let firsts: Vec<_> = p
            .leaves
            .iter()
            .map(|l| match l.ops[0] {
                Op::Normalize { first, .. } => first,
                _ => usize::MAX,
            })
            .collect();
        // Leaves are listed in leaf order; offsets in list order.
        assert_eq!(firsts, [1, 0]);
    }

    #[test]
    fn test_plan_one_hot_refuses_float_leaves() {
        assert!(matches!(
            plan(&Transforms::new().one_hot("pos", 3)),
            Err(TransformError::Unsupported { .. })
        ));
    }
}
