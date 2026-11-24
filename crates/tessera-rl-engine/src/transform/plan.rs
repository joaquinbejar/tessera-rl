// SPDX-License-Identifier: MIT OR Apache-2.0

//! A transform list checked against an observation space (ADR-0078 items 2
//! and 3): the exported space, and per leaf the operations, their byte
//! sizes and their offsets into the frame-stack arena and the statistics.

use tessera_rl_core::space::{BoxBounds, DType, Space};

use super::error::TransformError;
use super::spec::{FRAME_STACK_DEPTH, MAX_TRANSFORMED_ROW_BYTES, Transform, Transforms};
use crate::layout::ObsLayout;

/// Bytes of a frame stack's header per agent: the ring position and the
/// generation of the agent the frames belong to (both `u32`).
pub(crate) const STACK_HEADER: usize = 8;

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
    /// `elements` integers of `dtype` to `elements * n` booleans.
    OneHot {
        dtype: DType,
        elements: usize,
        n: usize,
    },
    /// `elements` values of `dtype` to `f32`, with the statistics from
    /// element `first` of the pool's normalized elements.
    Normalize {
        dtype: DType,
        elements: usize,
        clip: f32,
        first: usize,
    },
}

impl Op {
    /// Bytes this operation writes for one agent.
    pub(crate) fn out_bytes(&self) -> usize {
        match *self {
            Self::Stack { frame, k, .. } => frame * k,
            Self::OneHot { elements, n, .. } => elements * n,
            Self::Normalize { elements, .. } => elements * 4,
        }
    }
}

/// The operations of one transformed leaf.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LeafPlan {
    /// The leaf's index (the same in the raw and the exported layout).
    pub(crate) leaf: usize,
    /// Bytes `E::observe` writes.
    pub(crate) raw: usize,
    pub(crate) ops: Vec<Op>,
}

/// A transform list made concrete for one observation space.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Plan {
    /// The exported observation space.
    pub(crate) space: Space,
    /// The transformed leaves, in leaf order.
    pub(crate) leaves: Vec<LeafPlan>,
    /// Frame-stack bytes per agent (`0` without a `FrameStack`).
    pub(crate) agent_state: usize,
    /// Normalized elements per agent row (`0` without a `Normalize`).
    pub(crate) normalized: usize,
    /// The largest intermediate row of any leaf, for the scratch buffers.
    pub(crate) widest: usize,
    /// Raw bytes of all transformed leaves of one agent.
    pub(crate) raw_bytes: usize,
}

impl Plan {
    /// Checks `transforms` against `space` and its leaf layout.
    pub(crate) fn new(
        transforms: &Transforms,
        space: &Space,
        layout: &ObsLayout,
    ) -> Result<Self, TransformError> {
        // The current space of each leaf, as transforms change it.
        let mut current: Vec<Option<Space>> = vec![None; layout.names.len()];
        let mut ops: Vec<Vec<Op>> = vec![Vec::new(); layout.names.len()];
        let mut builder = Builder {
            agent_state: 0,
            elements_normalized: 0,
            normalized: vec![false; layout.names.len()],
            stacked: vec![false; layout.names.len()],
        };
        for t in transforms.as_slice() {
            let index = layout
                .names
                .iter()
                .position(|n| n == t.leaf())
                .ok_or_else(|| TransformError::UnknownLeaf {
                    leaf: t.leaf().to_owned(),
                })?;
            if layout
                .sets
                .as_slice()
                .iter()
                .any(|s| (s.first..=s.mask).contains(&index))
            {
                return Err(TransformError::OnSetLeaf {
                    leaf: t.leaf().to_owned(),
                });
            }
            let slot = current.get_mut(index).ok_or(TransformError::Internal)?;
            let input = match slot.take() {
                Some(s) => s,
                None => leaf_space(space, index)
                    .ok_or(TransformError::Internal)?
                    .clone(),
            };
            let (op, output) = builder.op(t, &input, index)?;
            if op.out_bytes() > MAX_TRANSFORMED_ROW_BYTES {
                return Err(too_large(t));
            }
            *slot = Some(output);
            if let Some(list) = ops.get_mut(index) {
                list.push(op);
            }
        }
        let mut leaves = Vec::new();
        let mut widest = 0usize;
        let mut raw_bytes = 0usize;
        let mut row = 0usize;
        for (index, (list, leaf)) in ops.into_iter().zip(&layout.leaves).enumerate() {
            let out = list.last().map_or(leaf.bytes, Op::out_bytes);
            row = row
                .checked_add(out)
                .ok_or(TransformError::RowTooLarge { bytes: usize::MAX })?;
            if list.is_empty() {
                continue;
            }
            widest = list
                .iter()
                .map(Op::out_bytes)
                .fold(widest.max(leaf.bytes), usize::max);
            raw_bytes = raw_bytes
                .checked_add(leaf.bytes)
                .ok_or(TransformError::RowTooLarge { bytes: usize::MAX })?;
            leaves.push(LeafPlan {
                leaf: index,
                raw: leaf.bytes,
                ops: list,
            });
        }
        if row > MAX_TRANSFORMED_ROW_BYTES {
            return Err(TransformError::RowTooLarge { bytes: row });
        }
        let mut exported = space.clone();
        let mut index = 0usize;
        replace_leaves(&mut exported, &mut current, &mut index);
        Ok(Self {
            space: exported,
            leaves,
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

/// The running totals of a plan under construction.
struct Builder {
    agent_state: usize,
    elements_normalized: usize,
    normalized: Vec<bool>,
    stacked: Vec<bool>,
}

impl Builder {
    /// The operation of `t` on leaf `index`, whose current space is
    /// `input`, and the leaf's space after it.
    fn op(
        &mut self,
        t: &Transform,
        input: &Space,
        index: usize,
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
                if let Some(flag) = self.stacked.get_mut(index) {
                    *flag = true;
                }
                (
                    Op::Stack {
                        frame: bytes,
                        k: k_usize,
                        offset,
                    },
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
                elements.checked_mul(n_usize).ok_or_else(|| too_large(t))?;
                (
                    Op::OneHot {
                        dtype,
                        elements,
                        n: n_usize,
                    },
                    Space::MultiBinary { shape },
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

/// The space of leaf `index`, counting leaves as the layout does (sets are
/// walked for counting only; transforms on them are refused earlier).
fn leaf_space(space: &Space, index: usize) -> Option<&Space> {
    fn walk<'a>(space: &'a Space, index: usize, seen: &mut usize) -> Option<&'a Space> {
        match space {
            Space::Dict(entries) => entries.iter().find_map(|(_, s)| walk(s, index, seen)),
            Space::Tuple(items) => items.iter().find_map(|s| walk(s, index, seen)),
            Space::Set { inner, .. } => {
                // Item leaves, then the mask leaf.
                let found = walk(inner, index, seen);
                *seen += 1;
                found
            }
            leaf => {
                let here = *seen == index;
                *seen += 1;
                here.then_some(leaf)
            }
        }
    }
    walk(space, index, &mut 0)
}

/// Replaces each transformed leaf by its final space, in layout order.
fn replace_leaves(space: &mut Space, finals: &mut [Option<Space>], seen: &mut usize) {
    match space {
        Space::Dict(entries) => {
            for (_, s) in entries {
                replace_leaves(s, finals, seen);
            }
        }
        Space::Tuple(items) => {
            for s in items {
                replace_leaves(s, finals, seen);
            }
        }
        Space::Set { inner, .. } => {
            replace_leaves(inner, finals, seen);
            *seen += 1;
        }
        leaf => {
            if let Some(done) = finals.get_mut(*seen).and_then(Option::take) {
                *leaf = done;
            }
            *seen += 1;
        }
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

    #[test]
    fn test_plan_refuses_every_transform_on_set_leaves() {
        for t in [
            Transforms::new().frame_stack("items", 2),
            Transforms::new().normalize("items", 1.0),
            Transforms::new().one_hot("items_mask", 2),
        ] {
            assert!(
                matches!(plan(&t), Err(TransformError::OnSetLeaf { .. })),
                "{t:?}"
            );
        }
    }

    #[test]
    fn test_plan_exported_space_and_sizes() {
        let p = plan(&Transforms::new().frame_stack("pos", 4).one_hot("cell", 5)).unwrap();
        let Space::Dict(entries) = &p.space else {
            panic!("{:?}", p.space);
        };
        assert!(matches!(&entries[0].1, Space::Box { shape, .. } if *shape == [4, 2]));
        assert_eq!(entries[1].1, Space::MultiBinary { shape: vec![1, 5] });
        // The set is untouched.
        assert_eq!(entries[2].1, space_items());
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
