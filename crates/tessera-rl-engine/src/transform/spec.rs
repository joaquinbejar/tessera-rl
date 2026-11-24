// SPDX-License-Identifier: MIT OR Apache-2.0

//! The transform list a pool is built with (ADR-0078 items 2 and 3).

use tessera_rl_core::encoding::{Encoder, HashSink};

/// Least and most frames a [`Transform::FrameStack`] keeps (ADR-0078 item 3).
pub const FRAME_STACK_DEPTH: core::ops::RangeInclusive<u32> = 2..=64;

/// Most bytes of one agent's transformed observation (ADR-0078 item 3).
pub const MAX_TRANSFORMED_ROW_BYTES: usize = 16 << 20;

/// Most bytes of the normalization chunk accumulators (ADR-0078 item 5.2).
pub const MAX_NORMALIZE_ACCUMULATOR_BYTES: usize = 256 << 20;

/// One observation transform, applied to the leaf it names.
///
/// A leaf is named as the pool exports it: `Dict` keys and `Tuple`
/// positions joined by `.`, the empty name for a space that is one leaf.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Transform {
    /// The last `k` observations of the leaf, oldest first: shape
    /// `[k, *shape]`, the leaf's dtype.
    ///
    /// Naming a `Set` path `p` stacks every leaf of the set (ADR-0092): it
    /// is exported as a set of `k * MAX` slots, frames oldest first, whose
    /// leaves are the items, then `p/frame_age` (`u8`, 0 the newest frame,
    /// 0 for absent slots), then `p_mask`. Stacking a single-leaf set
    /// renames its item leaf from `p` to `p/item`; a `Dict` inner keeps its
    /// keys.
    FrameStack {
        /// The leaf.
        leaf: String,
        /// Frames, in [`FRAME_STACK_DEPTH`].
        k: u32,
    },
    /// An integer leaf as one-hot rows: shape `[*shape, n]`, `bool`; a value
    /// outside `0..n` gives an all-false row.
    OneHot {
        /// The leaf.
        leaf: String,
        /// Classes, at least 1.
        n: u32,
    },
    /// A float or integer leaf, normalized by the pool's running
    /// statistics: `clamp((x - mean) / sqrt(var + 1e-8), -clip, clip)`, `f32`.
    Normalize {
        /// The leaf.
        leaf: String,
        /// The clip, finite and positive.
        clip: f32,
    },
}

impl Transform {
    /// The leaf this transform reads.
    #[must_use]
    pub fn leaf(&self) -> &str {
        match self {
            Self::FrameStack { leaf, .. }
            | Self::OneHot { leaf, .. }
            | Self::Normalize { leaf, .. } => leaf,
        }
    }

    /// The transform's name, for errors.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::FrameStack { .. } => "FrameStack",
            Self::OneHot { .. } => "OneHot",
            Self::Normalize { .. } => "Normalize",
        }
    }
}

/// An ordered list of transforms (ADR-0078 item 2.1): applied in list order
/// per leaf, after `E::observe`, on every path that observes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transforms {
    list: Vec<Transform>,
}

impl Transforms {
    /// No transforms: the pool exports the environment's observation.
    #[must_use]
    pub const fn new() -> Self {
        Self { list: Vec::new() }
    }

    /// Appends a [`Transform::FrameStack`].
    #[must_use]
    pub fn frame_stack(mut self, leaf: impl Into<String>, k: u32) -> Self {
        self.list.push(Transform::FrameStack {
            leaf: leaf.into(),
            k,
        });
        self
    }

    /// Appends a [`Transform::OneHot`].
    #[must_use]
    pub fn one_hot(mut self, leaf: impl Into<String>, n: u32) -> Self {
        self.list.push(Transform::OneHot {
            leaf: leaf.into(),
            n,
        });
        self
    }

    /// Appends a [`Transform::Normalize`].
    #[must_use]
    pub fn normalize(mut self, leaf: impl Into<String>, clip: f32) -> Self {
        self.list.push(Transform::Normalize {
            leaf: leaf.into(),
            clip,
        });
        self
    }

    /// Appends any transform.
    #[must_use]
    pub fn push(mut self, transform: Transform) -> Self {
        self.list.push(transform);
        self
    }

    /// Whether the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// The transforms, in order.
    #[must_use]
    pub fn as_slice(&self) -> &[Transform] {
        &self.list
    }

    /// Whether any transform is a [`Transform::Normalize`].
    #[must_use]
    pub fn has_normalize(&self) -> bool {
        self.list
            .iter()
            .any(|t| matches!(t, Transform::Normalize { .. }))
    }

    /// The transform identity (ADR-0078 item 2.3): a digest of the list and
    /// its parameters, plus `chunk_size` when a `Normalize` is present.
    /// `0` for an empty list.
    #[must_use]
    pub fn identity(&self, chunk_size: u32) -> u64 {
        if self.list.is_empty() {
            return 0;
        }
        let mut enc = Encoder::new(HashSink::new());
        enc.write_bytes(b"tessera.transforms.v1");
        enc.write_bytes(
            &u64::try_from(self.list.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for t in &self.list {
            let (tag, param): (u8, [u8; 4]) = match t {
                Transform::FrameStack { k, .. } => (1, k.to_le_bytes()),
                Transform::OneHot { n, .. } => (2, n.to_le_bytes()),
                Transform::Normalize { clip, .. } => (3, clip.to_bits().to_le_bytes()),
            };
            enc.write_bytes(&[tag]);
            enc.write_bytes(&param);
            let leaf = t.leaf().as_bytes();
            enc.write_bytes(&u64::try_from(leaf.len()).unwrap_or(u64::MAX).to_le_bytes());
            enc.write_bytes(leaf);
        }
        if self.has_normalize() {
            enc.write_bytes(&chunk_size.to_le_bytes());
        }
        enc.sink().digest()
    }
}
