// SPDX-License-Identifier: MIT OR Apache-2.0

//! The joint index over an agent's discrete heads (ADR-0086).
//!
//! Search treats `MultiDiscrete(nvec)` as one head of `J = prod(nvec)`
//! choices. A joint index is mixed radix with the first head most
//! significant: `j = ((h0 x n1 + h1) x n2 + h2) ...`.
//!
//! A conditional head (SPEC 11.3) may be unused under some values of its
//! parents (`ActionMask::head_active`). A joint is **canonical** when every
//! inactive head in it is 0; only canonical joints can be legal, so each
//! game action has exactly one joint.
//!
//! The mask row layout is the one of SPEC 11.3: the heads' blocks end to
//! end, a head with parents `(p1, p2?)` holding `[size(p1), size(p2)?,
//! size]` entries with the last parent varying fastest.

use thiserror::Error;

/// Most discrete heads a joint index covers.
pub const MAX_JOINT_HEADS: usize = 16;

/// Why a joint index cannot be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum JointError {
    /// The product of the head sizes is above the cap.
    #[error("{joint} joint actions, above the maximum of {max}")]
    TooMany {
        /// The product of the head sizes (saturating).
        joint: u64,
        /// The cap.
        max: u32,
    },
    /// The heads or their parents are not a valid declaration.
    #[error("invalid joint heads: {what}")]
    Invalid {
        /// What is wrong.
        what: &'static str,
    },
}

/// One head's place in the joint index and the mask row.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Head {
    size: u32,
    parents: [Option<usize>; 2],
    /// Size of the second parent (1 without one): the stride of the first.
    second: usize,
    /// First entry of this head's block in an agent's mask row.
    offset: usize,
    /// Activity per parent value pair `v1 x second + v2` (one entry
    /// without parents).
    active: Vec<bool>,
}

/// The joint index of one agent's discrete heads, with each head's
/// activity and mask row offsets (ADR-0086). Built at construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JointHeads {
    heads: Vec<Head>,
    joint: u32,
    mask_len: usize,
}

fn usize_of(v: u32) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}

impl JointHeads {
    /// The joint index of heads with choices `sizes` and parents `parents`
    /// (as `ActionMask::head_parents`), at most `max` joints, with
    /// `active(head, [v1, v2])` as `ActionMask::head_active`.
    ///
    /// # Errors
    ///
    /// [`JointError::TooMany`] above `max`; [`JointError::Invalid`] for no
    /// heads, more than [`MAX_JOINT_HEADS`], a head of 0 choices, a
    /// parents list of another length, or a parent that is not an earlier
    /// head, a second parent without a first, or the same parent twice.
    pub fn new(
        sizes: &[u32],
        parents: &[[Option<u16>; 2]],
        max: u32,
        mut active: impl FnMut(usize, [u32; 2]) -> bool,
    ) -> Result<Self, JointError> {
        let bad = |what| Err(JointError::Invalid { what });
        if sizes.is_empty() {
            return bad("no discrete heads");
        }
        if sizes.len() > MAX_JOINT_HEADS {
            return bad("more discrete heads than MAX_JOINT_HEADS");
        }
        if parents.len() != sizes.len() {
            return bad("one parents entry per head is required");
        }
        if sizes.contains(&0) {
            return bad("a head with no choices");
        }
        let joint = sizes
            .iter()
            .try_fold(1u64, |acc, &n| acc.checked_mul(u64::from(n)))
            .unwrap_or(u64::MAX);
        if joint > u64::from(max) {
            return Err(JointError::TooMany { joint, max });
        }
        let joint = u32::try_from(joint).map_err(|_| JointError::TooMany { joint, max })?;
        let mut heads = Vec::with_capacity(sizes.len());
        let mut offset = 0usize;
        for (head, (&size, declared)) in sizes.iter().zip(parents).enumerate() {
            if declared[0].is_none() && declared[1].is_some() {
                return bad("a second parent without a first");
            }
            if declared[0].is_some() && declared[0] == declared[1] {
                return bad("the same parent twice");
            }
            let mut list = [None, None];
            for (slot, p) in declared.iter().enumerate() {
                let Some(p) = p else { continue };
                let p = usize::from(*p);
                if p >= head {
                    return bad("a parent that is not an earlier head");
                }
                if let Some(e) = list.get_mut(slot) {
                    *e = Some(p);
                }
            }
            let size_of =
                |p: Option<usize>| p.and_then(|p| sizes.get(p)).map_or(1, |&n| usize_of(n));
            let (first, second) = (size_of(list[0]), size_of(list[1]));
            // Both parent sizes divide `joint`, so their product fits.
            let pairs = first.saturating_mul(second);
            let mut table = Vec::with_capacity(pairs);
            for pair in 0..pairs {
                let v1 = u32::try_from(pair / second).unwrap_or(0);
                let v2 = u32::try_from(pair % second).unwrap_or(0);
                let v1 = if list[0].is_some() { v1 } else { 0 };
                let v2 = if list[1].is_some() { v2 } else { 0 };
                table.push(active(head, [v1, v2]));
            }
            heads.push(Head {
                size,
                parents: list,
                second,
                offset,
                active: table,
            });
            offset = offset.saturating_add(pairs.saturating_mul(usize_of(size)));
        }
        Ok(Self {
            heads,
            joint,
            mask_len: offset,
        })
    }

    /// The number of joint actions `J`.
    #[must_use]
    pub const fn joint(&self) -> u32 {
        self.joint
    }

    /// The number of discrete heads.
    #[must_use]
    pub fn heads(&self) -> usize {
        self.heads.len()
    }

    /// Choices of head `head`.
    #[must_use]
    pub fn head_size(&self, head: usize) -> Option<u32> {
        self.heads.get(head).map(|h| h.size)
    }

    /// First entry of head `head`'s block in an agent's mask row.
    #[must_use]
    pub fn mask_offset(&self, head: usize) -> Option<usize> {
        self.heads.get(head).map(|h| h.offset)
    }

    /// The entries of one agent's mask row (SPEC 11.3).
    #[must_use]
    pub const fn mask_len(&self) -> usize {
        self.mask_len
    }

    /// Writes the head values of joint `j` into `out` (one `i32` per
    /// head); `false`, writing nothing, if `j` is out of range or `out`
    /// has another length.
    pub fn decode(&self, j: u32, out: &mut [i32]) -> bool {
        if j >= self.joint || out.len() != self.heads.len() {
            return false;
        }
        let mut rest = j;
        for (head, value) in self.heads.iter().zip(out.iter_mut()).rev() {
            *value = i32::try_from(rest % head.size).unwrap_or(0);
            rest /= head.size;
        }
        true
    }

    /// The joint index of head values `values`, or `None` for a value out
    /// of range or another number of values.
    #[must_use]
    pub fn encode(&self, values: &[i32]) -> Option<u32> {
        if values.len() != self.heads.len() {
            return None;
        }
        let mut j = 0u32;
        for (head, &v) in self.heads.iter().zip(values) {
            let v = u32::try_from(v).ok().filter(|&v| v < head.size)?;
            j = j.checked_mul(head.size)?.checked_add(v)?;
        }
        Some(j)
    }

    /// The parent value pair index of `head` under `values`.
    fn pair(head: &Head, values: &[i32]) -> usize {
        let value = |p: Option<usize>| {
            p.and_then(|p| values.get(p))
                .and_then(|&v| usize::try_from(v).ok())
                .unwrap_or(0)
        };
        value(head.parents[0])
            .saturating_mul(head.second)
            .saturating_add(value(head.parents[1]))
    }

    /// Whether head `head` is active under head values `values`.
    #[must_use]
    pub fn is_active(&self, head: usize, values: &[i32]) -> bool {
        self.heads
            .get(head)
            .and_then(|h| h.active.get(Self::pair(h, values)))
            .copied()
            .unwrap_or(false)
    }

    /// Whether joint `j` is canonical: every inactive head is 0.
    #[must_use]
    pub fn is_canonical(&self, j: u32) -> bool {
        let mut buf = [0i32; MAX_JOINT_HEADS];
        let Some(values) = buf.get_mut(..self.heads.len()) else {
            return false;
        };
        if !self.decode(j, values) {
            return false;
        }
        self.canonical_values(values)
    }

    fn canonical_values(&self, values: &[i32]) -> bool {
        self.heads.iter().enumerate().all(|(head, h)| {
            values.get(head).is_some_and(|&v| v == 0)
                || h.active
                    .get(Self::pair(h, values))
                    .copied()
                    .unwrap_or(false)
        })
    }

    /// Whether joint `j` is legal under one agent's mask row, read through
    /// `mask(entry)`: canonical, and every active head's value allowed in
    /// its row under its parents' values (inactive heads are not read).
    #[must_use]
    pub fn is_legal(&self, j: u32, mut mask: impl FnMut(usize) -> bool) -> bool {
        let mut buf = [0i32; MAX_JOINT_HEADS];
        let Some(values) = buf.get_mut(..self.heads.len()) else {
            return false;
        };
        if !self.decode(j, values) {
            return false;
        }
        let values = &*values;
        for (head, h) in self.heads.iter().enumerate() {
            let pair = Self::pair(h, values);
            let v = values.get(head).copied().unwrap_or(0);
            if !h.active.get(pair).copied().unwrap_or(false) {
                if v != 0 {
                    return false;
                }
                continue;
            }
            let at = pair
                .saturating_mul(usize_of(h.size))
                .saturating_add(usize::try_from(v).unwrap_or(usize::MAX))
                .saturating_add(h.offset);
            if !mask(at) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[3, 2, 4]`: head 1 under head 0, head 2 under heads 0 and 1;
    /// head 1 is unused when head 0 is 0, head 2 when head 0 is 2.
    fn sample() -> JointHeads {
        JointHeads::new(
            &[3, 2, 4],
            &[[None, None], [Some(0), None], [Some(0), Some(1)]],
            4096,
            |head, [v1, _]| !matches!((head, v1), (1, 0) | (2, 2)),
        )
        .unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn test_joint_heads_decode_encode_round_trip_first_head_most_significant() {
        let h = sample();
        assert_eq!(h.joint(), 24);
        let mut v = [0i32; 3];
        assert!(h.decode(8 + 4 + 3, &mut v));
        assert_eq!(v, [1, 1, 3]);
        for j in 0..h.joint() {
            assert!(h.decode(j, &mut v));
            assert_eq!(h.encode(&v), Some(j));
        }
        assert!(!h.decode(24, &mut v));
        assert!(!h.decode(0, &mut [0; 2]));
        assert_eq!(h.encode(&[3, 0, 0]), None);
        assert_eq!(h.encode(&[0, 0]), None);
    }

    #[test]
    fn test_joint_heads_mask_offsets_follow_spec_11_3() {
        let h = sample();
        // 3 + 3 x 2 + 3 x 2 x 4.
        assert_eq!(h.mask_len(), 3 + 6 + 24);
        // Joint [1, 1, 3]: head 0 at 1, head 1 at 3 + 1 x 2 + 1, head 2 at
        // 9 + (1 x 2 + 1) x 4 + 3.
        let mut read = Vec::new();
        assert!(h.is_legal(15, |at| {
            read.push(at);
            true
        }));
        assert_eq!(read, [1, 6, 24]);
    }

    #[test]
    fn test_joint_heads_canonical_joints_are_one_per_game_action() {
        let h = sample();
        let canonical: Vec<u32> = (0..h.joint()).filter(|&j| h.is_canonical(j)).collect();
        // head 0 = 0: head 1 off (0), head 2 on: 4. head 0 = 1: 2 x 4 = 8.
        // head 0 = 2: head 1 on, head 2 off: 2.
        assert_eq!(canonical.len(), 4 + 8 + 2);
        let mut v = [0i32; 3];
        for j in 0..h.joint() {
            assert!(h.decode(j, &mut v));
            let off1 = v[0] == 0;
            let off2 = v[0] == 2;
            let want = !(off1 && v[1] != 0 || off2 && v[2] != 0);
            assert_eq!(h.is_canonical(j), want, "{v:?}");
            // Without masks, legal is canonical.
            assert_eq!(h.is_legal(j, |_| true), want, "{v:?}");
        }
    }

    #[test]
    fn test_joint_heads_inactive_heads_are_not_read_from_the_mask() {
        let h = sample();
        // [2, 1, 0]: head 2 inactive; only heads 0 and 1 are read.
        let j = h.encode(&[2, 1, 0]).unwrap_or(u32::MAX);
        let mut read = Vec::new();
        assert!(h.is_legal(j, |at| {
            read.push(at);
            true
        }));
        assert_eq!(read, [2, 3 + 2 * 2 + 1]);
        // A false entry in an active row makes the joint illegal.
        assert!(!h.is_legal(j, |at| at != 8));
    }

    #[test]
    fn test_joint_heads_refuses_bad_declarations() {
        let flat = [[None, None]; 2];
        assert_eq!(
            JointHeads::new(&[64, 65], &flat, 4096, |_, _| true),
            Err(JointError::TooMany {
                joint: 4160,
                max: 4096
            })
        );
        assert!(JointHeads::new(&[64, 64], &flat, 4096, |_, _| true).is_ok());
        let too_many = JointHeads::new(&[u32::MAX, u32::MAX], &flat, u32::MAX, |_, _| true);
        assert!(matches!(too_many, Err(JointError::TooMany { .. })));
        let invalid = |sizes: &[u32], parents: &[[Option<u16>; 2]]| {
            matches!(
                JointHeads::new(sizes, parents, 4096, |_, _| true),
                Err(JointError::Invalid { .. })
            )
        };
        assert!(invalid(&[], &[]));
        assert!(invalid(&[2, 0], &flat));
        assert!(invalid(&[2, 2], &flat[..1]));
        assert!(invalid(&[2, 2], &[[None, None], [Some(1), None]]));
        assert!(invalid(&[2, 2], &[[None, None], [None, Some(0)]]));
        assert!(invalid(
            &[2, 2, 2],
            &[[None, None], [None, None], [Some(0), Some(0)]]
        ));
        assert!(invalid(
            &[1; MAX_JOINT_HEADS + 1],
            &[[None, None]; MAX_JOINT_HEADS + 1]
        ));
    }
}
