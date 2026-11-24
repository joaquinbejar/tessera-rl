// SPDX-License-Identifier: MIT OR Apache-2.0

//! Entity-centric action heads (SPEC 11.3, #385): the same heads repeated
//! for every entity an agent controls, each head optionally the child of
//! an earlier head of the same entity.
//!
//! An environment with entity-centric actions (the RTS: per unit an action
//! type, a target in a window and a product) declares its per-entity heads
//! once as an [`EntityHeads`] and derives from it the action space, the
//! parents ([`crate::ActionMask::head_parents`] or
//! [`crate::HostEnvironment::head_parents`]), the entity groups (ADR-0051)
//! and the indices it reads and writes, instead of repeating that
//! arithmetic. The layout it describes is the engine's (`MaskLayout`):
//! entities one after the other, each entity's heads in order, and a child
//! head's mask with one row per value of its parent.

use core::num::NonZeroU16;

use crate::space::Space;

/// One head of one entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadSpec {
    /// Choices of the head.
    pub choices: u32,
    /// The index, within the same entity, of an earlier head whose value
    /// selects this head's mask row; `None` for a root head.
    pub parent: Option<u16>,
}

impl HeadSpec {
    /// A root head with `choices` choices.
    #[must_use]
    pub const fn root(choices: u32) -> Self {
        Self {
            choices,
            parent: None,
        }
    }

    /// A head with `choices` choices, the child of the entity's head
    /// `parent`.
    #[must_use]
    pub const fn child(choices: u32, parent: u16) -> Self {
        Self {
            choices,
            parent: Some(parent),
        }
    }
}

/// `K` heads per entity, repeated for `entities` entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityHeads<const K: usize> {
    entities: u16,
    heads: [HeadSpec; K],
}

impl<const K: usize> EntityHeads<K> {
    /// The heads `heads` for each of `entities` entities. A head's parent
    /// must be an earlier head of the same entity; [`EntityHeads::check`]
    /// says whether it is.
    #[must_use]
    pub const fn new(entities: u16, heads: [HeadSpec; K]) -> Self {
        Self { entities, heads }
    }

    /// Whether every parent names an earlier head and every head has at
    /// least one choice.
    #[must_use]
    pub fn check(&self) -> bool {
        K > 0
            && self
                .heads
                .iter()
                .enumerate()
                .all(|(k, h)| h.choices > 0 && h.parent.is_none_or(|p| usize::from(p) < k))
    }

    /// Entities.
    #[must_use]
    pub const fn entities(&self) -> u16 {
        self.entities
    }

    /// The action space: `MultiDiscrete`, entity-major.
    #[must_use]
    pub fn space(&self) -> Space {
        let one: Vec<u32> = self.heads.iter().map(|h| h.choices).collect();
        Space::MultiDiscrete {
            nvec: one.repeat(usize::from(self.entities)),
        }
    }

    /// The flat head index of head `k` of `entity`.
    #[must_use]
    pub const fn head(&self, entity: usize, k: usize) -> usize {
        entity * K + k
    }

    /// The parents of flat head `head`, for `head_parents`: the parent's
    /// flat index within the same entity.
    #[must_use]
    pub fn head_parents(&self, head: usize) -> [Option<u16>; 2] {
        let (entity, k) = (head / K.max(1), head % K.max(1));
        let parent = self
            .heads
            .get(k)
            .and_then(|h| h.parent)
            .and_then(|p| u16::try_from(self.head(entity, usize::from(p))).ok());
        [parent, None]
    }

    /// The entity groups, for `entity_group_heads`: `K` heads per group.
    #[must_use]
    pub fn entity_group_heads(&self) -> Option<NonZeroU16> {
        u16::try_from(K).ok().and_then(NonZeroU16::new)
    }

    /// Mask bytes of head `k` of one entity: one row of its choices per
    /// value of its parent; 0 for an index past the heads.
    #[must_use]
    pub fn head_mask_len(&self, k: usize) -> usize {
        let Some(head) = self.heads.get(k) else {
            return 0;
        };
        let rows = head
            .parent
            .and_then(|p| self.heads.get(usize::from(p)))
            .map_or(1, |p| p.choices);
        usize::try_from(u64::from(rows) * u64::from(head.choices)).unwrap_or(usize::MAX)
    }

    /// Mask bytes of one entity: the sum over its heads.
    #[must_use]
    pub fn entity_mask_len(&self) -> usize {
        (0..K).map(|k| self.head_mask_len(k)).sum()
    }

    /// The offset, in an agent's mask row, of the row of head `k` of
    /// `entity` for parent value `parent_value` (0 for a root head):
    /// `None` when `parent_value` is not a choice of the parent.
    #[must_use]
    pub fn mask_row(&self, entity: usize, k: usize, parent_value: u32) -> Option<usize> {
        let head = self.heads.get(k)?;
        let rows = head
            .parent
            .and_then(|p| self.heads.get(usize::from(p)))
            .map_or(1, |p| p.choices);
        if parent_value >= rows {
            return None;
        }
        let before: usize = (0..k).map(|j| self.head_mask_len(j)).sum();
        let row = usize::try_from(parent_value).ok()? * usize::try_from(head.choices).ok()?;
        entity
            .checked_mul(self.entity_mask_len())?
            .checked_add(before)?
            .checked_add(row)
    }
}

#[cfg(test)]
mod tests {
    use super::{EntityHeads, HeadSpec};
    use crate::space::Space;

    const HEADS: [HeadSpec; 3] = [HeadSpec::root(2), HeadSpec::child(3, 0), HeadSpec::root(4)];

    #[test]
    fn test_entity_heads_layout_is_entity_major_with_parent_rows() {
        let h = EntityHeads::new(2, HEADS);
        assert!(h.check());
        assert_eq!(
            h.space(),
            Space::MultiDiscrete {
                nvec: vec![2, 3, 4, 2, 3, 4]
            }
        );
        assert_eq!(h.head(1, 2), 5);
        assert_eq!(h.head_parents(1), [Some(0), None]);
        assert_eq!(h.head_parents(4), [Some(3), None]);
        assert_eq!(h.head_parents(5), [None, None]);
        assert_eq!(
            h.entity_group_heads().map(core::num::NonZeroU16::get),
            Some(3)
        );
        // Per entity: 2 + 2 * 3 + 4 = 12 bytes.
        assert_eq!(h.entity_mask_len(), 12);
        assert_eq!(h.mask_row(0, 1, 1), Some(2 + 3));
        assert_eq!(h.mask_row(1, 2, 0), Some(12 + 8));
        assert_eq!(h.mask_row(0, 1, 2), None);
        assert_eq!(h.mask_row(0, 0, 1), None);
        assert_eq!(h.mask_row(0, 3, 0), None);
    }

    #[test]
    fn test_entity_heads_check_refuses_bad_parents_and_empty_heads() {
        assert!(!EntityHeads::new(1, [HeadSpec::child(2, 0)]).check());
        assert!(!EntityHeads::new(1, [HeadSpec::root(2), HeadSpec::child(2, 1)]).check());
        assert!(!EntityHeads::new(1, [HeadSpec::root(0)]).check());
        assert!(!EntityHeads::<0>::new(1, []).check());
    }
}
