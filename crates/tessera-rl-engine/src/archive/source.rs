// SPDX-License-Identifier: MIT OR Apache-2.0

//! An attached reset source (ADR-0075 items 2 and 4): the archive, its
//! weights and the `u64` prefix table the selection searches.

use core::num::NonZeroU64;

use tessera_rl_core::Environment;
use tessera_rl_num::identity::{RngIdentity, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

use super::{Archive, ResetSourceContext};
use crate::error::EngineError;

/// What a pool's resets start from (ADR-0075 item 2).
pub enum ResetSource<E: Environment> {
    /// `Environment::reset` (the default).
    Fresh,
    /// Archive resets. `weights` holds one `u32` per entry; `None` weighs
    /// every entry 1.
    Archive {
        /// The archive, shared, never copied.
        archive: Archive<E>,
        /// One weight per entry, or `None` for all ones.
        weights: Option<Vec<u32>>,
    },
}

impl<E: Environment> core::fmt::Debug for ArchiveSource<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ArchiveSource")
            .field("archive", &self.archive)
            .field("total", &self.total)
            .finish_non_exhaustive()
    }
}

/// An attached archive source: built at attach, read by the kernels.
pub(crate) struct ArchiveSource<E: Environment> {
    archive: Archive<E>,
    weights: Vec<u32>,
    /// `prefix[i]` is the sum of the weights of entries `0..=i`.
    prefix: Vec<u64>,
    total: NonZeroU64,
}

impl<E: Environment> ArchiveSource<E> {
    /// Checks and builds the weight table (item 4).
    ///
    /// # Errors
    ///
    /// [`EngineError::Archive`] for a weight count that differs from the
    /// entry count, or all-zero weights.
    pub(crate) fn new(archive: Archive<E>, weights: Option<Vec<u32>>) -> Result<Self, EngineError> {
        let weights = weights.unwrap_or_else(|| vec![1; archive.len()]);
        if weights.len() != archive.len() {
            return Err(EngineError::Archive {
                reason: "one weight per archive entry is needed",
            });
        }
        let mut prefix = Vec::with_capacity(weights.len());
        let mut sum: u64 = 0;
        for &w in &weights {
            // At most 2^24 entries of at most 2^32 - 1: below 2^56.
            sum = sum.checked_add(u64::from(w)).ok_or(EngineError::Archive {
                reason: "the archive weights overflow",
            })?;
            prefix.push(sum);
        }
        let total = NonZeroU64::new(sum).ok_or(EngineError::Archive {
            reason: "the archive weights are all zero",
        })?;
        Ok(Self {
            archive,
            weights,
            prefix,
            total,
        })
    }

    /// The archive.
    pub(crate) fn archive(&self) -> &Archive<E> {
        &self.archive
    }

    /// The pool-context form of this source.
    pub(crate) fn context(&self) -> ResetSourceContext {
        ResetSourceContext::Archive {
            digest: self.archive.digest(),
            weights: self.weights.clone(),
        }
    }

    /// The entry of the reset whose (already advanced) identity is
    /// `identity`: a uniform `u64` below the weight total from the
    /// `archive_select` stream, then the first entry whose prefix sum
    /// exceeds it. Integers only; weight-0 entries are never chosen.
    #[must_use]
    pub(crate) fn select(&self, authority: &RngAuthority, identity: &RngIdentity) -> usize {
        let mut rng = StepRng::new(authority, identity, StreamId::ARCHIVE_SELECT);
        pick(&self.prefix, rng.below_u64(self.total))
    }
}

/// The first entry whose prefix sum exceeds `r` (`r` below the total):
/// entry `i` is chosen for `r` in `[prefix[i - 1], prefix[i])`, so a
/// weight-0 entry, whose range is empty, never is.
#[must_use]
fn pick(prefix: &[u64], r: u64) -> usize {
    prefix.partition_point(|&p| p <= r)
}

#[cfg(test)]
mod tests {
    use super::pick;

    #[test]
    fn test_pick_maps_each_draw_to_its_weight_range() {
        // Weights (1, 0, 3, 4): prefix (1, 1, 4, 8).
        let prefix = [1, 1, 4, 8];
        let picks: Vec<usize> = (0..8).map(|r| pick(&prefix, r)).collect();
        assert_eq!(picks, [0, 2, 2, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn test_pick_single_entry_always_zero() {
        assert_eq!(pick(&[5], 0), 0);
        assert_eq!(pick(&[5], 4), 0);
    }
}
