// SPDX-License-Identifier: MIT OR Apache-2.0

//! The policy map in replay files (ADR-0079 item 6, #347 stage B2): the
//! recorder writes the pool's map into the header, and replay verifies that
//! the caller-built pool has the same map before any event (erratum B2 to
//! item 6.1: the caller builds the pool, replay checks it).

use tessera_rl_core::PolicyGroup;

use super::ReplayError;
use super::reader::{Replay, ReplayHeader};
use crate::error::EngineError;
use crate::persist::PersistError;
use crate::policy::PolicyMap;

impl ReplayHeader<'_> {
    /// The recorded policy map (ADR-0079 item 6): `None` for a slot count
    /// of 0, the default map (every slot a learner; the caller builds it
    /// with [`PolicyMap::new`] and the environment's `MAX_AGENTS`), else
    /// `Some` with one group per agent slot, in slot order.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Unsupported`] for a group code this engine has no
    /// [`PolicyGroup`] for (4, opponent, ADR-0088); codes past
    /// [`super::MAX_POLICY_GROUP`] are refused when the file is opened.
    pub fn policy_map(&self) -> Result<Option<PolicyMap>, ReplayError> {
        if self.policy_map.is_empty() {
            return Ok(None);
        }
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(self.policy_map.len())
            .map_err(|_| ReplayError::Allocation)?;
        for &code in self.policy_map {
            let group = PolicyGroup::from_code(code).ok_or(ReplayError::Unsupported {
                feature: "opponent policy groups (ADR-0088)",
            })?;
            groups.push(group);
        }
        Ok(Some(PolicyMap::from_groups(groups)))
    }

    /// The digest of the recorded map ([`PolicyMap::digest`]): 0 for the
    /// default map, the value keyframe pool contexts carry.
    ///
    /// # Errors
    ///
    /// As [`ReplayHeader::policy_map`].
    pub fn policy_map_digest(&self) -> Result<u64, ReplayError> {
        Ok(self.policy_map()?.as_ref().map_or(0, PolicyMap::digest))
    }
}

impl Replay<'_> {
    /// Refuses a pool whose map is not the header's, before anything is
    /// mutated: [`EngineError::PolicyMapMismatch`] with the header's digest
    /// as `found` and the pool's as `expected` (as `check_context` reports a
    /// context), wrapped as [`ReplayError::Engine`].
    pub(super) fn check_policy_map(&self, pool: &PolicyMap) -> Result<(), ReplayError> {
        let same = match self.header.policy_map()? {
            None => pool.is_default(),
            Some(recorded) => recorded == *pool || (recorded.is_default() && pool.is_default()),
        };
        if same {
            return Ok(());
        }
        Err(EngineError::PolicyMapMismatch {
            expected: pool.digest(),
            found: self.header.policy_map_digest()?,
        }
        .into())
    }
}

/// The header field's codes for `map` (ADR-0079 item 6.2): empty for the
/// default map (count 0), else one code per slot; the map must have
/// `agents` slots.
pub(super) fn header_codes(map: &PolicyMap, agents: usize) -> Result<Vec<u8>, ReplayError> {
    if map.is_default() {
        return Ok(Vec::new());
    }
    if map.groups().len() != agents || u16::try_from(agents).is_err() {
        return Err(PersistError::InvalidField {
            field: "policy map",
        }
        .into());
    }
    let mut codes = Vec::new();
    codes
        .try_reserve_exact(agents)
        .map_err(|_| ReplayError::Allocation)?;
    codes.extend(map.groups().iter().map(|g| g.code()));
    Ok(codes)
}
