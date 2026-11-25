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
    /// [`PolicyGroup`] for; codes past [`super::MAX_POLICY_GROUP`] are
    /// refused when the file is opened. Code 4 decodes to
    /// [`PolicyGroup::Opponent`] (ADR-0088), but replaying a map with an
    /// opponent slot is refused (league replay, #384 head D).
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
                feature: "an unknown policy group",
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
        // An `Opponent` slot (code 4) needs a league file (ADR-0088 section
        // 8): the sync replay checks the league (`check_league`).
        let league_map = |m: &PolicyMap| m.groups().contains(&PolicyGroup::Opponent);
        if self.header.policy_map()?.as_ref().is_some_and(league_map) != self.header.has_league {
            return Err(ReplayError::Malformed {
                event_index: 0,
                field: "has_league without an opponent slot, or one without has_league",
            });
        }
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

/// Refuses a map with an `Opponent` slot where no league can be recorded or
/// replayed: async pools (ADR-0088, #384 head B refuses a league there).
pub(super) fn refuse_league(map: &PolicyMap) -> Result<(), ReplayError> {
    if map.groups().contains(&PolicyGroup::Opponent) {
        return Err(ReplayError::Unsupported {
            feature: "leagues in async pools (ADR-0088)",
        });
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use tessera_rl_core::PolicyGroup;

    use super::{header_codes, refuse_league};
    use crate::policy::PolicyMap;

    /// Code 4 round-trips (ADR-0088 section 8): `Opponent` encodes to 4 and
    /// 4 decodes back, for every group.
    #[test]
    fn test_policy_map_codes_round_trip_every_group_including_opponent() {
        let map = PolicyMap::from_groups(PolicyGroup::ALL.to_vec());
        let codes = header_codes(&map, PolicyGroup::ALL.len()).unwrap();
        assert_eq!(codes, [0, 1, 2, 3, 4]);
        let back: Vec<_> = codes
            .iter()
            .map(|&c| PolicyGroup::from_code(c).unwrap())
            .collect();
        assert_eq!(back, PolicyGroup::ALL);
    }

    /// A map with an opponent slot is refused where no league is recorded
    /// (async pools); sync league files carry it (#384 head D).
    #[test]
    fn test_policy_map_with_an_opponent_is_refused_without_a_league() {
        let league = PolicyMap::from_groups(vec![PolicyGroup::Learner, PolicyGroup::Opponent]);
        assert!(matches!(
            refuse_league(&league),
            Err(super::super::ReplayError::Unsupported { feature }) if feature.contains("async")
        ));
        let heuristic = PolicyMap::from_groups(vec![PolicyGroup::Learner, PolicyGroup::Heuristic]);
        assert!(refuse_league(&heuristic).is_ok());
    }
}
