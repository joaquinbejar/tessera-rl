// SPDX-License-Identifier: MIT OR Apache-2.0

//! `make(..., policy_map=...)` (ADR-0079 items 2 and 5): the Python layer
//! passes the map as one string, and each environment's maker applies it
//! with the setter its heuristic allows.

use tessera_rl_core::{Heuristic, InfoHeuristic, PolicyGroup};
use tessera_rl_engine::{EnvPool, MaskProvider, PolicyMap};

use crate::env::{BindError, PoolArgs};

/// A policy map as Python gives it, before the slot count is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PolicySpec {
    /// One group per agent slot, in slot order (a list).
    List(Vec<PolicyGroup>),
    /// Groups of some slots; every other slot is a learner (a dict).
    Slots(Vec<(usize, PolicyGroup)>),
}

/// Parses the string `tessera_rl.make` builds: group names separated by
/// commas (a list), or `slot=name` items (a dict).
///
/// # Errors
///
/// The message of an unknown group name or a bad slot.
pub(crate) fn policy_spec(text: &str) -> Result<PolicySpec, String> {
    let group = |name: &str| {
        PolicyGroup::from_name(name.trim()).ok_or_else(|| {
            format!(
                "policy_map groups are \"learner\", \"external\", \"heuristic\", \
                 \"privileged_heuristic\" or \"opponent\", not {name:?}"
            )
        })
    };
    let items: Vec<&str> = text.split(',').filter(|s| !s.trim().is_empty()).collect();
    if items.iter().any(|s| s.contains('=')) {
        let slots = items
            .iter()
            .map(|item| {
                let (slot, name) = item
                    .split_once('=')
                    .ok_or_else(|| format!("policy_map item {item:?} is not slot=group"))?;
                let slot = slot
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| format!("policy_map slot {slot:?} is not an index"))?;
                Ok((slot, group(name)?))
            })
            .collect::<Result<Vec<_>, String>>()?;
        return Ok(PolicySpec::Slots(slots));
    }
    Ok(PolicySpec::List(
        items
            .iter()
            .map(|name| group(name))
            .collect::<Result<_, _>>()?,
    ))
}

/// The group of every one of `slots` agent slots.
fn groups_of(spec: &PolicySpec, slots: usize) -> Result<Vec<PolicyGroup>, BindError> {
    match spec {
        PolicySpec::List(groups) if groups.len() == slots => Ok(groups.clone()),
        PolicySpec::List(groups) => Err(BindError::Argument(format!(
            "policy_map has {} groups for {slots} agent slots",
            groups.len()
        ))),
        PolicySpec::Slots(items) => {
            let mut groups = vec![PolicyGroup::Learner; slots];
            for &(slot, g) in items {
                let entry = groups.get_mut(slot).ok_or_else(|| {
                    BindError::Argument(format!(
                        "policy_map slot {slot} is out of range for {slots} agent slots"
                    ))
                })?;
                *entry = g;
            }
            Ok(groups)
        }
    }
}

/// The map's learner and external part, and its heuristic, privileged
/// heuristic and opponent slots.
type Split = (PolicyMap, Vec<usize>, Vec<usize>, Vec<usize>);

fn split(groups: &[PolicyGroup]) -> Split {
    let base = groups
        .iter()
        .map(|&g| {
            if g.is_heuristic() || g == PolicyGroup::Opponent {
                PolicyGroup::Learner
            } else {
                g
            }
        })
        .collect();
    let slots = |want: PolicyGroup| {
        groups
            .iter()
            .enumerate()
            .filter(|&(_, &g)| g == want)
            .map(|(slot, _)| slot)
            .collect::<Vec<_>>()
    };
    (
        PolicyMap::from_groups(base),
        slots(PolicyGroup::Heuristic),
        slots(PolicyGroup::PrivilegedHeuristic),
        slots(PolicyGroup::Opponent),
    )
}

/// The league of `args` for `opponents`: both or neither (`ArgumentError`).
/// A league with an ONNX policy is refused (`CapabilityError`): ONNX
/// opponents from Python come later (rev, E3 plan ruling 2).
fn league_for(
    args: &PoolArgs,
    opponents: &[usize],
) -> Result<
    Option<(
        tessera_rl_engine::league::LeagueView,
        tessera_rl_league::RoleHandle,
    )>,
    BindError,
> {
    match (&args.league, opponents.is_empty()) {
        (None, true) => Ok(None),
        (None, false) => Err(BindError::Argument(
            "an \"opponent\" slot needs league=".to_owned(),
        )),
        (Some(_), true) => Err(BindError::Argument(
            "league= needs an \"opponent\" slot in policy_map".to_owned(),
        )),
        (Some((league, role)), false) => {
            let view = league
                .view(*role)
                .map_err(|e| BindError::Argument(e.to_string()))?;
            if let Some(i) = view
                .kinds()
                .iter()
                .position(|k| matches!(k, tessera_rl_engine::league::OpponentKind::Onnx { .. }))
            {
                return Err(BindError::Capability(format!(
                    "league policy {i} is an ONNX opponent: ONNX opponents from Python come later"
                )));
            }
            let member = league
                .member(*role)
                .map_err(|e| BindError::Argument(e.to_string()))?;
            Ok(Some((view, member)))
        }
    }
}

/// Applies `args.policy_map` to a perfect-information environment whose
/// `Heuristic::act` drives heuristic slots (ADR-0079 item 4.1).
pub(crate) fn perfect<E: Heuristic, M: MaskProvider<E>>(
    pool: EnvPool<E, M>,
    args: &PoolArgs,
) -> Result<EnvPool<E, M>, BindError> {
    let Some(spec) = &args.policy_map else {
        league_for(args, &[])?;
        return Ok(pool);
    };
    let (base, heuristic, privileged, opponents) = split(&groups_of(spec, E::MAX_AGENTS)?);
    if !privileged.is_empty() {
        return Err(BindError::Capability(
            "privileged_heuristic slots are for imperfect-information environments".to_owned(),
        ));
    }
    let pool = pool
        .with_policy_map(base)?
        .with_heuristic_slots(&heuristic)?;
    match league_for(args, &opponents)? {
        Some((view, member)) => Ok(pool.with_league(&opponents, view, Box::new(member))?),
        None => Ok(pool),
    }
}

/// Applies `args.policy_map` to an imperfect-information environment:
/// heuristic slots see the information state only (`InfoHeuristic`,
/// ADR-0079 item 4.2), privileged ones the full state (item 4.3). One pool
/// runs one kind.
pub(crate) fn info<E: InfoHeuristic + Heuristic, M: MaskProvider<E>>(
    pool: EnvPool<E, M>,
    args: &PoolArgs,
) -> Result<EnvPool<E, M>, BindError> {
    let Some(spec) = &args.policy_map else {
        league_for(args, &[])?;
        return Ok(pool);
    };
    let (base, heuristic, privileged, opponents) = split(&groups_of(spec, E::MAX_AGENTS)?);
    let pool = pool.with_policy_map(base)?;
    let pool = match (heuristic.is_empty(), privileged.is_empty()) {
        (_, true) => pool.with_info_heuristic_slots(&heuristic)?,
        (true, false) => pool.with_privileged_heuristic_slots(&privileged)?,
        (false, false) => {
            return Err(BindError::Capability(
                "a pool runs heuristic or privileged_heuristic slots, not both".to_owned(),
            ));
        }
    };
    match league_for(args, &opponents)? {
        Some((view, member)) => Ok(pool.with_info_league(&opponents, view, Box::new(member))?),
        None => Ok(pool),
    }
}

/// Applies `args.policy_map` to any environment, unless its maker already
/// did: learner and external groups only. A heuristic group needs an
/// environment whose maker runs heuristics (`CapabilityError`).
pub(crate) fn plain<E: tessera_rl_core::Environment, M: MaskProvider<E>>(
    pool: EnvPool<E, M>,
    args: &PoolArgs,
) -> Result<EnvPool<E, M>, BindError> {
    if args.league.is_some() {
        return Err(BindError::Capability(
            "this environment has no in-engine league opponents (league=)".to_owned(),
        ));
    }
    let Some(spec) = &args.policy_map else {
        return Ok(pool);
    };
    let groups = groups_of(spec, E::MAX_AGENTS)?;
    if groups.contains(&PolicyGroup::Opponent) {
        return Err(BindError::Capability(
            "this environment has no in-engine league opponents (\"opponent\")".to_owned(),
        ));
    }
    if pool.policy_groups() == groups.as_slice() {
        return Ok(pool);
    }
    if groups.iter().any(|g| g.is_heuristic()) {
        return Err(BindError::Capability(
            "this environment has no in-engine heuristic for policy_map".to_owned(),
        ));
    }
    Ok(pool.with_policy_map(PolicyMap::from_groups(groups))?)
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::PolicyGroup;

    use super::{PolicySpec, groups_of, policy_spec, split};

    #[test]
    fn test_policy_spec_list_and_slots_parse() {
        assert_eq!(
            policy_spec("learner, heuristic"),
            Ok(PolicySpec::List(vec![
                PolicyGroup::Learner,
                PolicyGroup::Heuristic
            ]))
        );
        assert_eq!(
            policy_spec("3=external"),
            Ok(PolicySpec::Slots(vec![(3, PolicyGroup::External)]))
        );
        assert!(policy_spec("learner,onnx").is_err());
        assert!(policy_spec("x=heuristic").is_err());
    }

    #[test]
    fn test_policy_spec_groups_fill_learners_and_check_counts() {
        let spec = PolicySpec::Slots(vec![(1, PolicyGroup::Heuristic)]);
        let groups = groups_of(&spec, 3).ok();
        assert_eq!(
            groups,
            Some(vec![
                PolicyGroup::Learner,
                PolicyGroup::Heuristic,
                PolicyGroup::Learner
            ])
        );
        assert!(groups_of(&spec, 1).is_err());
        assert!(groups_of(&PolicySpec::List(vec![PolicyGroup::Learner]), 2).is_err());
        let (base, heuristic, privileged, _) = split(&[
            PolicyGroup::External,
            PolicyGroup::Heuristic,
            PolicyGroup::PrivilegedHeuristic,
        ]);
        assert_eq!(
            base.groups(),
            &[
                PolicyGroup::External,
                PolicyGroup::Learner,
                PolicyGroup::Learner
            ]
        );
        assert_eq!((heuristic, privileged), (vec![1], vec![2]));
    }
}
