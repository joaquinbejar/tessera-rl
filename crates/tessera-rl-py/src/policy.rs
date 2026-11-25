// SPDX-License-Identifier: MIT OR Apache-2.0

//! `make(..., policy_map=...)` (ADR-0079 items 2 and 5): the Python layer
//! passes the map as one string, and each environment's maker applies it
//! with the setter its heuristic allows.

use tessera_rl_core::{Heuristic, InfoHeuristic, PolicyGroup};
use tessera_rl_engine::league::{LeagueView, OpponentKind};
use tessera_rl_engine::{EnvPool, MaskProvider, PolicyMap};
use tessera_rl_league::RoleHandle;

use crate::env::{BindError, PoolArgs};

/// `make(..., onnx_opponents=...)` (#619): the ONNX models that serve a
/// league's ONNX policies, by content hash, and how ONNX Runtime is loaded
/// (ADR-0057; ADR-0088 item 2.3, erratum 2).
#[derive(Debug, Clone)]
#[cfg_attr(
    not(feature = "onnx"),
    expect(dead_code, reason = "read only by the onnx feature's opponents")
)]
pub(crate) struct OnnxOpponentArgs {
    /// The model files; each is identified by its content hash.
    pub(crate) models: Vec<std::path::PathBuf>,
    /// The ONNX Runtime library, by explicit path.
    pub(crate) library: std::path::PathBuf,
    /// `accept_telemetry=True`.
    pub(crate) accept_telemetry: bool,
    /// The model size cap; the default when `None`.
    pub(crate) max_model_bytes: Option<u64>,
}

/// A league view, its role handle, and its ONNX policies (index and
/// content hash).
type LeagueParts = (LeagueView, RoleHandle, Vec<(usize, [u8; 16])>);

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
/// A league with an ONNX policy needs `onnx_opponents=` (`ArgumentError`,
/// naming the policy), and `onnx_opponents=` needs a league; a build
/// without the `onnx` feature refuses `onnx_opponents=`
/// (`CapabilityError`, #619).
fn league_for(args: &PoolArgs, opponents: &[usize]) -> Result<Option<LeagueParts>, BindError> {
    if args.onnx_opponents.is_some() {
        if !cfg!(feature = "onnx") {
            return Err(not_built());
        }
        if args.league.is_none() {
            return Err(BindError::Argument(
                "onnx_opponents= needs league=".to_owned(),
            ));
        }
    }
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
            let onnx: Vec<(usize, [u8; 16])> = view
                .kinds()
                .iter()
                .enumerate()
                .filter_map(|(i, k)| match k {
                    OpponentKind::Onnx { content_hash } => Some((i, *content_hash)),
                    _ => None,
                })
                .collect();
            if let (Some((i, hash)), None) = (onnx.first(), &args.onnx_opponents) {
                return Err(BindError::Argument(format!(
                    "league policy {i} is ONNX {} and make has no onnx_opponents=",
                    hex(hash)
                )));
            }
            let member = league
                .member(*role)
                .map_err(|e| BindError::Argument(e.to_string()))?;
            Ok(Some((view, member, onnx)))
        }
    }
}

/// Applies `args.policy_map` to a perfect-information environment whose
/// `Heuristic::act` drives heuristic slots (ADR-0079 item 4.1).
pub(crate) fn perfect<E: Heuristic, M: MaskProvider<E>>(
    pool: EnvPool<E, M>,
    args: &PoolArgs,
) -> Result<(EnvPool<E, M>, crate::env::OnnxLoaded), BindError>
where
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
{
    let Some(spec) = &args.policy_map else {
        league_for(args, &[])?;
        return Ok((pool, Vec::new()));
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
        Some((view, member, onnx)) => {
            let pool = pool.with_league(&opponents, view, Box::new(member))?;
            onnx_opponents(pool, args, &onnx)
        }
        None => Ok((pool, Vec::new())),
    }
}

/// Applies `args.policy_map` to an imperfect-information environment:
/// heuristic slots see the information state only (`InfoHeuristic`,
/// ADR-0079 item 4.2), privileged ones the full state (item 4.3). One pool
/// runs one kind.
pub(crate) fn info<E: InfoHeuristic + Heuristic, M: MaskProvider<E>>(
    pool: EnvPool<E, M>,
    args: &PoolArgs,
) -> Result<(EnvPool<E, M>, crate::env::OnnxLoaded), BindError>
where
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
{
    let Some(spec) = &args.policy_map else {
        league_for(args, &[])?;
        return Ok((pool, Vec::new()));
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
        Some((view, member, onnx)) => {
            let pool = pool.with_info_league(&opponents, view, Box::new(member))?;
            onnx_opponents(pool, args, &onnx)
        }
        None => Ok((pool, Vec::new())),
    }
}

/// Applies `args.policy_map` to any environment, unless its maker already
/// did: learner and external groups only. A heuristic group needs an
/// environment whose maker runs heuristics (`CapabilityError`).
pub(crate) fn plain<E: tessera_rl_core::Environment, M: MaskProvider<E>>(
    pool: EnvPool<E, M>,
    args: &PoolArgs,
) -> Result<EnvPool<E, M>, BindError> {
    // A maker that runs heuristics applied the league already.
    if pool.policy_groups().contains(&PolicyGroup::Opponent) {
        return Ok(pool);
    }
    if args.league.is_some() || args.onnx_opponents.is_some() {
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

/// A content hash as 32 lowercase hex digits, as the league writes it.
fn hex(hash: &[u8; 16]) -> String {
    use std::fmt::Write as _;
    hash.iter().fold(String::with_capacity(32), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// The refusal of a build without the `onnx` feature.
fn not_built() -> BindError {
    BindError::Capability(
        "this build of tessera-rl has no ONNX opponents (the onnx feature)".to_owned(),
    )
}

/// Loads `args.onnx_opponents` for `pool`'s environment and sets them as
/// the pool's ONNX policies (ADR-0088 item 2.3): once, at construction,
/// reading the config the pool already holds. Every ONNX policy of the
/// league (`onnx`: index and content hash) must be served by a model
/// (`ArgumentError`, naming it); models no policy names are loaded and
/// unused.
#[cfg(feature = "onnx")]
fn onnx_opponents<E, M>(
    pool: EnvPool<E, M>,
    args: &PoolArgs,
    onnx: &[(usize, [u8; 16])],
) -> Result<(EnvPool<E, M>, crate::env::OnnxLoaded), BindError>
where
    E: tessera_rl_core::Environment,
    M: MaskProvider<E>,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
{
    use tessera_rl_engine::league::OpponentPolicies as _;
    let Some(o) = &args.onnx_opponents else {
        return Ok((pool, Vec::new()));
    };
    let mut options = tessera_rl_onnx::OnnxOptions::new(&o.library);
    if o.accept_telemetry {
        options.telemetry = tessera_rl_onnx::TelemetryPolicy::Accept;
    }
    if let Some(cap) = o.max_model_bytes {
        options.max_model_bytes = cap;
    }
    let paths: Vec<&std::path::Path> = o.models.iter().map(std::path::PathBuf::as_path).collect();
    let policies = tessera_rl_onnx::OnnxOpponents::new::<E>(pool.config(), &paths, &options)
        .map_err(|e| BindError::Onnx(e.to_string()))?;
    if let Some((i, hash)) = onnx.iter().find(|(_, h)| !policies.serves(h)) {
        return Err(BindError::Argument(format!(
            "league policy {i} is ONNX {} and no onnx_opponents model has that content hash",
            hex(hash)
        )));
    }
    // The record of exactly the models loaded (not an earlier read of the
    // files): the pool's wrapper keeps it (`Env.onnx_provenance`).
    let loaded: crate::env::OnnxLoaded = policies
        .models()
        .map(|(h, p)| (hex(h), crate::onnx_api::info_of(p)))
        .collect();
    Ok((pool.with_onnx_opponents(Box::new(policies))?, loaded))
}

/// Without the `onnx` feature `league_for` already refused
/// `onnx_opponents=`, and an ONNX policy without it.
#[cfg(not(feature = "onnx"))]
#[expect(
    clippy::unnecessary_wraps,
    reason = "the same signature as the onnx feature's loader"
)]
fn onnx_opponents<E: tessera_rl_core::Environment, M: MaskProvider<E>>(
    pool: EnvPool<E, M>,
    _: &PoolArgs,
    _: &[(usize, [u8; 16])],
) -> Result<(EnvPool<E, M>, crate::env::OnnxLoaded), BindError> {
    Ok((pool, Vec::new()))
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
