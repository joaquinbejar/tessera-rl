// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a host object says about itself (SPEC 14.4.2 and 14.4.3 draft,
//! ADR-0065 item 5): the `layout()` dict and the version range.

use tessera_rl_core::TurnStructure;
use tessera_rl_core::space::DType;
use tessera_rl_engine::{AutoReset, DynHostPool};

use crate::ffi::capi::{self, CResult, Gil, Mine};

/// The NumPy dtype name of `dtype` (SPEC 11.1).
pub(super) const fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::I8 => "int8",
        DType::I16 => "int16",
        DType::I32 => "int32",
        DType::I64 => "int64",
        DType::U8 => "uint8",
        DType::U16 => "uint16",
        DType::U32 => "uint32",
        DType::F32 => "float32",
        DType::F64 => "float64",
    }
}

/// `MAJOR.MINOR.PATCH` of this crate, the `tessera_rl` release it ships
/// with: the lowest `tessera_rl` a host object built with it accepts. A
/// pre-release or build suffix is dropped: the protocol's versions are
/// plain `MAJOR.MINOR.PATCH` (SPEC 14.4.2).
#[must_use]
pub fn version_min() -> String {
    plain(env!("CARGO_PKG_VERSION"))
}

/// `version` without a pre-release or build suffix.
fn plain(version: &str) -> String {
    version
        .split(['-', '+'])
        .next()
        .unwrap_or(version)
        .to_owned()
}

/// The first `tessera_rl` a host object built with this crate does not
/// accept: the next minor release before 1.0 (a 0.x minor release may break
/// protocol surfaces), the next major release from 1.0, since every 1.x
/// release supports the host protocols of 1.0 (ADR-0094 item 4).
#[must_use]
pub fn version_below() -> String {
    next_breaking(env!("CARGO_PKG_VERSION"))
}

/// `0.(MINOR + 1).0` for a `0.MINOR.PATCH` `version`, `(MAJOR + 1).0.0` from
/// 1.0, ignoring any pre-release or build suffix; the version itself if it
/// is not `MAJOR.MINOR.PATCH` or the next number does not fit in `u64`.
fn next_breaking(version: &str) -> String {
    let core = version.split(['-', '+']).next().unwrap_or(version);
    let mut parts = core.split('.');
    match (
        parts.next().and_then(|p| p.parse::<u64>().ok()),
        parts.next().and_then(|p| p.parse::<u64>().ok()),
    ) {
        (Some(0), Some(minor)) => minor
            .checked_add(1)
            .map_or_else(|| version.to_owned(), |next| format!("0.{next}.0")),
        (Some(major), Some(_)) => major
            .checked_add(1)
            .map_or_else(|| version.to_owned(), |next| format!("{next}.0.0")),
        _ => version.to_owned(),
    }
}

fn int(g: Gil<'_>, v: usize) -> CResult<Mine<'_>> {
    capi::int_u64(g, u64::try_from(v).unwrap_or(u64::MAX))
}

fn list_of<'py>(
    g: Gil<'py>,
    items: impl IntoIterator<Item = CResult<Mine<'py>>>,
) -> CResult<Mine<'py>> {
    let list = capi::list(g)?;
    for item in items {
        capi::list_append(g, &list, &item?)?;
    }
    Ok(list)
}

/// The `auto_reset` option's protocol name.
pub(super) const fn auto_reset_name(mode: AutoReset) -> &'static str {
    match mode {
        AutoReset::NextStep => "next_step",
        AutoReset::SameStep => "same_step",
        AutoReset::Disabled => "disabled",
    }
}

/// `layout()`: every key of SPEC 14.4.3 (draft), with ADR-0065's
/// `initial_agents` and `spawns` and ADR-0070's `refuses_agents`; `agents`
/// is the capacity.
pub(super) fn layout<'py>(g: Gil<'py>, pool: &dyn DynHostPool) -> CResult<Mine<'py>> {
    let l = pool.layout();
    let reg = pool.registration();
    let d = capi::dict(g)?;
    capi::dict_set(g, &d, c"num_envs", &int(g, l.num_envs)?)?;
    capi::dict_set(g, &d, c"agents", &int(g, l.agents)?)?;
    capi::dict_set(g, &d, c"initial_agents", &int(g, l.initial_agents)?)?;
    capi::dict_set(g, &d, c"spawns", &capi::boolean(g, l.spawns))?;
    capi::dict_set(
        g,
        &d,
        c"refuses_agents",
        &capi::boolean(g, l.refuses_agents),
    )?;
    capi::dict_set(g, &d, c"snapshots", &capi::boolean(g, l.snapshots))?;
    let choices = match l.discrete_heads.as_slice() {
        [n] => u64::from(*n),
        _ => 0,
    };
    capi::dict_set(g, &d, c"action_choices", &capi::int_u64(g, choices)?)?;
    let bounds = match l.continuous_heads.as_slice() {
        [(low, high)] => capi::tuple(g, vec![capi::float(g, *low)?, capi::float(g, *high)?])?,
        _ => capi::none(g),
    };
    capi::dict_set(g, &d, c"action_bounds", &bounds)?;
    let nvec = list_of(
        g,
        l.discrete_heads
            .iter()
            .map(|&n| capi::int_u64(g, u64::from(n))),
    )?;
    capi::dict_set(g, &d, c"action_nvec", &nvec)?;
    capi::dict_set(g, &d, c"mask_len", &int(g, l.mask_len)?)?;
    let heads = list_of(
        g,
        l.mask_heads.iter().map(|m| {
            capi::tuple(
                g,
                vec![
                    int(g, m.offset)?,
                    list_of(g, m.parent_sizes.iter().flatten().map(|&p| int(g, p)))?,
                    int(g, m.size)?,
                ],
            )
        }),
    )?;
    capi::dict_set(g, &d, c"mask_heads", &heads)?;
    // Entity groups (ADR-0051, #385): read generically by every protocol 1
    // wrapper, so a nonzero value is an addition, not a protocol change.
    capi::dict_set(g, &d, c"entity_group_heads", &int(g, l.entity_group_heads)?)?;
    let turns = match reg.turns() {
        TurnStructure::Simultaneous => "simultaneous",
        TurnStructure::Sequential => "sequential",
        TurnStructure::Mixed => "mixed",
    };
    capi::dict_set(g, &d, c"turns", &capi::string(g, turns)?)?;
    capi::dict_set(g, &d, c"outputs", &capi::string(g, "borrowed")?)?;
    let leaves = list_of(
        g,
        l.leaves.iter().map(|leaf| {
            capi::tuple(
                g,
                vec![
                    capi::string(g, &leaf.name)?,
                    capi::string(g, dtype_name(leaf.dtype))?,
                    list_of(g, leaf.shape.iter().map(|&s| capi::int_u64(g, s.into())))?,
                ],
            )
        }),
    )?;
    capi::dict_set(g, &d, c"leaves", &leaves)?;
    capi::dict_set(g, &d, c"terminal", &capi::boolean(g, l.terminal))?;
    capi::dict_set(g, &d, c"env_id", &capi::string(g, reg.id())?)?;
    capi::dict_set(
        g,
        &d,
        c"env_version",
        &capi::int_u64(g, u64::from(reg.version()))?,
    )?;
    capi::dict_set(
        g,
        &d,
        c"perfect_information",
        &capi::boolean(g, reg.perfect_information()),
    )?;
    capi::dict_set(g, &d, c"kind", &capi::string(g, "host")?)?;
    let mode = auto_reset_name(pool.options().auto_reset);
    capi::dict_set(g, &d, c"auto_reset", &capi::string(g, mode)?)?;
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::{next_breaking, plain, version_below, version_min};

    #[test]
    fn test_host_version_range_is_the_next_minor_before_1_0_and_the_next_major_from_it() {
        assert_eq!(next_breaking("0.2.0"), "0.3.0");
        assert_eq!(next_breaking("0.3.0-rc.1"), "0.4.0");
        assert_eq!(next_breaking("0.99.7"), "0.100.0");
        assert_eq!(next_breaking("1.0.0"), "2.0.0");
        assert_eq!(next_breaking("1.4.9"), "2.0.0");
        assert_eq!(next_breaking("1.0.0-rc.1+build.2"), "2.0.0");
        assert_eq!(next_breaking("2.7.1"), "3.0.0");
        assert_eq!(next_breaking("garbage"), "garbage");
        let top_minor = format!("0.{}.0", u64::MAX);
        assert_eq!(next_breaking(&top_minor), top_minor);
        let top_major = format!("{}.0.0", u64::MAX);
        assert_eq!(next_breaking(&top_major), top_major);
        assert_eq!(plain("0.3.0-rc.1+build.5"), "0.3.0");
        assert_eq!(plain("0.2.0"), "0.2.0");
        assert_eq!(version_min(), plain(env!("CARGO_PKG_VERSION")));
        assert_eq!(version_below(), next_breaking(env!("CARGO_PKG_VERSION")));
    }
}
