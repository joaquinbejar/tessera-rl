// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native` module: its definition, function table and
//! initialization, as safe impls of the typed layer of `tessera_rl_pyffi`
//! (ADR-0062 item 8). `tessera_rl_pyffi` defines `PyInit__native` (its
//! `native-module` feature) and runs [`NativeInit`], registered here.

use tessera_rl_pyffi::ffi::boundary::{self, Policy};
use tessera_rl_pyffi::ffi::capi::{self, CResult, Gil, Mine, Ref};
use tessera_rl_pyffi::ffi::module_def::{
    Function, FunctionNoArgs, FunctionRow, FunctionTable, ModuleDef, ModuleInit,
};

use super::{env_type, fork, search_group_type, search_type, snapshot_type, view_type};
use crate::errors::{Kind, error};
use tessera_rl_core::Adapters;

/// `version()`: the Tessera version.
struct Version;

impl FunctionNoArgs for Version {
    fn call(g: Gil<'_>) -> CResult<Mine<'_>> {
        capi::string(g, env!("CARGO_PKG_VERSION"))
    }
}

/// `physical_cores()`: the default worker count of the engine, one per
/// physical core this process may use (SPEC 6.1).
struct PhysicalCores;

impl FunctionNoArgs for PhysicalCores {
    fn call(g: Gil<'_>) -> CResult<Mine<'_>> {
        crate::api::int(g, tessera_rl_engine::physical_cores().get())
    }
}

/// `build_profile()`: `"release"` or `"debug"`, as the native module was
/// compiled (benchmark reports record it).
struct BuildProfile;

impl FunctionNoArgs for BuildProfile {
    fn call(g: Gil<'_>) -> CResult<Mine<'_>> {
        let text = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        };
        capi::string(g, text)
    }
}

/// `fp_boundary()`: which native floating-point boundary this build
/// carries (ADR-0041), or that there is none on this target.
struct FpBoundary;

impl FunctionNoArgs for FpBoundary {
    fn call(g: Gil<'_>) -> CResult<Mine<'_>> {
        capi::string(g, tessera_rl_pyffi::ffi::BOUNDARY)
    }
}

/// `set_fp_policy(name)`: `"canonicalize"` (the default) or `"reject"`
/// (ADR-0041).
struct SetFpPolicy;

impl Function for SetFpPolicy {
    fn call<'py>(g: Gil<'py>, args: Ref<'py>) -> CResult<Mine<'py>> {
        let items = capi::tuple_items(g, args, "the arguments")?;
        let [name] = items.as_slice() else {
            return Err(error(
                g,
                Kind::Argument,
                "set_fp_policy takes one argument".to_owned(),
            ));
        };
        let policy = match capi::to_string(g, *name, "the policy")?.as_str() {
            "canonicalize" => Policy::Canonicalize,
            "reject" => Policy::Reject,
            other => {
                return Err(error(
                    g,
                    Kind::Argument,
                    format!(
                        "the floating-point policy is \"canonicalize\" or \"reject\", not {other:?}"
                    ),
                ));
            }
        };
        boundary::set_policy(policy);
        Ok(capi::none(g))
    }
}

/// `get_fp_policy()`: the current policy's name.
struct GetFpPolicy;

impl FunctionNoArgs for GetFpPolicy {
    fn call(g: Gil<'_>) -> CResult<Mine<'_>> {
        let name = match boundary::policy() {
            Policy::Canonicalize => "canonicalize",
            Policy::Reject => "reject",
        };
        capi::string(g, name)
    }
}

/// `fp_non_canonical_entries()`: entries made with non-default caller
/// controls, process-wide (observability, ADR-0037).
struct FpNonCanonicalEntries;

impl FunctionNoArgs for FpNonCanonicalEntries {
    fn call(g: Gil<'_>) -> CResult<Mine<'_>> {
        capi::int_u64(g, tessera_rl_num::fp::non_canonical_entries())
    }
}

/// The Python name of a determinism tier (SPEC 8.4).
fn tier_name(tier: tessera_rl_core::Tier) -> &'static str {
    match tier {
        tessera_rl_core::Tier::CrossPlatform => "cross-platform",
        tessera_rl_core::Tier::SamePlatform => "same-platform",
    }
}

/// `env_records()`: one dict per `register!` id, in id order (ADR-0080
/// item 5): the facts of `registered_envs()` plus `adapters` (a tuple of
/// `"parallel"`, `"aec"`, `"turn_masked"`) and `deprecated` (`None` or a
/// dict with `since` and `note`).
struct EnvRecords;

impl FunctionNoArgs for EnvRecords {
    fn call(g: Gil<'_>) -> CResult<Mine<'_>> {
        let records = tessera_rl_core::envs::registered()
            .map_err(|e| error(g, Kind::Config, e.to_string()))?;
        let list = capi::list(g)?;
        for r in records {
            let max_agents =
                u64::try_from(r.max_agents()).map_err(|_| capi::out_of_range(g, "max_agents"))?;
            let item = capi::dict(g)?;
            capi::dict_set(g, &item, c"id", &capi::string(g, r.id())?)?;
            capi::dict_set(g, &item, c"version", &capi::int_u64(g, u64::from(r.version()))?)?;
            capi::dict_set(g, &item, c"max_agents", &capi::int_u64(g, max_agents)?)?;
            capi::dict_set(g, &item, c"tier", &capi::string(g, tier_name(r.tier()))?)?;
            capi::dict_set(
                g,
                &item,
                c"perfect_information",
                &capi::boolean(g, r.perfect_information()),
            )?;
            let a = r.adapters();
            let mut names = Vec::new();
            for (flag, name) in [
                (Adapters::PZ_PARALLEL, "parallel"),
                (Adapters::PZ_AEC, "aec"),
                (Adapters::PZ_TURN_MASKED, "turn_masked"),
            ] {
                if a.contains(flag) {
                    names.push(capi::string(g, name)?);
                }
            }
            capi::dict_set(g, &item, c"adapters", &capi::tuple(g, names)?)?;
            let deprecated = match r.deprecation() {
                None => capi::none(g),
                Some(d) => {
                    let dd = capi::dict(g)?;
                    capi::dict_set(g, &dd, c"since", &capi::string(g, d.since())?)?;
                    capi::dict_set(g, &dd, c"note", &capi::string(g, d.note())?)?;
                    dd
                }
            };
            capi::dict_set(g, &item, c"deprecated", &deprecated)?;
            capi::list_append(g, &list, &item)?;
        }
        Ok(list)
    }
}

/// `registered_envs()`: every environment registered in this build, in id
/// order, as `(id, version, max_agents, tier, perfect_information)` tuples
/// (ADR-0054 item 6). `version` is the outputs version (`ENV_VERSION`),
/// not the `-vN` suffix of the id. A duplicate id raises the configuration
/// error.
struct RegisteredEnvs;

impl FunctionNoArgs for RegisteredEnvs {
    fn call(g: Gil<'_>) -> CResult<Mine<'_>> {
        let records = tessera_rl_core::envs::registered()
            .map_err(|e| error(g, Kind::Config, e.to_string()))?;
        let list = capi::list(g)?;
        for r in records {
            let tier = tier_name(r.tier());
            let max_agents =
                u64::try_from(r.max_agents()).map_err(|_| capi::out_of_range(g, "max_agents"))?;
            let item = capi::tuple(
                g,
                vec![
                    capi::string(g, r.id())?,
                    capi::int_u64(g, u64::from(r.version()))?,
                    capi::int_u64(g, max_agents)?,
                    capi::string(g, tier)?,
                    capi::boolean(g, r.perfect_information()),
                ],
            )?;
            capi::list_append(g, &list, &item)?;
        }
        Ok(list)
    }
}

/// `_pendulum_rewards(rows)` (test hooks only): the kernel's raw f64 reward
/// of each little-endian `(f64 theta, f64 theta_dot, f32 u)` row, for the
/// ADR-0050 gate: the public reward is f32.
#[cfg(feature = "test-hooks")]
struct PendulumRewards;

#[cfg(feature = "test-hooks")]
impl tessera_rl_pyffi::ffi::module_def::FunctionArg for PendulumRewards {
    fn call<'py>(g: Gil<'py>, arg: Ref<'py>) -> CResult<Mine<'py>> {
        let rows = capi::bytes_of(g, arg, "rows")?;
        let out = crate::api::pendulum_rewards(rows);
        capi::bytes(g, &out)
    }
}

/// `_fail_after(n)` (test hooks only): the `n`-th next container insertion
/// fails.
#[cfg(feature = "test-hooks")]
struct FailAfter;

#[cfg(feature = "test-hooks")]
impl tessera_rl_pyffi::ffi::module_def::FunctionArg for FailAfter {
    fn call<'py>(g: Gil<'py>, arg: Ref<'py>) -> CResult<Mine<'py>> {
        capi::fail_after(capi::to_u32(g, arg, "n")?);
        Ok(capi::none(g))
    }
}

/// `_init_fails_after_register()` (test hooks only): a module
/// initialization that registers the exception classes, then fails.
#[cfg(feature = "test-hooks")]
struct InitFailsAfterRegister;

#[cfg(feature = "test-hooks")]
impl FunctionNoArgs for InitFailsAfterRegister {
    fn call(g: Gil<'_>) -> CResult<Mine<'_>> {
        tessera_rl_pyffi::ffi::module_def::init_fails_after_register(g)?;
        Ok(capi::none(g))
    }
}

const FUNCTION_COUNT: usize = if cfg!(feature = "test-hooks") { 34 } else { 28 };

static ROWS: [FunctionRow; FUNCTION_COUNT] = [
    FunctionRow::noargs::<Version>(c"version", c"version($module, /)\n--\n\nThe Tessera version."),
    FunctionRow::noargs::<PhysicalCores>(
        c"physical_cores",
        c"physical_cores($module, /)\n--\n\nPhysical cores this process may use (capped by CPU quotas): the engine's default worker count.",
    ),
    FunctionRow::varargs::<SetFpPolicy>(
        c"set_fp_policy",
        c"set_fp_policy($module, name, /)\n--\n\nset_fp_policy(name): \"canonicalize\" (default) or \"reject\".",
    ),
    FunctionRow::noargs::<GetFpPolicy>(
        c"get_fp_policy",
        c"get_fp_policy($module, /)\n--\n\nThe floating-point policy for non-default caller controls.",
    ),
    FunctionRow::noargs::<FpNonCanonicalEntries>(
        c"fp_non_canonical_entries",
        c"fp_non_canonical_entries($module, /)\n--\n\nEntries made with non-default caller floating-point controls.",
    ),
    FunctionRow::noargs::<FpBoundary>(
        c"fp_boundary",
        c"fp_boundary($module, /)\n--\n\nWhich native floating-point boundary this build carries.",
    ),
    FunctionRow::noargs::<BuildProfile>(
        c"build_profile",
        c"build_profile($module, /)\n--\n\nThe native build profile: release or debug.",
    ),
    FunctionRow::noargs::<EnvRecords>(
        c"env_records",
        c"env_records($module, /)\n--\n\nOne dict per registered id: the facts of registered_envs() plus adapters and deprecated (ADR-0080).",
    ),
    FunctionRow::noargs::<RegisteredEnvs>(
        c"registered_envs",
        c"registered_envs($module, /)\n--\n\nregistered_envs() -> [(id, version, max_agents, tier, perfect_information)]; version is ENV_VERSION, not the id's -vN",
    ),
    env_type::MAKERS[0],
    env_type::MAKERS[1],
    env_type::MAKERS[2],
    env_type::MAKERS[3],
    env_type::MAKERS[4],
    env_type::MAKERS[5],
    env_type::MAKERS[6],
    env_type::MAKERS[7],
    env_type::MAKERS[8],
    env_type::MAKERS[9],
    env_type::MAKERS[10],
    env_type::MAKERS[11],
    env_type::MAKERS[12],
    env_type::MAKERS[13],
    env_type::MAKERS[14],
    search_type::MAKER,
    search_group_type::MAKER,
    search_type::ONNX_LOAD,
    #[cfg(feature = "test-hooks")]
    env_type::TEST_MAKERS[0],
    #[cfg(feature = "test-hooks")]
    env_type::TEST_MAKERS[1],
    #[cfg(feature = "test-hooks")]
    env_type::TEST_MAKERS[2],
    #[cfg(feature = "test-hooks")]
    FunctionRow::arg::<PendulumRewards>(
        c"_pendulum_rewards",
        c"_pendulum_rewards($module, rows, /)\n--\n\nTest hook: the kernel's f64 rewards of (theta, theta_dot, u) rows.",
    ),
    #[cfg(feature = "test-hooks")]
    FunctionRow::arg::<FailAfter>(
        c"_fail_after",
        c"_fail_after($module, n, /)\n--\n\nTest hook: the n-th next container insertion fails.",
    ),
    #[cfg(feature = "test-hooks")]
    FunctionRow::noargs::<InitFailsAfterRegister>(
        c"_init_fails_after_register",
        c"_init_fails_after_register($module, /)\n--\n\nTest hook: a module initialization that fails after registering the exception classes.",
    ),
    FunctionRow::END,
];

static FUNCTIONS: FunctionTable = FunctionTable::new(&ROWS);

static MODULE: ModuleDef = ModuleDef::new(
    c"tessera_rl._native",
    c"Native Tessera bindings.",
    &FUNCTIONS,
);

/// `_native`'s initialization: the exception classes, the types, and the
/// fork handler.
struct NativeInit;

impl ModuleInit for NativeInit {
    const DEF: &'static ModuleDef = &MODULE;

    fn init<'py>(g: Gil<'py>, module: Ref<'py>) -> CResult<()> {
        crate::errors::register(g, module)?;
        env_type::register(g, module)?;
        view_type::register(g, module)?;
        snapshot_type::register(g, module)?;
        search_type::register(g, module)?;
        search_group_type::register(g, module)?;
        tessera_rl_pyffi::ffi::registry::attach(g)?;
        fork::register(g)
    }
}

tessera_rl_pyffi::export_module!(native: NativeInit);

#[cfg(test)]
mod tests {
    use super::tier_name;
    use tessera_rl_core::Tier;

    #[test]
    fn test_tier_name_both_tiers_their_python_names() {
        assert_eq!(tier_name(Tier::CrossPlatform), "cross-platform");
        assert_eq!(tier_name(Tier::SamePlatform), "same-platform");
    }
}
