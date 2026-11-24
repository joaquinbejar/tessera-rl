// SPDX-License-Identifier: MIT OR Apache-2.0

//! A test host environment wheel (#272 step 4): the counter of
//! `tests/host-env`, exported with `export_host_env!`, and two test
//! functions for the "two front ends" check (ADR-0065 item 3):
//!
//! - `_config_python(**config)`: the canonical form of `Rules` decoded
//!   from Python keyword arguments;
//! - `_config_rust(case)`: the canonical form of `Rules` decoded from the
//!   Rust source of the named case (`rust_case`).
//!
//! Both raise the same Python error for a source that does not decode.
//! This crate forbids `unsafe`, so it also checks that the macro's
//! expansion has none.

use tessera_rl_core::config::canonical_bytes;
use tessera_rl_core::{ConfigDecode, ConfigItem, ConfigSource};
use tessera_rl_macros::EnvConfig;
use tessera_rl_num::{Fx32, Fx64};
use tessera_rl_pyffi::ffi::boundary::{self, Policy};
use tessera_rl_pyffi::ffi::capi::{self, CResult, Gil, Mine, Ref};
use tessera_rl_pyffi::ffi::module_def::{FunctionArg, FunctionKw, FunctionNoArgs, FunctionRow};
use tessera_rl_pyffi::host::config::raise_decode;
use tessera_rl_pyffi::host::factory::config_canonical;

/// A nested table.
#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct Inner {
    /// A float.
    #[config(default = 1.5)]
    scale: f64,
    /// An integer.
    count: u8,
}

/// One field of every type a config takes, for the decoding rules.
#[derive(EnvConfig, Debug, Clone, PartialEq)]
struct Rules {
    /// An unsigned byte.
    byte: u8,
    /// A signed 64-bit integer.
    wide: i64,
    /// An unsigned 64-bit integer.
    big: u64,
    /// An f32 (rounded to nearest from an f64 source).
    single: f32,
    /// An f64.
    double: f64,
    /// A boolean.
    #[config(default = true)]
    flag: bool,
    /// Q16.16.
    fx32: Fx32,
    /// Q32.32.
    fx64: Fx64,
    /// A nested config.
    inner: Inner,
    /// Adaptive: refused in a source.
    #[config(default = 2, adaptive)]
    level: u8,
}

/// The Rust source of each named case. The Python test passes the same
/// values as keyword arguments.
fn rust_case<R>(name: &str, decode: impl FnOnce(&ConfigSource<'_>) -> R) -> Option<R> {
    let nested = [
        ("scale", ConfigItem::Float(-2.25)),
        ("count", ConfigItem::Int(7)),
    ];
    let deep9 = [("x", ConfigItem::Int(1))];
    let deep8 = [("x", ConfigItem::Table(ConfigSource::new(&deep9)))];
    let deep7 = [("x", ConfigItem::Table(ConfigSource::new(&deep8)))];
    let deep6 = [("x", ConfigItem::Table(ConfigSource::new(&deep7)))];
    let deep5 = [("x", ConfigItem::Table(ConfigSource::new(&deep6)))];
    let deep4 = [("x", ConfigItem::Table(ConfigSource::new(&deep5)))];
    let deep3 = [("x", ConfigItem::Table(ConfigSource::new(&deep4)))];
    let deep2 = [("scale", ConfigItem::Table(ConfigSource::new(&deep3)))];
    let entries: &[(&str, ConfigItem<'_>)] = match name {
        "empty" => &[],
        "ints" => &[
            ("byte", ConfigItem::Int(255)),
            ("wide", ConfigItem::Int(i128::from(i64::MIN))),
            ("big", ConfigItem::Int(i128::from(u64::MAX))),
        ],
        "floats" => &[
            ("single", ConfigItem::Float(0.1)),
            ("double", ConfigItem::Float(-0.0)),
        ],
        "f32_tie" => &[(
            "single",
            ConfigItem::Float(1.0 + f64::from(f32::EPSILON) / 2.0),
        )],
        "f32_huge" => &[("single", ConfigItem::Float(1e300))],
        "int_into_float" => &[
            ("double", ConfigItem::Int(1 << 53)),
            ("single", ConfigItem::Int(1 << 24)),
        ],
        "fixed" => &[
            ("fx32", ConfigItem::Str("-1.5")),
            ("fx64", ConfigItem::Str("0.25")),
        ],
        "bool" => &[("flag", ConfigItem::Bool(false))],
        "nested" => &[("inner", ConfigItem::Table(ConfigSource::new(&nested)))],
        "unknown" => &[("colour", ConfigItem::Int(1))],
        "adaptive" => &[("level", ConfigItem::Int(3))],
        "byte_range" => &[("byte", ConfigItem::Int(256))],
        "float_inexact" => &[("double", ConfigItem::Int((1 << 53) + 1))],
        "fixed_inexact" => &[("fx32", ConfigItem::Str("0.1"))],
        "fixed_float" => &[("fx32", ConfigItem::Float(1.5))],
        "bool_int" => &[("flag", ConfigItem::Int(1))],
        "int_bool" => &[("byte", ConfigItem::Bool(true))],
        "nested_not_table" => &[("inner", ConfigItem::Int(1))],
        "nan" => &[("double", ConfigItem::Float(f64::NAN))],
        "deep_table" => &[("inner", ConfigItem::Table(ConfigSource::new(&deep2)))],
        _ => return None,
    };
    Some(decode(&ConfigSource::new(entries)))
}

struct ConfigPython;

impl FunctionKw for ConfigPython {
    fn call<'py>(g: Gil<'py>, args: Ref<'py>, kwargs: Option<Ref<'py>>) -> CResult<Mine<'py>> {
        config_canonical::<Rules>(g, args, kwargs)
    }
}

struct ConfigRust;

impl FunctionArg for ConfigRust {
    fn call<'py>(g: Gil<'py>, arg: Ref<'py>) -> CResult<Mine<'py>> {
        let name = capi::to_string(g, arg, "case")?;
        match rust_case(&name, Rules::decode) {
            Some(Ok(cfg)) => capi::bytes(g, &canonical_bytes(&cfg)),
            Some(Err(e)) => Err(raise_decode(g, &e)),
            None => Err(capi::raise(
                g,
                capi::type_error(),
                &format!("no case {name:?}"),
            )),
        }
    }
}

/// `make_respawn(**options, **config)`: the spawning test environment
/// `test/Respawn-v1` (#71), built as `make` builds the counter.
struct MakeRespawn;

impl FunctionKw for MakeRespawn {
    fn call<'py>(g: Gil<'py>, args: Ref<'py>, kwargs: Option<Ref<'py>>) -> CResult<Mine<'py>> {
        tessera_rl_pyffi::host::factory::make::<
            tessera_rl_test_host_env::Respawner,
            { tessera_rl_test_host_env::RESPAWN_MAX_AGENTS },
        >(g, args, kwargs)
    }
}

/// `make_multihead(**options, **config)`: the two-head masked test
/// environment `test/MultiHead-v1` (#311), built as `make` builds the counter.
struct MakeMultiHead;

impl FunctionKw for MakeMultiHead {
    fn call<'py>(g: Gil<'py>, args: Ref<'py>, kwargs: Option<Ref<'py>>) -> CResult<Mine<'py>> {
        tessera_rl_pyffi::host::factory::make::<
            tessera_rl_test_host_env::MultiHead,
            { tessera_rl_test_host_env::MULTIHEAD_MAX_AGENTS },
        >(g, args, kwargs)
    }
}

/// `make_refuser(**options, **config)`: the refusing test environment
/// `test/Refuser-v1` (ADR-0070), a host protocol 2 object.
struct MakeRefuser;

impl FunctionKw for MakeRefuser {
    fn call<'py>(g: Gil<'py>, args: Ref<'py>, kwargs: Option<Ref<'py>>) -> CResult<Mine<'py>> {
        tessera_rl_pyffi::host::factory::make::<
            tessera_rl_test_host_env::Refuser,
            { tessera_rl_test_host_env::REFUSER_MAX_AGENTS },
        >(g, args, kwargs)
    }
}

/// `_fp_policy()`: this module's floating-point policy (ADR-0041), as the
/// host object's `_tessera_set_fp_policy` set it.
struct FpPolicy;

impl FunctionNoArgs for FpPolicy {
    fn call(g: Gil<'_>) -> CResult<Mine<'_>> {
        let name = match boundary::policy() {
            Policy::Canonicalize => "canonicalize",
            Policy::Reject => "reject",
        };
        capi::string(g, name)
    }
}

tessera_rl_pyffi::export_host_env!(
    tessera_rl_test_host_env::Counter,
    module = c"_tessera_host",
    snapshots,
    functions = [
        FunctionRow::kw::<ConfigPython>(
            c"_config_python",
            c"The canonical form of the test config decoded from keyword arguments."
        ),
        FunctionRow::arg::<ConfigRust>(
            c"_config_rust",
            c"The canonical form of the test config decoded from a named Rust case."
        ),
        FunctionRow::noargs::<FpPolicy>(c"_fp_policy", c"This module's floating-point policy."),
        FunctionRow::kw::<MakeRespawn>(
            c"make_respawn",
            c"make_respawn(*, num_envs, seed, auto_reset, max_episode_steps, workers, cpu_set, **config)"
        ),
        FunctionRow::kw::<MakeMultiHead>(
            c"make_multihead",
            c"make_multihead(*, num_envs, seed, auto_reset, max_episode_steps, workers, cpu_set, **config)"
        ),
        FunctionRow::kw::<MakeRefuser>(
            c"make_refuser",
            c"make_refuser(*, num_envs, seed, auto_reset, max_episode_steps, workers, cpu_set, **config)"
        ),
    ]
);
