// SPDX-License-Identifier: MIT OR Apache-2.0

//! The implementation of every Python-visible operation, over `ffi::capi`
//! (ADR-0041 callouts). Arguments are read only from exact built-in types;
//! the Python layer converts anything else before calling.

use core::sync::atomic::{AtomicBool, Ordering};
use core::time::Duration;

use std::sync::Arc;

use tessera_rl_engine::{ActionInput, EngineError};

use crate::batched::{AsyncOps, BatchOutputs};
use crate::env::{BindError, Owned, PoolArgs, PoolOps, ResetSeed, TagFields};
use crate::errors::{Kind, error, to_py};
use crate::ffi::capi::{self, CResult, Gil, Mine, Raised, Ref};
use crate::ffi::inner::Inner;
use crate::ffi::snapshot_type;
use crate::ffi::view_type::{Track, new_view};
use crate::split::{HalfOutput, Polled, SplitOps};
use crate::view::{ViewSet, ViewSpec};

/// Keyword arguments, as exact `(str, value)` pairs.
pub(crate) struct Kwargs<'py>(Vec<(String, Mine<'py>)>);

impl<'py> Kwargs<'py> {
    pub(crate) fn parse(g: Gil<'py>, kwargs: Option<Ref<'_>>) -> CResult<Self> {
        Ok(Self(match kwargs {
            Some(d) => capi::dict_items(g, d, "keyword arguments")?,
            None => Vec::new(),
        }))
    }

    /// The value of `key`, absent or `None` as `None`.
    pub(crate) fn get(&self, key: &str) -> Option<Ref<'_>> {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_ref())
            .filter(|v| !v.is_none())
    }

    pub(crate) fn check(&self, g: Gil, allowed: &[&[&str]]) -> CResult<()> {
        for (key, _) in &self.0 {
            if !allowed.iter().any(|set| set.contains(&key.as_str())) {
                return Err(error(
                    g,
                    Kind::Argument,
                    format!("unexpected argument {key:?}"),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn u32(&self, g: Gil, key: &str) -> CResult<Option<u32>> {
        self.get(key).map(|v| capi::to_u32(g, v, key)).transpose()
    }

    pub(crate) fn i32(&self, g: Gil, key: &str) -> CResult<Option<i32>> {
        self.get(key)
            .map(|v| {
                let x = capi::to_i64(g, v, key)?;
                i32::try_from(x).map_err(|_| capi::out_of_range(g, key))
            })
            .transpose()
    }

    pub(crate) fn bool(&self, g: Gil, key: &str) -> CResult<Option<bool>> {
        self.get(key).map(|v| capi::to_bool(g, v, key)).transpose()
    }

    pub(crate) fn u64(&self, g: Gil, key: &str) -> CResult<Option<u64>> {
        self.get(key).map(|v| capi::to_u64(g, v, key)).transpose()
    }

    pub(crate) fn string(&self, g: Gil, key: &str) -> CResult<Option<String>> {
        self.get(key)
            .map(|v| capi::to_string(g, v, key))
            .transpose()
    }

    pub(crate) fn f64(&self, g: Gil, key: &str) -> CResult<Option<f64>> {
        self.get(key).map(|v| capi::to_f64(g, v, key)).transpose()
    }
}

/// Refuses positional arguments to a keyword-only native function (#376):
/// its text signature is keyword-only, so they are an error, not ignored.
pub(crate) fn no_positional(g: Gil, args: Ref<'_>, name: &str) -> CResult<()> {
    if capi::tuple_items(g, args, "the arguments")?.is_empty() {
        Ok(())
    } else {
        Err(error(
            g,
            Kind::Argument,
            format!("{name} takes keyword arguments only"),
        ))
    }
}

const POOL_KEYS: [&str; 13] = [
    "mode",
    "batch_size",
    "gather",
    "num_envs",
    "seed",
    "max_episode_steps",
    "auto_reset",
    "chunk_size",
    "schedule",
    "workers",
    "spin",
    "wait",
    "wait_cap_us",
];

/// The `outputs` mode: `"borrowed"` (the native default, ADR-0003) or
/// `"owned"`.
fn borrowed(g: Gil, kw: &Kwargs<'_>) -> CResult<bool> {
    match kw.string(g, "outputs")?.as_deref() {
        None | Some("borrowed") => Ok(true),
        Some("owned") => Ok(false),
        Some(other) => Err(error(
            g,
            Kind::Argument,
            format!("outputs must be \"borrowed\" or \"owned\", not {other:?}"),
        )),
    }
}

/// The `mode` (`"sync"` by default, `"pipelined"` or `"async"`) and, for
/// Async, `batch_size` and `gather`: whether it is Pipelined, and the Async
/// options.
fn mode_args(g: Gil, kw: &Kwargs<'_>) -> CResult<(bool, Option<(usize, BatchOutputs)>)> {
    let mode = kw.string(g, "mode")?;
    let batch = kw.u32(g, "batch_size")?;
    let gather = kw.string(g, "gather")?;
    if mode.as_deref() != Some("async") && (batch.is_some() || gather.is_some()) {
        return Err(error(
            g,
            Kind::Argument,
            "batch_size and gather belong to mode=\"async\"".to_owned(),
        ));
    }
    match mode.as_deref() {
        None | Some("sync") => Ok((false, None)),
        Some("pipelined") => Ok((true, None)),
        Some("async") => {
            let Some(batch) = batch else {
                return Err(error(
                    g,
                    Kind::Argument,
                    "mode=\"async\" needs batch_size".to_owned(),
                ));
            };
            let outputs = match (borrowed(g, kw)?, gather.as_deref()) {
                (true, None | Some("gathered")) => BatchOutputs::Gathered,
                (true, Some("indexed")) => BatchOutputs::Indexed,
                (false, None | Some("gathered")) => BatchOutputs::Owned,
                (false, Some("indexed")) => {
                    return Err(error(
                        g,
                        Kind::Argument,
                        "gather=\"indexed\" needs outputs=\"borrowed\"".to_owned(),
                    ));
                }
                (_, Some(other)) => {
                    return Err(error(
                        g,
                        Kind::Argument,
                        format!("gather must be \"gathered\" or \"indexed\", not {other:?}"),
                    ));
                }
            };
            Ok((
                false,
                Some((usize::try_from(batch).unwrap_or(usize::MAX), outputs)),
            ))
        }
        Some(other) => Err(error(
            g,
            Kind::Argument,
            format!("mode must be \"sync\", \"pipelined\" or \"async\", not {other:?}"),
        )),
    }
}

fn pool_args(g: Gil, kw: &Kwargs<'_>, default_limit: u32) -> CResult<PoolArgs> {
    let (pipelined, asynced) = mode_args(g, kw)?;
    Ok(PoolArgs {
        pipelined,
        asynced,
        num_envs: kw.u32(g, "num_envs")?.unwrap_or(1),
        seed: kw.u64(g, "seed")?.unwrap_or(0),
        max_episode_steps: Some(kw.u32(g, "max_episode_steps")?.unwrap_or(default_limit)),
        auto_reset: kw
            .string(g, "auto_reset")?
            .unwrap_or_else(|| "next_step".to_owned()),
        chunk_size: kw.u32(g, "chunk_size")?,
        schedule: kw
            .string(g, "schedule")?
            .unwrap_or_else(|| "static".to_owned()),
        workers: kw
            .u32(g, "workers")?
            .map(|w| usize::try_from(w).unwrap_or(usize::MAX)),
        spin: kw.u32(g, "spin")?,
        wait: kw.string(g, "wait")?,
        wait_cap_us: kw.u32(g, "wait_cap_us")?,
    })
}

/// The kernel's raw f64 reward of each `(f64, f64, f32)` little-endian row
/// (test hooks only, ADR-0050).
#[cfg(feature = "test-hooks")]
pub(crate) fn pendulum_rewards(rows: &[u8]) -> Vec<u8> {
    let (rows, _) = rows.as_chunks::<20>();
    let mut out = Vec::with_capacity(rows.len() * 8);
    tessera_rl_num::fp::with_canonical(|| {
        for row in rows {
            let (theta, rest) = row.split_at(8);
            let (theta_dot, u) = rest.split_at(8);
            let read = |b: &[u8]| f64::from_le_bytes(b.try_into().unwrap_or([0; 8]));
            let u = f32::from_le_bytes(u.try_into().unwrap_or([0; 4]));
            let reward =
                tessera_rl_classic::pendulum::physics::reward([read(theta), read(theta_dot)], u);
            out.extend_from_slice(&reward.to_le_bytes());
        }
    });
    out
}

/// `_make_panic_counter(**kwargs)` (test hooks only): a pool that panics.
#[cfg(feature = "test-hooks")]
pub(crate) fn make_panic_counter(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &["outputs"]])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, 100)?;
    let ops = g
        .detach(|| crate::env::panic_counter(&args))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// `_make_continuous_probe(**kwargs)` (test hooks only): one continuous head.
#[cfg(feature = "test-hooks")]
pub(crate) fn make_continuous_probe(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &["outputs"]])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, 100)?;
    let ops = g
        .detach(|| crate::env::continuous_probe(&args))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// `_make_multi_probe(**kwargs)` (test hooks only): two discrete heads.
#[cfg(feature = "test-hooks")]
pub(crate) fn make_multi_probe(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &["outputs"]])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, 100)?;
    let ops = g
        .detach(|| crate::env::multi_probe(&args))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// `make_pendulum(**kwargs)`: the pool of a pendulum environment object.
/// `max_episode_steps` defaults to 200 (`Pendulum-v1`).
pub(crate) fn make_pendulum(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &["outputs"]])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, tessera_rl_classic::pendulum::MAX_EPISODE_STEPS)?;
    let ops = g
        .detach(|| crate::env::pendulum(&args))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// `make_cartpole(**kwargs)`: the pool of a CartPole environment object.
/// `max_episode_steps` defaults to 500 (`CartPole-v1`).
pub(crate) fn make_cartpole(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &["outputs"]])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, tessera_rl_classic::cartpole::MAX_EPISODE_STEPS)?;
    let ops = g
        .detach(|| crate::env::cartpole(&args))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// `make_acrobot(**kwargs)`: the pool of an acrobot environment object.
/// `max_episode_steps` defaults to 500 (`Acrobot-v1`).
pub(crate) fn make_acrobot(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &["outputs"]])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, tessera_rl_classic::acrobot::MAX_EPISODE_STEPS)?;
    let ops = g
        .detach(|| crate::env::acrobot(&args))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// `make_mountaincar(**kwargs)`: the pool of a mountain car environment
/// object. `max_episode_steps` defaults to 200 (`MountainCar-v0`).
pub(crate) fn make_mountaincar(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &["outputs"]])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, tessera_rl_classic::mountaincar::MAX_EPISODE_STEPS)?;
    let ops = g
        .detach(|| crate::env::mountaincar(&args))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// `make_grid(**kwargs)`: the pool of a reference grid world environment
/// object. The world truncates at `max_steps`, and the engine's own limit is
/// set above the largest value, so it never acts first.
pub(crate) fn make_grid(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(
        g,
        &[&POOL_KEYS, &["outputs", "size", "max_steps", "layout"]],
    )?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, 10_001)?;
    let (size, max_steps) = (kw.u32(g, "size")?, kw.u32(g, "max_steps")?);
    let layout = kw
        .get("layout")
        .map(|v| asset_arg::<tessera_rl_classic::GridLayout>(g, v, "layout"))
        .transpose()?;
    let ops = g
        .detach(|| crate::env::grid(&args, size, max_steps, layout))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// An asset argument (ADR-0077): a bundled asset's name (`str`) or its
/// bytes (`bytes`, at most 16 MiB). A refused asset is `ConfigError`.
fn asset_arg<T: tessera_rl_core::AssetData>(
    g: Gil,
    v: Ref<'_>,
    what: &str,
) -> CResult<tessera_rl_core::Asset<T>> {
    let built = if capi::kind(v) == capi::Kind::Str {
        tessera_rl_core::Asset::bundled(&capi::to_string(g, v, what)?)
    } else {
        tessera_rl_core::Asset::from_bytes(capi::bytes_of(g, v, what)?)
    };
    built.map_err(|e| error(g, Kind::Config, format!("{what}: {e}")))
}

/// `make_keydoor(**kwargs)`: the pool of a key and door environment object;
/// `max_episode_steps` defaults to 512.
pub(crate) fn make_keydoor(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &["outputs", "size", "agents"]])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, 512)?;
    let (size, agents) = (kw.u32(g, "size")?, kw.u32(g, "agents")?);
    let ops = g
        .detach(|| crate::env::keydoor(&args, size, agents))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// `make_paddleduel(**kwargs)`: the pool of a paddle duel environment
/// object; `obs` is `"pixels"` (the default) or `"state"`; `max_episode_steps`
/// defaults to 5000 decisions.
pub(crate) fn make_paddleduel(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(
        g,
        &[
            &POOL_KEYS,
            &[
                "outputs",
                "agents",
                "frameskip",
                "max_score",
                "bounce",
                "obs",
            ],
        ],
    )?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, 5000)?;
    let fields = crate::env::PaddleDuelFields {
        agents: kw.u32(g, "agents")?,
        frameskip: kw.u32(g, "frameskip")?,
        max_score: kw.u32(g, "max_score")?,
        bounce: kw.u32(g, "bounce")?,
    };
    let obs = kw.string(g, "obs")?;
    let ops = g
        .detach(|| crate::env::paddleduel(&args, &fields, obs.as_deref()))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// `make_brickwall(**kwargs)`: the pool of a brick wall environment object;
/// `obs` is `"pixels"` (the default) or `"state"`; `max_episode_steps`
/// defaults to 4000 decisions.
pub(crate) fn make_brickwall(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &["outputs", "frameskip", "lives", "obs"]])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, 4000)?;
    let (frameskip, lives) = (kw.u32(g, "frameskip")?, kw.u32(g, "lives")?);
    let obs = kw.string(g, "obs")?;
    let ops = g
        .detach(|| crate::env::brickwall(&args, frameskip, lives, obs.as_deref()))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// `make_snake(**kwargs)`: the pool of a snake environment object; `obs` is
/// `"pixels"` (the default) or `"state"`; `max_episode_steps` defaults to 1000.
pub(crate) fn make_snake(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &["outputs", "size", "agents", "obs"]])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, 1000)?;
    let (size, agents) = (kw.u32(g, "size")?, kw.u32(g, "agents")?);
    let obs = kw.string(g, "obs")?;
    let ops = g
        .detach(|| crate::env::snake(&args, size, agents, obs.as_deref()))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

const RTS_KEYS: [&str; 7] = [
    "size",
    "decision_ticks",
    "max_units",
    "step_limit",
    "seat_swap",
    "opponent",
    "outputs",
];

/// `make_rts(**kwargs)`: the pool of an RTS environment object
/// (`tessera/RTS-v1`). Every `RtsConfig` field by name; the episode ends at
/// the environment's own `step_limit` (in decisions), so
/// `max_episode_steps` defaults to 0 (no extra limit).
pub(crate) fn make_rts(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &RTS_KEYS])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, 0)?;
    let fields = crate::env::RtsFields {
        size: kw.u32(g, "size")?,
        decision_ticks: kw.u32(g, "decision_ticks")?,
        max_units: kw.u32(g, "max_units")?,
        step_limit: kw.u32(g, "step_limit")?,
        seat_swap: kw.bool(g, "seat_swap")?,
        opponent: kw.u32(g, "opponent")?,
    };
    let ops = g
        .detach(|| crate::env::rts(&args, &fields))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

const TAG_KEYS: [&str; 7] = [
    "width", "height", "pursuers", "evaders", "slip", "radius", "outputs",
];

/// `make_tag(**kwargs)`: the pool of a Tag environment object.
/// `max_episode_steps` defaults to 200.
pub(crate) fn make_tag(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &TAG_KEYS])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, 200)?;
    let fields = TagFields {
        width: kw.u32(g, "width")?,
        height: kw.u32(g, "height")?,
        pursuers: kw.u32(g, "pursuers")?,
        evaders: kw.u32(g, "evaders")?,
        slip: kw.u32(g, "slip")?,
        radius: kw.u32(g, "radius")?,
    };
    let ops = g
        .detach(|| crate::env::tag(&args, &fields))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// `make_connect4(**kwargs)`: the pool of a Connect Four environment
/// object. A game lasts at most 42 plies; `max_episode_steps` defaults to
/// 64, so it never truncates a game.
pub(crate) fn make_connect4(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &["outputs"]])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, 64)?;
    let ops = g
        .detach(|| crate::env::connect4(&args))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// `make_leduc(**kwargs)`: the pool of a Leduc hold'em environment
/// object. A hand lasts at most 8 actions; `max_episode_steps` defaults to
/// 16, so it never truncates a hand.
pub(crate) fn make_leduc(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &["outputs"]])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, 16)?;
    let ops = g
        .detach(|| crate::env::leduc(&args))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// `make_othello(**kwargs)`: the pool of an Othello environment object. A
/// game lasts at most 119 plies; `max_episode_steps` defaults to 128, so it
/// never truncates a game.
pub(crate) fn make_othello(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &["outputs"]])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, 128)?;
    let ops = g
        .detach(|| crate::env::othello(&args))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// `make_go9(**kwargs)`: the pool of a Go 9x9 environment object. A game
/// lasts at most 162 plies; `max_episode_steps` defaults to 200, so it never
/// truncates a game.
pub(crate) fn make_go9(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Inner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&POOL_KEYS, &["outputs", "komi_half_points"]])?;
    let borrowed = borrowed(g, &kw)?;
    let args = pool_args(g, &kw, 200)?;
    let komi = kw.i32(g, "komi_half_points")?;
    let ops = g
        .detach(|| crate::env::go9(&args, komi))
        .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, borrowed))
}

pub(crate) fn int(g: Gil<'_>, v: usize) -> CResult<Mine<'_>> {
    capi::int_u64(g, u64::try_from(v).unwrap_or(u64::MAX))
}

pub(crate) fn list_of<'py>(
    g: Gil<'py>,
    items: impl IntoIterator<Item = CResult<Mine<'py>>>,
) -> CResult<Mine<'py>> {
    let list = capi::list(g)?;
    for item in items {
        capi::list_append(g, &list, &item?)?;
    }
    Ok(list)
}

/// `layout()`: the shapes and dtypes the Python layer needs.
pub(crate) fn layout<'py>(g: Gil<'py>, inner: &Inner) -> CResult<Mine<'py>> {
    inner.with(g, |ops| {
        let l = ops.layout();
        let d = capi::dict(g)?;
        capi::dict_set(g, &d, c"num_envs", &int(g, l.num_envs)?)?;
        capi::dict_set(g, &d, c"agents", &int(g, l.agents)?)?;
        capi::dict_set(g, &d, c"initial_agents", &int(g, l.initial_agents)?)?;
        capi::dict_set(
            g,
            &d,
            c"action_choices",
            &capi::int_u64(g, l.action_choices.into())?,
        )?;
        let nvec = list_of(g, l.action_nvec.iter().map(|&n| capi::int_u64(g, n.into())))?;
        capi::dict_set(g, &d, c"action_nvec", &nvec)?;
        let bounds = match l.action_bounds {
            Some((low, high)) => capi::tuple(g, vec![capi::float(g, low)?, capi::float(g, high)?])?,
            None => capi::none(g),
        };
        capi::dict_set(g, &d, c"action_bounds", &bounds)?;
        capi::dict_set(g, &d, c"mask_len", &int(g, l.mask_len)?)?;
        let heads = list_of(
            g,
            l.mask_heads.iter().map(|(offset, parents, size)| {
                capi::tuple(
                    g,
                    vec![
                        int(g, *offset)?,
                        list_of(g, parents.iter().map(|&p| int(g, p)))?,
                        int(g, *size)?,
                    ],
                )
            }),
        )?;
        capi::dict_set(g, &d, c"mask_heads", &heads)?;
        capi::dict_set(g, &d, c"entity_group_heads", &int(g, l.entity_group_heads)?)?;
        capi::dict_set(g, &d, c"turns", &capi::string(g, l.turns)?)?;
        let mode = if inner.borrowed() {
            "borrowed"
        } else {
            "owned"
        };
        capi::dict_set(g, &d, c"outputs", &capi::string(g, mode)?)?;
        let leaves = list_of(
            g,
            l.leaves.iter().map(|leaf| {
                capi::tuple(
                    g,
                    vec![
                        capi::string(g, &leaf.name)?,
                        capi::string(g, leaf.dtype)?,
                        list_of(g, leaf.shape.iter().map(|&s| capi::int_u64(g, s.into())))?,
                        list_of(g, leaf.low.iter().map(|&v| capi::float(g, v)))?,
                        list_of(g, leaf.high.iter().map(|&v| capi::float(g, v)))?,
                    ],
                )
            }),
        )?;
        capi::dict_set(g, &d, c"terminal", &capi::boolean(g, l.terminal))?;
        capi::dict_set(g, &d, c"leaves", &leaves)?;
        if let Some(split) = ops.split() {
            let halves = list_of(
                g,
                split
                    .halves()
                    .iter()
                    .map(|r| capi::tuple(g, vec![int(g, r.start)?, int(g, r.end)?])),
            )?;
            capi::dict_set(g, &d, c"halves", &halves)?;
        }
        Ok(d)
    })
}

/// The positional arguments of a method, from an exact tuple.
fn positional<'a>(
    g: Gil,
    args: Ref<'a>,
    name: &str,
    at_most: usize,
) -> CResult<Vec<Option<Ref<'a>>>> {
    let items = capi::tuple_items(g, args, "the arguments")?;
    if items.len() > at_most {
        return Err(error(
            g,
            Kind::Argument,
            format!("{name} takes at most {at_most} arguments"),
        ));
    }
    let mut out: Vec<Option<Ref<'a>>> = items
        .into_iter()
        .map(|v| (!v.is_none()).then_some(v))
        .collect();
    out.resize(at_most, None);
    Ok(out)
}

pub(crate) fn exactly<'a, const N: usize>(
    g: Gil,
    args: Ref<'a>,
    name: &str,
) -> CResult<[Ref<'a>; N]> {
    let items = capi::tuple_items(g, args, "the arguments")?;
    items.try_into().map_err(|_| {
        error(
            g,
            Kind::Argument,
            format!("{name} takes exactly {N} argument(s)"),
        )
    })
}

pub(crate) fn u32_list(g: Gil, v: Ref<'_>, what: &str) -> CResult<Vec<u32>> {
    capi::list_items(g, v, what)?
        .into_iter()
        .map(|item| capi::to_u32(g, item.as_ref(), what))
        .collect()
}

/// `reset(seed=None, ids=None)`: resets every instance, or only `ids`, and
/// returns the outputs, captured in the same admitted section. `seed` is
/// `None` (keep the seeds), an int (the k-th reset instance gets
/// `seed + k`) or a list with one seed (or `None`) per reset instance.
pub(crate) fn reset<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a = positional(g, args, "reset", 3)?;
    let seed = match a.first().copied().flatten() {
        None => ResetSeed::Keep,
        Some(v) if capi::is_list(v) => ResetSeed::List(
            capi::list_items(g, v, "seed")?
                .into_iter()
                .map(|s| {
                    (!s.as_ref().is_none())
                        .then(|| capi::to_u64(g, s.as_ref(), "a seed"))
                        .transpose()
                })
                .collect::<CResult<_>>()?,
        ),
        Some(v) => ResetSeed::Base(capi::to_u64(g, v, "seed")?),
    };
    let ids = a
        .get(1)
        .copied()
        .flatten()
        .map(|v| u32_list(g, v, "ids"))
        .transpose()?;
    if let Some(ops) = batched_of(g, inner)? {
        return crate::async_api::reset(
            g,
            inner,
            &*ops,
            ids.as_deref(),
            &seed,
            a.get(2).copied().flatten(),
        );
    }
    if let Some(split) = split_of(g, inner)? {
        if ids.is_some() {
            return Err(to_py(
                g,
                BindError::Engine(EngineError::WrongMode {
                    call: "reset with ids",
                    mode: "pipelined",
                }),
            ));
        }
        let owned = !inner.borrowed();
        let halves = g
            .detach(|| split.reset(&seed, owned))
            .map_err(|e| to_py(g, e))?;
        return list_of(
            g,
            halves.iter().map(|out| half_output(g, inner, &*split, out)),
        );
    }
    inner.with(g, |ops| {
        g.detach(|| ops.reset(ids.as_deref(), &seed))
            .map_err(|e| to_py(g, e))?;
        outputs_of(g, inner, ops)
    })
}

/// Whether a call takes an actions `memoryview` (only `step`, #152 E2) or
/// exact `bytes` only (the split-phase `send`s).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Views {
    Allowed,
    BytesOnly,
}

/// Little-endian int32 actions, from exact `bytes`, or (`Views::Allowed`,
/// discrete head) from an exact `memoryview` of C-contiguous native int32,
/// copied by CPython and validated on the copy (#152 E2: the Python layer
/// passes an `<i4` array's view without a copy of its own).
fn actions_of(g: Gil, inner: &Inner, actions: Ref<'_>, views: Views) -> CResult<Wire> {
    let continuous = inner.with(g, |ops| Ok(ops.layout().action_bounds.is_some()))?;
    if views == Views::Allowed && !continuous && capi::is_memoryview(actions) {
        let copy = capi::memoryview_copy(g, actions, "actions")?;
        return int32_view(&copy)
            .map(Wire::Discrete)
            .map_err(|m| error(g, Kind::Action, m.to_owned()));
    }
    let raw = capi::bytes_of(g, actions, "actions")?;
    if raw.len() % 4 != 0 {
        let what = if continuous { "float32" } else { "int32" };
        return Err(error(
            g,
            Kind::Action,
            format!("actions must be {what} values"),
        ));
    }
    let words = raw.as_chunks::<4>().0.iter();
    Ok(if continuous {
        Wire::Continuous(words.map(|c| f32::from_le_bytes(*c)).collect())
    } else {
        Wire::Discrete(words.map(|c| i32::from_le_bytes(*c)).collect())
    })
}

/// The int32 values of a buffer copy: C-contiguous, 4-byte items, and a
/// native or little-endian 4-byte signed format (`i`, or `l` at 4 bytes);
/// the message of the `ActionError` otherwise.
fn int32_view(v: &capi::BufferCopy) -> Result<Vec<i32>, &'static str> {
    const FORMATS: [&[u8]; 6] = [b"i", b"=i", b"<i", b"l", b"=l", b"<l"];
    if !v.c_contiguous {
        return Err("an actions memoryview must be C-contiguous");
    }
    if v.itemsize != 4 || !FORMATS.contains(&v.format.as_slice()) {
        return Err("an actions memoryview must hold int32 values");
    }
    // Two monomorphic loops, not a chosen function pointer, so each one can
    // be inlined and vectorized (#152 E2).
    let words = v.bytes.as_chunks::<4>().0.iter();
    Ok(if v.format.first() == Some(&b'<') {
        words.map(|c| i32::from_le_bytes(*c)).collect()
    } else {
        words.map(|c| i32::from_ne_bytes(*c)).collect()
    })
}

/// One call's decoded actions, typed by the pool's action head.
enum Wire {
    Discrete(Vec<i32>),
    Continuous(Vec<f32>),
}

impl Wire {
    fn input(&self) -> ActionInput<'_> {
        match self {
            Self::Discrete(discrete) => ActionInput {
                discrete,
                continuous: &[],
            },
            Self::Continuous(continuous) => ActionInput {
                discrete: &[],
                continuous,
            },
        }
    }
}

/// The split-phase operations of `inner`, cloned so the guard is not held
/// while they run.
fn batched_of(g: Gil, inner: &Inner) -> CResult<Option<Arc<dyn AsyncOps>>> {
    inner.with(g, |ops| Ok(ops.batched()))
}

/// The split-phase operations of `inner`, cloned as above.
fn split_of(g: Gil, inner: &Inner) -> CResult<Option<Arc<dyn SplitOps>>> {
    inner.with(g, |ops| Ok(ops.split()))
}

/// As [`split_of`]; a closed environment has none (`close` is idempotent).
fn split_of_open(g: Gil, inner: &Inner) -> CResult<Option<Arc<dyn SplitOps>>> {
    inner.peek(g, |ops| ops.and_then(PoolOps::split))
}

fn require_split(g: Gil, inner: &Inner, call: &'static str) -> CResult<Arc<dyn SplitOps>> {
    split_of(g, inner)?.ok_or_else(|| {
        to_py(
            g,
            BindError::Engine(EngineError::WrongMode { call, mode: "sync" }),
        )
    })
}

/// One half's outputs: `{half, start, end, outputs}`.
fn half_output<'py>(
    g: Gil<'py>,
    inner: &Inner,
    split: &dyn SplitOps,
    out: &HalfOutput,
) -> CResult<Mine<'py>> {
    let rows = split
        .halves()
        .get(out.half)
        .cloned()
        .ok_or_else(|| error(g, Kind::Argument, "no such half".to_owned()))?;
    let outputs = if let Some(owned) = &out.owned {
        owned_dict(g, owned)?
    } else {
        let (set, storage) = inner.with(g, |ops| Ok((ops.half_views(out.half), ops.storage())))?;
        let set = set.ok_or_else(no_format(g))?;
        let generations = split.half_generations();
        let unsettled = split.unsettled_flag();
        views_dict(g, &set, |spec| {
            new_view(
                g,
                &storage,
                spec,
                Track::Half(&generations, out.half, out.generation, &unsettled),
            )
        })?
    };
    let d = capi::dict(g)?;
    capi::dict_set(g, &d, c"half", &int(g, out.half)?)?;
    capi::dict_set(g, &d, c"seq", &capi::int_u64(g, out.seq)?)?;
    capi::dict_set(g, &d, c"start", &int(g, rows.start)?)?;
    capi::dict_set(g, &d, c"end", &int(g, rows.end)?)?;
    capi::dict_set(g, &d, c"outputs", &outputs)?;
    Ok(d)
}

/// `send(half, actions: bytes, delivered)`: starts the next step of `half`;
/// `actions` are little-endian int32 (float32 for a continuous head), one per agent row of the half.
/// `delivered` is the last `seq` the caller holds.
pub(crate) fn send<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 3] = exactly(g, args, "send")?;
    let [half, actions, delivered] = a;
    let delivered = capi::to_u64(g, delivered, "delivered")?;
    if let Some(ops) = batched_of(g, inner)? {
        let ids = u32_list(g, half, "ids")?;
        let actions = actions_of(g, inner, actions, Views::BytesOnly)?;
        g.detach(|| ops.send(&ids, actions.input(), delivered))
            .map_err(|e| to_py(g, e))?;
        return Ok(capi::none(g));
    }
    let half = capi::to_u32(g, half, "half")?;
    let actions = actions_of(g, inner, actions, Views::BytesOnly)?;
    let split = require_split(g, inner, "send")?;
    g.detach(|| split.send(half, actions.input(), delivered))
        .map_err(|e| to_py(g, e))?;
    Ok(capi::none(g))
}

/// Ends the one running `recv` on drop.
pub(crate) struct Receiving<'a>(pub(crate) &'a AtomicBool);

impl Drop for Receiving<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

pub(crate) const CONCURRENT_RECV: &str =
    "ConcurrentRecv: another recv on this environment is running";

/// `snapshot_all()`: an opaque whole-pool snapshot, for `restore_all` on
/// this same environment. No work may be in flight.
pub(crate) fn snapshot_all<'py>(g: Gil<'py>, inner: &Inner) -> CResult<Mine<'py>> {
    if let Some(ops) = batched_of(g, inner)? {
        let taken = g.detach(|| ops.snapshot_all()).map_err(|e| to_py(g, e))?;
        return snapshot_type::new_object(g, taken);
    }
    let split = require_split(g, inner, "snapshot_all")?;
    let taken = g.detach(|| split.snapshot_all()).map_err(|e| to_py(g, e))?;
    snapshot_type::new_object(g, taken)
}

/// `restore_all(snapshot)`: restores every instance and the cycle state and
/// returns both halves, as `reset`.
pub(crate) fn restore_all<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 1] = exactly(g, args, "restore_all")?;
    let [snapshot] = a;
    let snapshot = snapshot_type::snapshot_of(g, snapshot)?;
    if let Some(ops) = batched_of(g, inner)? {
        g.detach(|| ops.restore_all(&snapshot))
            .map_err(|e| to_py(g, e))?;
        return Ok(capi::none(g));
    }
    let split = require_split(g, inner, "restore_all")?;
    let owned = !inner.borrowed();
    let halves = g
        .detach(|| split.restore_all(&snapshot, owned))
        .map_err(|e| to_py(g, e))?;
    list_of(
        g,
        halves.iter().map(|out| half_output(g, inner, &*split, out)),
    )
}

/// How long one wait of `recv` lasts with the interpreter released.
pub(crate) const RECV_WAIT: Duration = Duration::from_millis(50);

/// `recv(delivered)`: waits for the oldest sent half and returns its outputs. It
/// waits in bounded steps, outside the mutex and with the interpreter
/// released, and runs signal handlers between them: an exception they raise
/// leaves the step in flight, and a later `recv` returns it. One `recv` at a
/// time: a second one raises `ConcurrencyError` at once. The result repeats
/// until `delivered` reaches its `seq`, so a signal raised after this returns
/// loses nothing.
pub(crate) fn recv<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 1] = exactly(g, args, "recv")?;
    let [delivered] = a;
    let delivered = capi::to_u64(g, delivered, "delivered")?;
    if let Some(ops) = batched_of(g, inner)? {
        return crate::async_api::recv(g, inner, &*ops, delivered);
    }
    let split = require_split(g, inner, "recv")?;
    if split
        .receiving()
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return Err(error(g, Kind::Concurrency, CONCURRENT_RECV.to_owned()));
    }
    let _receiving = Receiving(split.receiving());
    let owned = !inner.borrowed();
    loop {
        let polled = g
            .detach(|| split.try_recv(owned, delivered))
            .map_err(|e| to_py(g, e))?;
        match polled {
            Polled::Half(out) => return half_output(g, inner, &*split, &out),
            Polled::Busy => return Err(error(g, Kind::Concurrency, CONCURRENT_RECV.to_owned())),
            Polled::Running(half) => {
                let _ = g.detach(|| split.waiter().wait_half(half, RECV_WAIT));
            }
        }
        capi::check_signals(g)?;
    }
}

/// `step(actions: bytes)`: one step; `actions` are little-endian int32,
/// one per agent row. Returns the outputs of this step, captured in the same
/// admitted section, so no other call can interleave between the mutation
/// and the copy.
pub(crate) fn step<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 1] = exactly(g, args, "step")?;
    let [actions] = a;
    let actions = actions_of(g, inner, actions, Views::Allowed)?;
    inner.with(g, |ops| {
        g.detach(|| ops.step(actions.input()))
            .map_err(|e| to_py(g, e))?;
        outputs_of(g, inner, ops)
    })
}

/// `outputs()`: the last call's outputs, as borrowed views or owned
/// copies (the environment's `outputs` mode).
pub(crate) fn outputs<'py>(g: Gil<'py>, inner: &Inner) -> CResult<Mine<'py>> {
    if let Some(ops) = batched_of(g, inner)? {
        drop(ops);
        return Err(to_py(
            g,
            BindError::Engine(EngineError::WrongMode {
                call: "outputs",
                mode: "async",
            }),
        ));
    }
    if split_of(g, inner)?.is_some() {
        return Err(to_py(
            g,
            BindError::Engine(EngineError::WrongMode {
                call: "outputs",
                mode: "pipelined",
            }),
        ));
    }
    inner.with(g, |ops| outputs_of(g, inner, ops))
}

/// The outputs dict, built inside the admitted call that produced them.
fn outputs_of<'py>(g: Gil<'py>, inner: &Inner, ops: &dyn PoolOps) -> CResult<Mine<'py>> {
    if inner.borrowed() {
        return borrowed_views(g, ops);
    }
    owned_dict(g, &ops.outputs().map_err(|e| to_py(g, e))?)
}

/// Owned copies as a dict of `bytearray`s.
pub(crate) fn owned_dict<'py>(g: Gil<'py>, o: &Owned) -> CResult<Mine<'py>> {
    let d = capi::dict(g)?;
    let obs = list_of(
        g,
        o.observations.iter().map(|leaf| capi::bytearray(g, leaf)),
    )?;
    capi::dict_set(g, &d, c"observations", &obs)?;
    let term = list_of(
        g,
        o.terminal_observations
            .iter()
            .map(|leaf| capi::bytearray(g, leaf)),
    )?;
    capi::dict_set(g, &d, c"terminal_observations", &term)?;
    for (key, buf) in [
        (c"rewards", &o.rewards),
        (c"agent_terminated", &o.agent_terminated),
        (c"agent_truncated", &o.agent_truncated),
        (c"alive", &o.alive),
        (c"acting", &o.acting),
        (c"slot_generations", &o.slot_generations),
        (c"episode_terminated", &o.episode_terminated),
        (c"episode_truncated", &o.episode_truncated),
        (c"masks", &o.masks),
    ] {
        capi::dict_set(g, &d, key, &capi::bytearray(g, buf)?)?;
    }
    Ok(d)
}

pub(crate) fn no_format(g: Gil) -> impl FnOnce() -> Raised {
    move || {
        error(
            g,
            Kind::Argument,
            "an observation dtype has no buffer format".to_owned(),
        )
    }
}

/// Borrowed views of every output (ADR-0003): read-only, valid until the
/// next call that writes outputs.
fn borrowed_views<'py>(g: Gil<'py>, ops: &dyn PoolOps) -> CResult<Mine<'py>> {
    let set = ops.views().ok_or_else(no_format(g))?;
    let storage = ops.readable_storage().map_err(|e| to_py(g, e))?;
    views_dict(g, set, |spec| new_view(g, &storage, spec, Track::Storage))
}

/// The views of `set`, as `view` builds each.
pub(crate) fn views_dict<'py>(
    g: Gil<'py>,
    set: &ViewSet,
    view: impl Fn(&ViewSpec) -> CResult<Mine<'py>>,
) -> CResult<Mine<'py>> {
    let d = capi::dict(g)?;
    let obs = list_of(g, set.observations.iter().map(&view))?;
    capi::dict_set(g, &d, c"observations", &obs)?;
    let term = list_of(g, set.terminal_observations.iter().map(&view))?;
    capi::dict_set(g, &d, c"terminal_observations", &term)?;
    for (key, spec) in &set.others {
        let key =
            std::ffi::CString::new(*key).map_err(|_| error(g, Kind::Argument, "key".to_owned()))?;
        capi::dict_set(g, &d, &key, &view(spec)?)?;
    }
    Ok(d)
}

/// `snapshot(id) -> bytes`: the canonical encoding of instance `id`.
pub(crate) fn snapshot<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 1] = exactly(g, args, "snapshot")?;
    let [id] = a;
    let id = capi::to_u32(g, id, "id")?;
    if let Some(ops) = batched_of(g, inner)? {
        // Async: an opaque record for `restore`, as `snapshot_all`.
        let taken = g.detach(|| ops.snapshot(id)).map_err(|e| to_py(g, e))?;
        return snapshot_type::new_object(g, taken);
    }
    inner.with(g, |ops| {
        let bytes = ops.snapshot(id).map_err(|e| to_py(g, e))?;
        capi::bytes(g, &bytes)
    })
}

/// `illegal_actions(id) -> list[int]`: instance `id`'s illegal-but-valid
/// actions replaced in its current episode, per agent slot (sync mode).
pub(crate) fn illegal_actions<'py>(
    g: Gil<'py>,
    inner: &Inner,
    args: Ref<'_>,
) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 1] = exactly(g, args, "illegal_actions")?;
    let [id] = a;
    let id = capi::to_u32(g, id, "id")?;
    inner.with(g, |ops| {
        let counts = ops.illegal_actions(id).map_err(|e| to_py(g, e))?;
        list_of(g, counts.iter().map(|&c| capi::int_u64(g, c.into())))
    })
}

/// `state_hash(id) -> int`: instance `id`'s state hash (sync mode).
pub(crate) fn state_hash<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 1] = exactly(g, args, "state_hash")?;
    let [id] = a;
    let id = capi::to_u32(g, id, "id")?;
    inner.with(g, |ops| {
        let hash = ops.state_hash(id).map_err(|e| to_py(g, e))?;
        capi::int_u64(g, hash)
    })
}

/// `import_record(id, record: bytes)`: imports an untrusted record.
pub(crate) fn import_record<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 2] = exactly(g, args, "import_record")?;
    let [id, record] = a;
    let id = capi::to_u32(g, id, "id")?;
    let record = capi::bytes_of(g, record, "record")?.to_vec();
    inner.with(g, |ops| {
        g.detach(|| ops.import(id, &record))
            .map_err(|e| to_py(g, e))?;
        Ok(capi::none(g))
    })
}

/// `encode(ids: list[int]) -> bytes`: the persisted file of `ids` (ADR-0011).
pub(crate) fn encode<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 1] = exactly(g, args, "encode")?;
    let [ids] = a;
    let ids = u32_list(g, ids, "ids")?;
    let file = if let Some(ops) = batched_of(g, inner)? {
        g.detach(|| ops.encode(&ids)).map_err(|e| to_py(g, e))?
    } else if let Some(split) = split_of(g, inner)? {
        g.detach(|| split.encode(&ids)).map_err(|e| to_py(g, e))?
    } else {
        inner.with(g, |ops| {
            g.detach(move || ops.encode(&ids)).map_err(|e| to_py(g, e))
        })?
    };
    capi::bytes(g, &file)
}

/// `load(data: bytes, ids, max_instances, max_decoded_bytes)`: loads a persisted
/// file. `data` must be exactly `bytes`: it is immutable, and the argument tuple
/// keeps it alive while the interpreter is released.
pub(crate) fn load<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 4] = exactly(g, args, "load")?;
    let [data, ids, max_instances, max_decoded_bytes] = a;
    let data = capi::bytes_of(g, data, "data")?;
    let mut limits = tessera_rl_engine::DecodeLimits::default();
    if !max_instances.is_none() {
        limits.max_instances = capi::to_u32(g, max_instances, "max_instances")?;
    }
    if !max_decoded_bytes.is_none() {
        limits.max_decoded_bytes = capi::to_u64(g, max_decoded_bytes, "max_decoded_bytes")?;
    }
    let ids = if ids.is_none() {
        None
    } else {
        Some(u32_list(g, ids, "ids")?)
    };
    let batched = batched_of(g, inner)?;
    let split = split_of(g, inner)?;
    if ids.is_some() && (batched.is_some() || split.is_some()) {
        return Err(to_py(
            g,
            BindError::Argument("load with ids is available only in sync mode".to_owned()),
        ));
    }
    if let Some(ops) = batched {
        g.detach(|| ops.load(data, limits))
            .map_err(|e| to_py(g, e))?;
        return Ok(capi::none(g));
    }
    if let Some(split) = split {
        let owned = !inner.borrowed();
        let halves = g
            .detach(|| split.load(data, limits, owned))
            .map_err(|e| to_py(g, e))?;
        return list_of(
            g,
            halves.iter().map(|out| half_output(g, inner, &*split, out)),
        );
    }
    inner.with(g, |ops| {
        g.detach(|| ops.load(data, ids.as_deref(), limits))
            .map_err(|e| to_py(g, e))?;
        Ok(capi::none(g))
    })
}

/// `fork(src, dst: list[int])`: copies instance `src` into `dst`.
pub(crate) fn fork<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 2] = exactly(g, args, "fork")?;
    let [src, dst] = a;
    let src = capi::to_u32(g, src, "src")?;
    let dst = u32_list(g, dst, "dst")?;
    inner.with(g, |ops| {
        g.detach(|| ops.fork(src, &dst)).map_err(|e| to_py(g, e))?;
        Ok(capi::none(g))
    })
}

/// `is_poisoned() -> bool`.
pub(crate) fn is_poisoned<'py>(g: Gil<'py>, inner: &Inner) -> CResult<Mine<'py>> {
    if let Some(ops) = batched_of(g, inner)? {
        let poisoned = g.detach(|| ops.is_poisoned());
        return Ok(capi::boolean(g, poisoned));
    }
    if let Some(split) = split_of(g, inner)? {
        // Detached and outside the guard: this takes the binding mutex.
        let poisoned = g.detach(|| split.is_poisoned());
        return Ok(capi::boolean(g, poisoned));
    }
    inner.with(g, |ops| Ok(capi::boolean(g, ops.is_poisoned())))
}

/// `close()`: frees the pool; idempotent. It takes the reentrancy guard
/// like every other call, so it cannot free a pool another thread is using.
pub(crate) fn close<'py>(g: Gil<'py>, inner: &Inner) -> CResult<Mine<'py>> {
    if let Some(ops) = inner.peek(g, |ops| ops.and_then(PoolOps::batched))? {
        g.detach(|| ops.close());
    }
    if let Some(split) = split_of_open(g, inner)? {
        g.detach(|| split.close());
    }
    let ops = inner.take(g)?;
    g.detach(move || drop(ops));
    Ok(capi::none(g))
}

#[cfg(test)]
mod tests {
    use super::{capi::BufferCopy, int32_view};

    fn view(bytes: &[u8], itemsize: usize, format: &[u8], c_contiguous: bool) -> BufferCopy {
        BufferCopy {
            bytes: bytes.to_vec(),
            itemsize,
            format: format.to_vec(),
            c_contiguous,
        }
    }

    #[test]
    fn test_int32_view_native_and_little_endian_formats_decode() {
        let raw: Vec<u8> = [7_i32, -2, i32::MAX, i32::MIN]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        for format in [&b"i"[..], b"=i", b"<i", b"l", b"=l", b"<l"] {
            assert_eq!(
                int32_view(&view(&raw, 4, format, true)),
                Ok(vec![7, -2, i32::MAX, i32::MIN]),
                "{format:?}"
            );
        }
    }

    #[test]
    fn test_int32_view_empty_buffer_decodes_to_nothing() {
        assert_eq!(int32_view(&view(&[], 4, b"i", true)), Ok(vec![]));
    }

    #[test]
    fn test_int32_view_wrong_item_or_format_is_refused() {
        let raw = [0_u8; 8];
        for (itemsize, format) in [
            (8, &b"q"[..]),
            (8, b"l"),
            (4, b"I"),
            (4, b"f"),
            (4, b">i"),
            (4, b"!i"),
            (1, b"B"),
            (2, b"h"),
        ] {
            assert_eq!(
                int32_view(&view(&raw, itemsize, format, true)),
                Err("an actions memoryview must hold int32 values"),
                "{itemsize} {format:?}"
            );
        }
    }

    #[test]
    fn test_int32_view_not_c_contiguous_is_refused_first() {
        assert_eq!(
            int32_view(&view(&[], 8, b"q", false)),
            Err("an actions memoryview must be C-contiguous")
        );
    }
}
