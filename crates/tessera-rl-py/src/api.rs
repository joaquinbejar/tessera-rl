// SPDX-License-Identifier: MIT OR Apache-2.0

//! The implementation of every Python-visible operation, over `ffi::capi`
//! (ADR-0041 callouts). Arguments are read only from exact built-in types;
//! the Python layer converts anything else before calling.

use crate::env::{PoolArgs, PoolOps, ResetSeed, TagFields};
use crate::errors::{Kind, error, to_py};
use crate::ffi::capi::{self, CResult, Gil, Mine, Ref};
use crate::ffi::inner::Inner;
use crate::ffi::view_type::new_view;

/// Keyword arguments, as exact `(str, value)` pairs.
struct Kwargs<'a>(Vec<(String, Ref<'a>)>);

impl<'a> Kwargs<'a> {
    fn parse(g: Gil, kwargs: Option<Ref<'a>>) -> CResult<Self> {
        Ok(Self(match kwargs {
            Some(d) => capi::dict_items(g, d, "keyword arguments")?,
            None => Vec::new(),
        }))
    }

    /// The value of `key`, absent or `None` as `None`.
    fn get(&self, key: &str) -> Option<Ref<'a>> {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| *v)
            .filter(|v| !v.is_none())
    }

    fn check(&self, g: Gil, allowed: &[&[&str]]) -> CResult<()> {
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

    fn u32(&self, g: Gil, key: &str) -> CResult<Option<u32>> {
        self.get(key).map(|v| capi::to_u32(g, v, key)).transpose()
    }

    fn u64(&self, g: Gil, key: &str) -> CResult<Option<u64>> {
        self.get(key).map(|v| capi::to_u64(g, v, key)).transpose()
    }

    fn string(&self, g: Gil, key: &str) -> CResult<Option<String>> {
        self.get(key)
            .map(|v| capi::to_string(g, v, key))
            .transpose()
    }
}

const POOL_KEYS: [&str; 6] = [
    "num_envs",
    "seed",
    "max_episode_steps",
    "auto_reset",
    "chunk_size",
    "workers",
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

fn pool_args(g: Gil, kw: &Kwargs<'_>, default_limit: u32) -> CResult<PoolArgs> {
    Ok(PoolArgs {
        num_envs: kw.u32(g, "num_envs")?.unwrap_or(1),
        seed: kw.u64(g, "seed")?.unwrap_or(0),
        max_episode_steps: Some(kw.u32(g, "max_episode_steps")?.unwrap_or(default_limit)),
        auto_reset: kw
            .string(g, "auto_reset")?
            .unwrap_or_else(|| "next_step".to_owned()),
        chunk_size: kw.u32(g, "chunk_size")?,
        workers: kw
            .u32(g, "workers")?
            .map_or(0, |w| usize::try_from(w).unwrap_or(usize::MAX)),
    })
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

fn int(g: Gil, v: usize) -> CResult<Mine> {
    capi::int_u64(g, u64::try_from(v).unwrap_or(u64::MAX))
}

fn list_of(g: Gil, items: impl IntoIterator<Item = CResult<Mine>>) -> CResult<Mine> {
    let list = capi::list(g)?;
    for item in items {
        capi::list_append(g, &list, &item?)?;
    }
    Ok(list)
}

/// `layout()`: the shapes and dtypes the Python layer needs.
pub(crate) fn layout(g: Gil, inner: &Inner) -> CResult<Mine> {
    inner.with(g, |ops| {
        let l = ops.layout();
        let d = capi::dict(g)?;
        capi::dict_set(g, &d, c"num_envs", &int(g, l.num_envs)?)?;
        capi::dict_set(g, &d, c"agents", &int(g, l.agents)?)?;
        capi::dict_set(
            g,
            &d,
            c"action_choices",
            &capi::int_u64(g, l.action_choices.into())?,
        )?;
        capi::dict_set(g, &d, c"mask_len", &int(g, l.mask_len)?)?;
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

fn exactly<'a, const N: usize>(g: Gil, args: Ref<'a>, name: &str) -> CResult<[Ref<'a>; N]> {
    let items = capi::tuple_items(g, args, "the arguments")?;
    items.try_into().map_err(|_| {
        error(
            g,
            Kind::Argument,
            format!("{name} takes exactly {N} argument(s)"),
        )
    })
}

fn u32_list(g: Gil, v: Ref<'_>, what: &str) -> CResult<Vec<u32>> {
    capi::list_items(g, v, what)?
        .into_iter()
        .map(|item| capi::to_u32(g, item, what))
        .collect()
}

/// `reset(seed=None, ids=None)`: resets every instance, or only `ids`, and
/// returns the outputs, captured in the same admitted section. `seed` is
/// `None` (keep the seeds), an int (the k-th reset instance gets
/// `seed + k`) or a list with one seed (or `None`) per reset instance.
pub(crate) fn reset(g: Gil, inner: &Inner, args: Ref<'_>) -> CResult<Mine> {
    let a = positional(g, args, "reset", 2)?;
    let seed = match a.first().copied().flatten() {
        None => ResetSeed::Keep,
        Some(v) if capi::is_list(v) => ResetSeed::List(
            capi::list_items(g, v, "seed")?
                .into_iter()
                .map(|s| {
                    (!s.is_none())
                        .then(|| capi::to_u64(g, s, "a seed"))
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
    inner.with(g, |ops| {
        g.detach(|| ops.reset(ids.as_deref(), &seed))
            .map_err(|e| to_py(g, e))?;
        outputs_of(g, inner, ops)
    })
}

/// `step(actions: bytes)`: one step; `actions` are little-endian int32,
/// one per agent row. Returns the outputs of this step, captured in the same
/// admitted section, so no other call can interleave between the mutation
/// and the copy.
pub(crate) fn step(g: Gil, inner: &Inner, args: Ref<'_>) -> CResult<Mine> {
    let a: [Ref<'_>; 1] = exactly(g, args, "step")?;
    let [actions] = a;
    let raw = capi::bytes_of(g, actions, "actions")?;
    if raw.len() % 4 != 0 {
        return Err(error(
            g,
            Kind::Action,
            "actions must be int32 values".to_owned(),
        ));
    }
    let actions: Vec<i32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect();
    inner.with(g, |ops| {
        g.detach(|| ops.step(&actions)).map_err(|e| to_py(g, e))?;
        outputs_of(g, inner, ops)
    })
}

/// `outputs()`: the last call's outputs, as borrowed views or owned
/// copies (the environment's `outputs` mode).
pub(crate) fn outputs(g: Gil, inner: &Inner) -> CResult<Mine> {
    inner.with(g, |ops| outputs_of(g, inner, ops))
}

/// The outputs dict, built inside the admitted call that produced them.
fn outputs_of(g: Gil, inner: &Inner, ops: &dyn PoolOps) -> CResult<Mine> {
    if inner.borrowed() {
        return borrowed_views(g, ops);
    }
    let o = ops.outputs();
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
        (c"episode_terminated", &o.episode_terminated),
        (c"episode_truncated", &o.episode_truncated),
        (c"masks", &o.masks),
    ] {
        capi::dict_set(g, &d, key, &capi::bytearray(g, buf)?)?;
    }
    Ok(d)
}

/// Borrowed views of every output (ADR-0003): read-only, valid until the
/// next call that writes outputs.
fn borrowed_views(g: Gil, ops: &dyn PoolOps) -> CResult<Mine> {
    let Some(set) = crate::view::view_set(ops.layout()) else {
        return Err(error(
            g,
            Kind::Argument,
            "an observation dtype has no buffer format".to_owned(),
        ));
    };
    let storage = ops.storage();
    let d = capi::dict(g)?;
    let obs = list_of(
        g,
        set.observations
            .iter()
            .map(|spec| new_view(g, &storage, spec)),
    )?;
    capi::dict_set(g, &d, c"observations", &obs)?;
    let term = list_of(
        g,
        set.terminal_observations
            .iter()
            .map(|spec| new_view(g, &storage, spec)),
    )?;
    capi::dict_set(g, &d, c"terminal_observations", &term)?;
    for (key, spec) in &set.others {
        let key =
            std::ffi::CString::new(*key).map_err(|_| error(g, Kind::Argument, "key".to_owned()))?;
        capi::dict_set(g, &d, &key, &new_view(g, &storage, spec)?)?;
    }
    Ok(d)
}

/// `snapshot(id) -> bytes`: the canonical encoding of instance `id`.
pub(crate) fn snapshot(g: Gil, inner: &Inner, args: Ref<'_>) -> CResult<Mine> {
    let a: [Ref<'_>; 1] = exactly(g, args, "snapshot")?;
    let [id] = a;
    let id = capi::to_u32(g, id, "id")?;
    inner.with(g, |ops| {
        let bytes = ops.snapshot(id).map_err(|e| to_py(g, e))?;
        capi::bytes(g, &bytes)
    })
}

/// `import_record(id, record: bytes)`: imports an untrusted record.
pub(crate) fn import_record(g: Gil, inner: &Inner, args: Ref<'_>) -> CResult<Mine> {
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

/// `fork(src, dst: list[int])`: copies instance `src` into `dst`.
pub(crate) fn fork(g: Gil, inner: &Inner, args: Ref<'_>) -> CResult<Mine> {
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
pub(crate) fn is_poisoned(g: Gil, inner: &Inner) -> CResult<Mine> {
    inner.with(g, |ops| Ok(capi::boolean(g, ops.is_poisoned())))
}

/// `close()`: frees the pool; idempotent. It takes the reentrancy guard
/// like every other call, so it cannot free a pool another thread is using.
pub(crate) fn close(g: Gil, inner: &Inner) -> CResult<Mine> {
    let ops = inner.take(g)?;
    g.detach(move || drop(ops));
    Ok(capi::none(g))
}
