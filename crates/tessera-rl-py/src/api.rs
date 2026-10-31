// SPDX-License-Identifier: MIT OR Apache-2.0

//! The safe implementation of every Python-visible operation. The C entry
//! points in `ffi` only translate pointers and call these functions.

use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyTuple};

use crate::env::{PoolArgs, PoolOps, ResetSeed, TagFields};
use crate::errors::{Kind, error, to_py};
use crate::ffi::inner::Inner;
use crate::ffi::view_type::new_view;

fn kw<'py, T: FromPyObjectOwned<'py>>(
    kwargs: Option<&Bound<'py, PyDict>>,
    key: &str,
) -> PyResult<Option<T>> {
    match kwargs.map(|d| d.get_item(key)).transpose()?.flatten() {
        Some(v) if !v.is_none() => Ok(Some(v.extract::<T>().map_err(Into::into)?)),
        _ => Ok(None),
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

fn check_keys(py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>, extra: &[&str]) -> PyResult<()> {
    if let Some(d) = kwargs {
        for key in d.keys() {
            let key: String = key.extract()?;
            if !POOL_KEYS.contains(&key.as_str()) && !extra.contains(&key.as_str()) {
                return Err(error(
                    py,
                    Kind::Argument,
                    format!("unexpected argument {key:?}"),
                ));
            }
        }
    }
    Ok(())
}

/// The `outputs` mode: `"borrowed"` (the native default, ADR-0003) or
/// `"owned"`.
fn borrowed(py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
    match kw::<String>(kwargs, "outputs")?.as_deref() {
        None | Some("borrowed") => Ok(true),
        Some("owned") => Ok(false),
        Some(other) => Err(error(
            py,
            Kind::Argument,
            format!("outputs must be \"borrowed\" or \"owned\", not {other:?}"),
        )),
    }
}

fn pool_args(kwargs: Option<&Bound<'_, PyDict>>, default_limit: u32) -> PyResult<PoolArgs> {
    Ok(PoolArgs {
        num_envs: kw(kwargs, "num_envs")?.unwrap_or(1),
        seed: kw(kwargs, "seed")?.unwrap_or(0),
        max_episode_steps: Some(kw(kwargs, "max_episode_steps")?.unwrap_or(default_limit)),
        auto_reset: kw(kwargs, "auto_reset")?.unwrap_or_else(|| "next_step".to_owned()),
        chunk_size: kw(kwargs, "chunk_size")?,
        workers: kw(kwargs, "workers")?.unwrap_or(0),
    })
}

/// `make_cartpole(**kwargs)`: the pool of a CartPole environment object.
/// `max_episode_steps` defaults to 500 (`CartPole-v1`).
pub(crate) fn make_cartpole(py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Inner> {
    check_keys(py, kwargs, &["outputs"])?;
    let borrowed = borrowed(py, kwargs)?;
    let args = pool_args(kwargs, tessera_rl_classic::cartpole::MAX_EPISODE_STEPS)?;
    let ops = py
        .detach(|| crate::env::cartpole(&args))
        .map_err(|e| to_py(py, e))?;
    Ok(Inner::new(ops, borrowed))
}

const TAG_KEYS: [&str; 7] = [
    "width", "height", "pursuers", "evaders", "slip", "radius", "outputs",
];

/// `make_tag(**kwargs)`: the pool of a Tag environment object.
/// `max_episode_steps` defaults to 200.
pub(crate) fn make_tag(py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Inner> {
    check_keys(py, kwargs, &TAG_KEYS)?;
    let borrowed = borrowed(py, kwargs)?;
    let args = pool_args(kwargs, 200)?;
    let fields = TagFields {
        width: kw(kwargs, "width")?,
        height: kw(kwargs, "height")?,
        pursuers: kw(kwargs, "pursuers")?,
        evaders: kw(kwargs, "evaders")?,
        slip: kw(kwargs, "slip")?,
        radius: kw(kwargs, "radius")?,
    };
    let ops = py
        .detach(|| crate::env::tag(&args, &fields))
        .map_err(|e| to_py(py, e))?;
    Ok(Inner::new(ops, borrowed))
}

/// `layout()`: the shapes and dtypes the Python layer needs.
pub(crate) fn layout<'py>(py: Python<'py>, inner: &Inner) -> PyResult<Bound<'py, PyAny>> {
    inner.with(py, |ops| {
        let l = ops.layout();
        let d = PyDict::new(py);
        d.set_item("num_envs", l.num_envs)?;
        d.set_item("agents", l.agents)?;
        d.set_item("action_choices", l.action_choices)?;
        d.set_item("mask_len", l.mask_len)?;
        d.set_item(
            "outputs",
            if inner.borrowed() {
                "borrowed"
            } else {
                "owned"
            },
        )?;
        let leaves = PyList::empty(py);
        for leaf in &l.leaves {
            leaves.append((
                leaf.name.as_str(),
                leaf.dtype,
                leaf.shape.clone(),
                leaf.low.clone(),
                leaf.high.clone(),
            ))?;
        }
        d.set_item("terminal", l.terminal)?;
        d.set_item("leaves", leaves)?;
        Ok(d.into_any())
    })
}

/// `reset(seed=None, ids=None)`: resets every instance, or only `ids`, and
/// returns the outputs, captured in the same admitted section. `seed` is
/// `None` (keep the seeds), an int (the k-th reset instance gets
/// `seed + k`) or a list with one seed per reset instance.
pub(crate) fn reset<'py>(
    py: Python<'py>,
    inner: &Inner,
    args: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    if args.len() > 2 {
        return Err(error(
            py,
            Kind::Argument,
            "reset takes at most two arguments".to_owned(),
        ));
    }
    let arg = |i: usize| -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(if i < args.len() {
            Some(args.get_item(i)?).filter(|v| !v.is_none())
        } else {
            None
        })
    };
    let seed = match arg(0)? {
        None => ResetSeed::Keep,
        Some(v) if v.cast::<PyList>().is_ok() => ResetSeed::List(v.extract()?),
        Some(v) => ResetSeed::Base(v.extract()?),
    };
    let ids: Option<Vec<u32>> = arg(1)?.map(|v| v.extract()).transpose()?;
    inner.with(py, |ops| {
        py.detach(|| ops.reset(ids.as_deref(), &seed))
            .map_err(|e| to_py(py, e))?;
        outputs_of(py, inner, ops)
    })
}

/// `step(actions: bytes)`: one step; `actions` are little-endian int32,
/// one per agent row. Returns the outputs of this step, captured in the same
/// admitted section, so no other call can interleave between the mutation
/// and the copy.
pub(crate) fn step<'py>(
    py: Python<'py>,
    inner: &Inner,
    args: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let bytes = one_arg(py, args, "step")?;
    let bytes = bytes.cast::<PyBytes>().map_err(PyErr::from)?;
    let raw = bytes.as_bytes();
    if raw.len() % 4 != 0 {
        return Err(error(
            py,
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
    inner.with(py, |ops| {
        py.detach(|| ops.step(&actions)).map_err(|e| to_py(py, e))?;
        outputs_of(py, inner, ops)
    })
}

fn one_arg<'py>(
    py: Python<'py>,
    args: &Bound<'py, PyTuple>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if args.len() != 1 {
        return Err(error(
            py,
            Kind::Argument,
            format!("{name} takes exactly one argument"),
        ));
    }
    args.get_item(0)
}

/// `outputs()`: the last call's outputs, as borrowed views or owned
/// copies (the environment's `outputs` mode).
pub(crate) fn outputs<'py>(py: Python<'py>, inner: &Inner) -> PyResult<Bound<'py, PyAny>> {
    inner.with(py, |ops| outputs_of(py, inner, ops))
}

/// The outputs dict, built inside the admitted call that produced them.
fn outputs_of<'py>(
    py: Python<'py>,
    inner: &Inner,
    ops: &dyn PoolOps,
) -> PyResult<Bound<'py, PyAny>> {
    if inner.borrowed() {
        return borrowed_views(py, ops);
    }
    let o = ops.outputs();
    let d = PyDict::new(py);
    let obs = PyList::empty(py);
    for leaf in &o.observations {
        obs.append(PyByteArray::new(py, leaf))?;
    }
    d.set_item("observations", obs)?;
    let term = PyList::empty(py);
    for leaf in &o.terminal_observations {
        term.append(PyByteArray::new(py, leaf))?;
    }
    d.set_item("terminal_observations", term)?;
    for (key, buf) in [
        ("rewards", &o.rewards),
        ("agent_terminated", &o.agent_terminated),
        ("agent_truncated", &o.agent_truncated),
        ("alive", &o.alive),
        ("episode_terminated", &o.episode_terminated),
        ("episode_truncated", &o.episode_truncated),
        ("masks", &o.masks),
    ] {
        d.set_item(key, PyByteArray::new(py, buf))?;
    }
    Ok(d.into_any())
}

/// Borrowed views of every output (ADR-0003): read-only, valid until the
/// next call that writes outputs.
fn borrowed_views<'py>(py: Python<'py>, ops: &dyn PoolOps) -> PyResult<Bound<'py, PyAny>> {
    let set = crate::view::view_set(ops.layout()).ok_or_else(|| {
        error(
            py,
            Kind::Argument,
            "an observation dtype has no buffer format".to_owned(),
        )
    })?;
    let storage = ops.storage();
    let d = PyDict::new(py);
    let obs = PyList::empty(py);
    for spec in &set.observations {
        obs.append(new_view(py, &storage, spec)?)?;
    }
    d.set_item("observations", obs)?;
    let term = PyList::empty(py);
    for spec in &set.terminal_observations {
        term.append(new_view(py, &storage, spec)?)?;
    }
    d.set_item("terminal_observations", term)?;
    for (key, spec) in &set.others {
        d.set_item(*key, new_view(py, &storage, spec)?)?;
    }
    Ok(d.into_any())
}

/// `snapshot(id) -> bytes`: the canonical encoding of instance `id`.
pub(crate) fn snapshot<'py>(
    py: Python<'py>,
    inner: &Inner,
    args: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let id: u32 = one_arg(py, args, "snapshot")?.extract()?;
    inner.with(py, |ops| {
        let bytes = ops.snapshot(id).map_err(|e| to_py(py, e))?;
        Ok(PyBytes::new(py, &bytes).into_any())
    })
}

/// `import_record(id, record: bytes)`: imports an untrusted record.
pub(crate) fn import_record<'py>(
    py: Python<'py>,
    inner: &Inner,
    args: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let (id, record): (u32, Vec<u8>) = args.extract()?;
    inner.with(py, |ops| {
        py.detach(|| ops.import(id, &record))
            .map_err(|e| to_py(py, e))?;
        Ok(py.None().into_bound(py))
    })
}

/// `fork(src, dst: list[int])`: copies instance `src` into `dst`.
pub(crate) fn fork<'py>(
    py: Python<'py>,
    inner: &Inner,
    args: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let (src, dst): (u32, Vec<u32>) = args.extract()?;
    inner.with(py, |ops| {
        py.detach(|| ops.fork(src, &dst))
            .map_err(|e| to_py(py, e))?;
        Ok(py.None().into_bound(py))
    })
}

/// `is_poisoned() -> bool`.
pub(crate) fn is_poisoned<'py>(py: Python<'py>, inner: &Inner) -> PyResult<Bound<'py, PyAny>> {
    inner.with(py, |ops| {
        Ok(pyo3::types::PyBool::new(py, ops.is_poisoned())
            .to_owned()
            .into_any())
    })
}

/// `close()`: frees the pool; idempotent. It takes the reentrancy guard
/// like every other call, so it cannot free a pool another thread is using.
pub(crate) fn close<'py>(py: Python<'py>, inner: &Inner) -> PyResult<Bound<'py, PyAny>> {
    let ops = inner.take(py)?;
    py.detach(move || drop(ops));
    Ok(py.None().into_bound(py))
}
