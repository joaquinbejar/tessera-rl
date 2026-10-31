// SPDX-License-Identifier: MIT OR Apache-2.0

//! The safe implementation of every Python-visible operation. The C entry
//! points in `ffi` only translate pointers and call these functions.

use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyTuple};

use crate::env::{PoolArgs, TagFields};
use crate::errors::{Kind, error, to_py};
use crate::ffi::inner::Inner;

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
    check_keys(py, kwargs, &[])?;
    let args = pool_args(kwargs, tessera_rl_classic::cartpole::MAX_EPISODE_STEPS)?;
    let ops = py
        .detach(|| crate::env::cartpole(&args))
        .map_err(|e| to_py(py, e))?;
    Ok(Inner::new(ops))
}

const TAG_KEYS: [&str; 6] = ["width", "height", "pursuers", "evaders", "slip", "radius"];

/// `make_tag(**kwargs)`: the pool of a Tag environment object.
/// `max_episode_steps` defaults to 200.
pub(crate) fn make_tag(py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Inner> {
    check_keys(py, kwargs, &TAG_KEYS)?;
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
    Ok(Inner::new(ops))
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
        let leaves = PyList::empty(py);
        for leaf in &l.leaves {
            leaves.append((leaf.name.as_str(), leaf.dtype, leaf.shape.clone()))?;
        }
        d.set_item("leaves", leaves)?;
        Ok(d.into_any())
    })
}

/// `reset(seed=None)`: resets every instance (a new seed base if given).
pub(crate) fn reset<'py>(
    py: Python<'py>,
    inner: &Inner,
    args: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let seed: Option<u64> = match args.len() {
        0 => None,
        1 => args.get_item(0)?.extract()?,
        _ => {
            return Err(error(
                py,
                Kind::Argument,
                "reset takes at most one argument".to_owned(),
            ));
        }
    };
    inner.with(py, |ops| {
        py.detach(|| ops.reset(seed)).map_err(|e| to_py(py, e))?;
        Ok(py.None().into_bound(py))
    })
}

/// `step(actions: bytes)`: one step; `actions` are little-endian int32,
/// one per agent row.
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
        Ok(py.None().into_bound(py))
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

/// `outputs()`: copies of the last call's outputs (`owned` mode), as fresh
/// bytearrays the caller owns (NumPy wraps them writable, without copying).
pub(crate) fn outputs<'py>(py: Python<'py>, inner: &Inner) -> PyResult<Bound<'py, PyAny>> {
    inner.with(py, |ops| {
        let o = ops.outputs();
        let d = PyDict::new(py);
        let obs = PyList::empty(py);
        for leaf in &o.observations {
            obs.append(PyByteArray::new(py, leaf))?;
        }
        d.set_item("observations", obs)?;
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
    })
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
