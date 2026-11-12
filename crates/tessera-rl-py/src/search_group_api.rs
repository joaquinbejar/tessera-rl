// SPDX-License-Identifier: MIT OR Apache-2.0

//! The entry points of `tessera_rl._native.SearchGroup` (ADR-0045): the
//! slots of independent searches whose leaves share one evaluator call
//! per round. The evaluator bridge is the `Search` one (`search_api`).
//!
//! The whole object is guarded for the whole call (`Inner`): a nested or
//! concurrent `run`, and any other method during a `run`, including on a
//! slot that does not take part, raises `ConcurrencyError`.
//!
//! A failure's exception is raised without being touched: the user's own
//! exception (the evaluator's, or a signal handler's) is raised again as
//! it is, and the bindings never set attributes on any exception. The
//! slot of the last failure is read with `last_failure()`.

use crate::api::Kwargs;
use crate::env::BindError;
use crate::errors::{Kind, error, to_py};
use crate::ffi::capi::{self, CResult, Gil, Mine, Raised, Ref};
use crate::ffi::inner::Inner;
use crate::search::{GroupOps, SearchArgs};
use crate::search_api::{PyEval, SEARCH_KEYS, apply_options, floats, ints, one_arg};

/// The state behind a `SearchGroup` object.
pub(crate) type GroupInner = Inner<dyn GroupOps>;

const GROUP_KEYS: [&str; 2] = ["trees", "slot_options"];

/// `make_search_group(trees=T, slot_options=None, **search_options)`:
/// `T` searches, each with the shared options updated by its entry of
/// `slot_options` (a list of `T` dicts of search options), if given.
pub(crate) fn make_search_group(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<GroupInner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&SEARCH_KEYS, &GROUP_KEYS])?;
    let trees = kw
        .u32(g, "trees")?
        .ok_or_else(|| error(g, Kind::Argument, "trees is required".to_owned()))?;
    let trees = usize::try_from(trees).unwrap_or(usize::MAX);
    if trees == 0 || trees > tessera_rl_search::MAX_TREES {
        return Err(error(
            g,
            Kind::Argument,
            format!("trees must be in 1..={}", tessera_rl_search::MAX_TREES),
        ));
    }
    let mut shared = SearchArgs::new();
    apply_options(g, &kw, &mut shared)?;
    let mut slots = vec![shared; trees];
    if let Some(list) = kw.get("slot_options").filter(|v| !v.is_none()) {
        let items = capi::list_items(g, list, "slot_options")?;
        if items.len() != trees {
            return Err(error(
                g,
                Kind::Argument,
                format!("slot_options has {} entries for {trees} trees", items.len()),
            ));
        }
        for (args, item) in slots.iter_mut().zip(items) {
            let one = Kwargs::parse(g, Some(item))?;
            one.check(g, &[&SEARCH_KEYS])?;
            if one.get("env_id").is_some() {
                return Err(error(
                    g,
                    Kind::Argument,
                    "env_id is shared by every slot".to_owned(),
                ));
            }
            apply_options(g, &one, args)?;
        }
    }
    let env_id = kw
        .string(g, "env_id")?
        .unwrap_or_else(|| "tessera/Connect4-v1".to_owned());
    let ops = match env_id.as_str() {
        "tessera/Connect4-v1" => crate::search::connect4_group(&slots),
        other => Err(BindError::Argument(format!(
            "no search for {other:?}; known: [\"tessera/Connect4-v1\"]"
        ))),
    }
    .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, false))
}

/// The slot index argument `i` of `signature`.
fn slot(g: Gil, v: Ref<'_>) -> CResult<usize> {
    let i = capi::to_u64(g, v, "slot")?;
    usize::try_from(i).map_err(|_| error(g, Kind::Argument, format!("slot {i} out of range")))
}

/// `layout()`: `{"trees": int, "actions": int, "obs_bytes": int, "memory_bytes": int}`.
pub(crate) fn layout(g: Gil, inner: &GroupInner) -> CResult<Mine> {
    inner.with(g, |ops| {
        let d = capi::dict(g)?;
        let n = |v: usize| u64::try_from(v).unwrap_or(u64::MAX);
        capi::dict_set(g, &d, c"trees", &capi::int_u64(g, n(ops.trees()))?)?;
        capi::dict_set(g, &d, c"actions", &capi::int_u64(g, ops.actions().into())?)?;
        capi::dict_set(g, &d, c"obs_bytes", &capi::int_u64(g, n(ops.obs_bytes()))?)?;
        capi::dict_set(
            g,
            &d,
            c"memory_bytes",
            &capi::int_u64(g, ops.memory_bytes())?,
        )?;
        Ok(d)
    })
}

/// The two positional arguments of `signature`.
fn two_args<'a>(g: Gil, args: Ref<'a>, signature: &str) -> CResult<[Ref<'a>; 2]> {
    let items = capi::tuple_items(g, args, signature)?;
    <[Ref<'a>; 2]>::try_from(items)
        .map_err(|_| error(g, Kind::Argument, format!("expected {signature}")))
}

/// `set_root(i, record: bytes)`.
pub(crate) fn set_root(g: Gil, inner: &GroupInner, args: Ref<'_>) -> CResult<Mine> {
    let [i, record] = two_args(g, args, "set_root(i, record)")?;
    let i = slot(g, i)?;
    let bytes = capi::bytes_of(g, record, "record")?;
    inner.with(g, |ops| ops.set_root(i, bytes).map_err(|e| to_py(g, e)))?;
    Ok(capi::none(g))
}

/// `set_active(i, flag: bool)`.
pub(crate) fn set_active(g: Gil, inner: &GroupInner, args: Ref<'_>) -> CResult<Mine> {
    let [i, flag] = two_args(g, args, "set_active(i, flag)")?;
    let i = slot(g, i)?;
    let flag = capi::to_bool(g, flag, "flag")?;
    inner.with(g, |ops| ops.set_active(i, flag).map_err(|e| to_py(g, e)))?;
    Ok(capi::none(g))
}

/// `active(i) -> bool`.
pub(crate) fn active(g: Gil, inner: &GroupInner, args: Ref<'_>) -> CResult<Mine> {
    let [i] = one_arg(g, args, "active(i)")?;
    let i = slot(g, i)?;
    let a = inner.with(g, |ops| ops.active(i).map_err(|e| to_py(g, e)))?;
    Ok(capi::boolean(g, a))
}

/// `run(evaluator)`: one group call.
pub(crate) fn run(g: Gil, inner: &GroupInner, args: Ref<'_>) -> CResult<Mine> {
    let [callable] = one_arg(g, args, "run(evaluator)")?;
    let mut eval = PyEval {
        g,
        callable,
        pending: None,
    };
    let result = inner.with(g, |ops| {
        let r = ops.run(&mut eval);
        ops.set_user_raised(r.is_err() && eval.pending.is_some());
        Ok(r)
    })?;
    match (result, eval.pending.take()) {
        (Ok(()), _) => Ok(capi::none(g)),
        // The user's own exception, raised again untouched, every slot
        // rolled back.
        (Err(_), Some(pending)) => {
            capi::restore_pending(pending);
            Err(Raised)
        }
        (Err((_, e)), None) => Err(to_py(g, e)),
    }
}

/// `last_failure() -> None | (slot: int | None, message: str, user_raised: bool)`.
pub(crate) fn last_failure(g: Gil, inner: &GroupInner) -> CResult<Mine> {
    let f = inner.with(g, |ops| Ok(ops.last_failure()))?;
    let Some(f) = f else {
        return Ok(capi::none(g));
    };
    let slot = match f.slot {
        Some(s) => capi::int_u64(g, u64::try_from(s).unwrap_or(u64::MAX))?,
        None => capi::none(g),
    };
    capi::tuple(
        g,
        vec![
            slot,
            capi::string(g, &f.message)?,
            capi::boolean(g, f.user_raised),
        ],
    )
}

/// `root_visits(i) -> list[int]`.
pub(crate) fn root_visits(g: Gil, inner: &GroupInner, args: Ref<'_>) -> CResult<Mine> {
    let [i] = one_arg(g, args, "root_visits(i)")?;
    let i = slot(g, i)?;
    inner.with(g, |ops| {
        let visits = ops.root_visits(i).map_err(|e| to_py(g, e))?;
        let list = capi::list(g)?;
        for v in visits {
            capi::list_append(g, &list, &capi::int_u64(g, v.into())?)?;
        }
        Ok(list)
    })
}

/// `select_action(i, temperature: float) -> int`.
pub(crate) fn select_action(g: Gil, inner: &GroupInner, args: Ref<'_>) -> CResult<Mine> {
    let [i, t] = two_args(g, args, "select_action(i, temperature)")?;
    let i = slot(g, i)?;
    let t = capi::to_f64(g, t, "temperature")?;
    let a = inner.with(g, |ops| ops.select_action(i, t).map_err(|e| to_py(g, e)))?;
    capi::int_u64(g, a.into())
}

/// `improved_policy(i) -> list[float] | None`.
pub(crate) fn improved_policy(g: Gil, inner: &GroupInner, args: Ref<'_>) -> CResult<Mine> {
    let [i] = one_arg(g, args, "improved_policy(i)")?;
    let i = slot(g, i)?;
    inner.with(g, |ops| {
        let v = ops.improved_policy(i).map_err(|e| to_py(g, e))?;
        floats(g, v)
    })
}

/// `considered(i) -> list[int] | None`.
pub(crate) fn considered(g: Gil, inner: &GroupInner, args: Ref<'_>) -> CResult<Mine> {
    let [i] = one_arg(g, args, "considered(i)")?;
    let i = slot(g, i)?;
    inner.with(g, |ops| {
        let v = ops.considered(i).map_err(|e| to_py(g, e))?;
        ints(g, v)
    })
}

/// `calls(i) -> int`.
pub(crate) fn calls(g: Gil, inner: &GroupInner, args: Ref<'_>) -> CResult<Mine> {
    let [i] = one_arg(g, args, "calls(i)")?;
    let i = slot(g, i)?;
    let n = inner.with(g, |ops| ops.calls(i).map_err(|e| to_py(g, e)))?;
    capi::int_u64(g, n)
}

/// `tree_hash(i) -> int`.
pub(crate) fn tree_hash(g: Gil, inner: &GroupInner, args: Ref<'_>) -> CResult<Mine> {
    let [i] = one_arg(g, args, "tree_hash(i)")?;
    let i = slot(g, i)?;
    let h = inner.with(g, |ops| ops.tree_hash(i).map_err(|e| to_py(g, e)))?;
    capi::int_u64(g, h)
}
