// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Async operations of an environment object, over `ffi::capi`.

use core::sync::atomic::Ordering;

use crate::api::{
    CONCURRENT_RECV, RECV_WAIT, Receiving, int, list_of, no_format, owned_dict, u32_list,
    views_dict,
};
use crate::batched::{AsyncOps, BatchOutput, BatchPolled};
use crate::env::ResetSeed;
use crate::errors::{Kind, error, to_py};
use crate::ffi::capi::{self, CResult, Gil, Mine, Ref};
use crate::ffi::inner::Inner;
use crate::ffi::snapshot_type;
use crate::ffi::view_type::{Track, new_gather_view, new_view};
use crate::view::{half_view_set, view_set};

/// A batch: `{ids, seq, outputs}`. Gathered and owned outputs have one row
/// per id in `ids`, in that order; indexed views have all N rows.
fn batch_output(g: Gil, inner: &Inner, ops: &dyn AsyncOps, out: &BatchOutput) -> CResult<Mine> {
    let poisoned = ops.poison_flag();
    let outputs = if let Some(owned) = &out.owned {
        owned_dict(g, owned)?
    } else if let Some(buffer) = ops.gather() {
        let rows = 0..out.ids.len();
        let set = inner.with(g, |p| Ok(half_view_set(p.layout(), &rows)))?;
        let set = set.ok_or_else(no_format(g))?;
        views_dict(g, &set, |spec| {
            new_gather_view(g, &buffer, spec, out.gathered, &poisoned)
        })?
    } else {
        let set = inner.with(g, |p| Ok(view_set(p.layout())))?;
        let set = set.ok_or_else(no_format(g))?;
        let (storage, generations, epochs) =
            (ops.storage(), ops.id_generations(), ops.release_epochs());
        views_dict(g, &set, |spec| {
            new_view(
                g,
                &storage,
                spec,
                Track::Ids(&generations, &out.ids, &epochs, &poisoned),
            )
        })?
    };
    let ids = list_of(g, out.ids.iter().map(|b| capi::int_u64(g, b.id.into())))?;
    let d = capi::dict(g)?;
    capi::dict_set(g, &d, c"ids", &ids)?;
    capi::dict_set(
        g,
        &d,
        c"seq",
        &int(g, usize::try_from(out.seq).unwrap_or(usize::MAX))?,
    )?;
    capi::dict_set(g, &d, c"outputs", &outputs)?;
    Ok(d)
}

/// `reset(seed, ids, delivered)`: every instance, returning them as one
/// batch; or `ids`, which the next `recv` delivers (returns `None`).
pub(crate) fn reset(
    g: Gil,
    inner: &Inner,
    ops: &dyn AsyncOps,
    ids: Option<&[u32]>,
    seed: &ResetSeed,
    delivered: Option<Ref<'_>>,
) -> CResult<Mine> {
    let delivered = delivered.map_or(Ok(0), |d| capi::to_u64(g, d, "delivered"))?;
    if let Some(ids) = ids {
        g.detach(|| ops.reset(ids, seed, delivered))
            .map_err(|e| to_py(g, e))?;
        return Ok(capi::none(g));
    }
    let out = g.detach(|| ops.reset_all(seed)).map_err(|e| to_py(g, e))?;
    batch_output(g, inner, ops, &out)
}

/// `recv`: as Pipelined, waiting on the engine's batch waiter, which polls
/// in 100 microsecond steps.
pub(crate) fn recv(g: Gil, inner: &Inner, ops: &dyn AsyncOps, delivered: u64) -> CResult<Mine> {
    if ops
        .receiving()
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return Err(error(g, Kind::Concurrency, CONCURRENT_RECV.to_owned()));
    }
    let _receiving = Receiving(ops.receiving());
    loop {
        let polled = g
            .detach(|| ops.try_recv(delivered))
            .map_err(|e| to_py(g, e))?;
        match polled {
            BatchPolled::Batch(out) => return batch_output(g, inner, ops, &out),
            BatchPolled::Busy => {
                return Err(error(g, Kind::Concurrency, CONCURRENT_RECV.to_owned()));
            }
            BatchPolled::Running => {
                let _ = g.detach(|| ops.waiter().wait_ready(RECV_WAIT));
            }
        }
        capi::check_signals(g)?;
    }
}

/// `restore(ids, snapshots, delivered)`: restores `ids` from one-instance
/// snapshots; the next `recv` delivers them.
pub(crate) fn restore(g: Gil, inner: &Inner, args: Ref<'_>) -> CResult<Mine> {
    let items = capi::tuple_items(g, args, "the arguments")?;
    let [ids, snapshots, delivered] = items.as_slice() else {
        return Err(error(
            g,
            Kind::Argument,
            "restore takes (ids, snapshots, delivered)".to_owned(),
        ));
    };
    let ids = u32_list(g, *ids, "ids")?;
    let delivered = capi::to_u64(g, *delivered, "delivered")?;
    let snapshots = capi::list_items(g, *snapshots, "snapshots")?
        .into_iter()
        .map(|s| snapshot_type::snapshot_of(g, s))
        .collect::<CResult<Vec<_>>>()?;
    let ops = inner.with(g, |p| Ok(p.batched()))?.ok_or_else(|| {
        error(
            g,
            Kind::Argument,
            "restore is available only in async mode".to_owned(),
        )
    })?;
    g.detach(|| ops.restore(&ids, &snapshots, delivered))
        .map_err(|e| to_py(g, e))?;
    Ok(capi::none(g))
}
