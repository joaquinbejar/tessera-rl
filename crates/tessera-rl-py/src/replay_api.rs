// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replay files from Python (ADR-0072 section 9): the native half of
//! `tessera_rl.replay`. Recording is held in memory and returned as bytes;
//! the Python side writes the file. First-party environments in Inline or
//! Sync mode only.

use tessera_rl_engine::ReplayError;
use tessera_rl_engine::replay::DivergenceKind;

use crate::api::exactly;
use crate::env::BindError;
use crate::errors::to_py;
use crate::ffi::capi::{self, CResult, Gil, Mine, Ref};
use crate::ffi::inner::Inner;

/// `_record_start(keyframe_every, checkpoint_every)`: starts recording.
pub(crate) fn record_start<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 2] = exactly(g, args, "_record_start")?;
    let [keyframe_every, checkpoint_every] = a;
    let keyframe_every = capi::to_u64(g, keyframe_every, "keyframe_every")?;
    let checkpoint_every = capi::to_u64(g, checkpoint_every, "checkpoint_every")?;
    inner.with(g, |ops| {
        g.detach(|| ops.record_start(keyframe_every, checkpoint_every))
            .map_err(|e| to_py(g, e))?;
        Ok(capi::none(g))
    })
}

/// `_record_finish() -> bytes`: ends the recording and returns the file.
pub(crate) fn record_finish<'py>(g: Gil<'py>, inner: &Inner) -> CResult<Mine<'py>> {
    let file = inner.with(g, |ops| {
        g.detach(|| ops.record_finish()).map_err(|e| to_py(g, e))
    })?;
    capi::bytes(g, &file)
}

/// `_record_abort()`: drops the recording, if any.
pub(crate) fn record_abort<'py>(g: Gil<'py>, inner: &Inner) -> CResult<Mine<'py>> {
    inner.with(g, |ops| {
        ops.record_abort();
        Ok(capi::none(g))
    })
}

/// The Python name of a divergence kind.
const fn kind_name(kind: DivergenceKind) -> &'static str {
    match kind {
        DivergenceKind::CallFailed => "call_failed",
        DivergenceKind::UnexpectedPoison => "unexpected_poison",
        DivergenceKind::NotPoisoned => "not_poisoned",
        DivergenceKind::Checkpoint => "checkpoint",
        DivergenceKind::Keyframe => "keyframe",
        DivergenceKind::BranchCounter => "branch_counter",
        DivergenceKind::KeyframeRejected => "keyframe_rejected",
        DivergenceKind::RejectionMismatch => "rejection_mismatch",
        DivergenceKind::RefusalMismatch => "refusal_mismatch",
        DivergenceKind::OutputsMismatch => "outputs_mismatch",
        DivergenceKind::RecvMismatch => "recv_mismatch",
    }
}

fn optional(g: Gil<'_>, v: Option<u64>) -> CResult<Mine<'_>> {
    v.map_or_else(|| Ok(capi::none(g)), |v| capi::int_u64(g, v))
}

/// `_replay(data: bytes) -> tuple`: replays a file into this pool.
/// `("ok", events, calls, poisoned, unverified_poison, unclaimed)`, or
/// `("diverged", event_index, kind, instance, last_matching_checkpoint)`,
/// which `tessera_rl.replay` raises as `ReplayDivergence`. Any other
/// refusal raises here (`PersistError` for the file, the engine's class
/// for a pool error).
pub(crate) fn replay<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 1] = exactly(g, args, "_replay")?;
    let [data] = a;
    let data = capi::bytes_of(g, data, "data")?;
    let result = inner.with(g, |ops| Ok(g.detach(|| ops.replay(data))))?;
    outcome(g, result, false)
}

/// `_replay_begin(data: bytes)`: starts a stepped replay (issue #378); the
/// pool refuses calls that change it until `_replay_finish`.
pub(crate) fn replay_begin<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 1] = exactly(g, args, "_replay_begin")?;
    let [data] = a;
    let data = capi::bytes_of(g, data, "data")?;
    inner.with(g, |ops| {
        g.detach(|| ops.replay_begin(data))
            .map_err(|e| to_py(g, e))?;
        Ok(capi::none(g))
    })
}

/// `_replay_advance(calls: int) -> tuple`: replays until `calls` more calls
/// were applied. `("ok", event_index)` for the last event applied, `("ok",
/// None)` at the end of the file, or the `"diverged"` tuple of `_replay`.
pub(crate) fn replay_advance<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 1] = exactly(g, args, "_replay_advance")?;
    let [calls] = a;
    let calls = capi::to_u64(g, calls, "calls")?;
    let result = inner.with(g, |ops| Ok(g.detach(|| ops.replay_advance(calls))))?;
    match result {
        Ok(point) => capi::tuple(g, vec![capi::string(g, "ok")?, optional(g, point)?]),
        Err(e) => diverged_or_raise(g, e),
    }
}

/// `_replay_seek(keyframe: int)`: imports the keyframe at that event and
/// moves the replay after it.
pub(crate) fn replay_seek<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 1] = exactly(g, args, "_replay_seek")?;
    let [keyframe] = a;
    let keyframe = capi::to_u64(g, keyframe, "keyframe")?;
    inner.with(g, |ops| {
        g.detach(|| ops.replay_seek(keyframe))
            .map_err(|e| to_py(g, e))?;
        Ok(capi::none(g))
    })
}

/// `_replay_keyframes() -> tuple[int, ...]`: the keyframes `_replay_seek`
/// accepts.
pub(crate) fn replay_keyframes<'py>(g: Gil<'py>, inner: &Inner) -> CResult<Mine<'py>> {
    // A scan of the file's frame index: cheap, so the interpreter stays
    // attached (the pool is borrowed shared here, and `dyn PoolOps` is not
    // `Sync`).
    let keyframes = inner.with(g, |ops| ops.replay_keyframes().map_err(|e| to_py(g, e)))?;
    capi::tuple(
        g,
        keyframes
            .iter()
            .map(|&k| capi::int_u64(g, k))
            .collect::<CResult<Vec<_>>>()?,
    )
}

/// `_replay_finish() -> tuple`: ends the stepped replay. The tuples of
/// `_replay`, the `"ok"` one with a trailing `stopped_at` (`None` when every
/// event was applied and checked, else the next event's index).
pub(crate) fn replay_finish<'py>(g: Gil<'py>, inner: &Inner) -> CResult<Mine<'py>> {
    let result = inner.with(g, |ops| Ok(g.detach(|| ops.replay_finish())))?;
    outcome(g, result, true)
}

/// A summary as the `"ok"` tuple (with `stopped_at` when `stopped` is set),
/// or a divergence as the `"diverged"` tuple; any other error raises.
fn outcome(
    g: Gil<'_>,
    result: Result<tessera_rl_engine::ReplaySummary, BindError>,
    stopped: bool,
) -> CResult<Mine<'_>> {
    match result {
        Ok(s) => {
            let mut items = vec![
                capi::string(g, "ok")?,
                capi::int_u64(g, s.events)?,
                capi::int_u64(g, s.calls)?,
                capi::boolean(g, s.poisoned),
                capi::boolean(g, s.unverified_poison),
                capi::tuple(
                    g,
                    s.unclaimed
                        .iter()
                        .map(|&id| capi::int_u64(g, u64::from(id)))
                        .collect::<CResult<Vec<_>>>()?,
                )?,
            ];
            if stopped {
                items.push(optional(g, s.stopped_at)?);
                items.push(optional(g, s.started_at)?);
            }
            capi::tuple(g, items)
        }
        Err(e) => diverged_or_raise(g, e),
    }
}

/// A divergence as the `"diverged"` tuple; any other error raises.
fn diverged_or_raise(g: Gil<'_>, e: BindError) -> CResult<Mine<'_>> {
    match e {
        BindError::Replay(ReplayError::Diverged {
            event_index,
            kind,
            instance,
            last_matching_checkpoint,
        }) => capi::tuple(
            g,
            vec![
                capi::string(g, "diverged")?,
                capi::int_u64(g, event_index)?,
                capi::string(g, kind_name(kind))?,
                optional(g, instance.map(u64::from))?,
                optional(g, last_matching_checkpoint)?,
            ],
        ),
        e => Err(to_py(g, e)),
    }
}
