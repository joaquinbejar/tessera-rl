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
/// `("ok", events, calls, poisoned, unverified_poison)`, or
/// `("diverged", event_index, kind, instance, last_matching_checkpoint)`,
/// which `tessera_rl.replay` raises as `ReplayDivergence`. Any other
/// refusal raises here (`PersistError` for the file, the engine's class
/// for a pool error).
pub(crate) fn replay<'py>(g: Gil<'py>, inner: &Inner, args: Ref<'_>) -> CResult<Mine<'py>> {
    let a: [Ref<'_>; 1] = exactly(g, args, "_replay")?;
    let [data] = a;
    let data = capi::bytes_of(g, data, "data")?;
    let result = inner.with(g, |ops| Ok(g.detach(|| ops.replay(data))))?;
    match result {
        Ok(s) => capi::tuple(
            g,
            vec![
                capi::string(g, "ok")?,
                capi::int_u64(g, s.events)?,
                capi::int_u64(g, s.calls)?,
                capi::boolean(g, s.poisoned),
                capi::boolean(g, s.unverified_poison),
            ],
        ),
        Err(BindError::Replay(ReplayError::Diverged {
            event_index,
            kind,
            instance,
            last_matching_checkpoint,
        })) => capi::tuple(
            g,
            vec![
                capi::string(g, "diverged")?,
                capi::int_u64(g, event_index)?,
                capi::string(g, kind_name(kind))?,
                optional(g, instance.map(u64::from))?,
                optional(g, last_matching_checkpoint)?,
            ],
        ),
        Err(e) => Err(to_py(g, e)),
    }
}
