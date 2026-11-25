// SPDX-License-Identifier: MIT OR Apache-2.0

//! The XLA FFI token path (ADR-0095), experimental: the safe host the
//! handlers of `tessera_rl_pyffi::xla::handler` call, and the Python entry
//! points of `tessera_rl.jax.ffi_handle`.
//!
//! A handle wraps an existing Sync `Env` and shares its reentrancy guard
//! (REVIEW-design-r1, answer 1). Refused at `_xla_open`, with
//! `CapabilityError`: other architectures than x86-64 (no native evidence
//! of the thunk there yet, REVIEW-plan-B), a privileged-state pool, the
//! split-phase modes, and continuous actions.

use tessera_rl_engine::ActionInput;
use tessera_rl_pyffi::xla::handler::{self, XlaEntry, XlaHost};
use tessera_rl_pyffi::xla::registry::{Handle, Registry};
use tessera_rl_pyffi::xla::sys::{
    DTYPE_F32, DTYPE_F64, DTYPE_PRED, DTYPE_S8, DTYPE_S32, DTYPE_S64, DTYPE_U8, DTYPE_U32, DataType,
};

use crate::env::{BindError, PoolOps, ResetSeed};
use crate::errors::{Kind, error};
use crate::ffi::capi::{self, CResult, Gil, Mine, Ref};
use crate::ffi::inner::Inner;

/// The first-party host: the pools behind `Env` objects.
pub(crate) struct NativeXla;

static REGISTRY: Registry<XlaEntry<dyn PoolOps>> = Registry::new();

impl XlaHost for NativeXla {
    type Pool = dyn PoolOps;

    fn registry() -> &'static Registry<XlaEntry<dyn PoolOps>> {
        &REGISTRY
    }

    fn step(
        pool: &mut Self::Pool,
        actions: &[i32],
        outs: &mut [&mut [u8]],
        dtypes: &[DataType],
    ) -> Result<(), (Kind, String)> {
        check_dtypes(pool, dtypes)?;
        pool.step(ActionInput {
            discrete: actions,
            continuous: &[],
        })
        .map_err(kind_of)?;
        pool.outputs_into(outs).map_err(kind_of)
    }

    fn reset(
        pool: &mut Self::Pool,
        seed: Option<u64>,
        outs: &mut [&mut [u8]],
        dtypes: &[DataType],
    ) -> Result<(), (Kind, String)> {
        check_dtypes(pool, dtypes)?;
        let seed = seed.map_or(ResetSeed::Keep, ResetSeed::Base);
        pool.reset(None, &seed).map_err(kind_of)?;
        pool.outputs_into(outs).map_err(kind_of)
    }
}

/// The class and message of `e`, without the interpreter.
fn kind_of(e: BindError) -> (Kind, String) {
    let kind = match &e {
        BindError::Engine(e) => tessera_rl_pyffi::errors::engine_kind(e),
        BindError::Capability(_) => Kind::Capability,
        _ => Kind::Argument,
    };
    let message = match e {
        BindError::Argument(m) | BindError::Capability(m) => m,
        BindError::Engine(e) => e.to_string(),
        other => format!("{other:?}"),
    };
    (kind, message)
}

/// The XLA dtype of a layout leaf dtype.
fn xla_dtype(numpy: &str) -> Option<DataType> {
    Some(match numpy {
        "float32" => DTYPE_F32,
        "float64" => DTYPE_F64,
        "int32" => DTYPE_S32,
        "int64" => DTYPE_S64,
        "int8" => DTYPE_S8,
        "uint8" => DTYPE_U8,
        "uint32" => DTYPE_U32,
        "bool" => DTYPE_PRED,
        _ => return None,
    })
}

/// The result dtypes in `outputs_into`'s order must match the layout:
/// observation leaves, rewards (`f32`), five flags (`bool`), slot
/// generations (`u32`), two episode flags (`bool`) and, with masks, the
/// masks (`bool`).
fn check_dtypes(pool: &dyn PoolOps, dtypes: &[DataType]) -> Result<(), (Kind, String)> {
    let layout = pool.layout();
    let leaves = layout.leaves.len();
    let masks = usize::from(layout.mask_len > 0);
    let want_len = leaves + 8 + masks;
    if dtypes.len() != want_len {
        return Err((
            Kind::Argument,
            format!(
                "the call declares {} results, the layout {want_len}",
                dtypes.len()
            ),
        ));
    }
    let fixed = [
        DTYPE_F32, DTYPE_PRED, DTYPE_PRED, DTYPE_PRED, DTYPE_PRED, DTYPE_U32, DTYPE_PRED,
        DTYPE_PRED,
    ];
    let leaf_ok = layout
        .leaves
        .iter()
        .zip(dtypes)
        .all(|(leaf, &d)| xla_dtype(leaf.dtype) == Some(d));
    let rest = dtypes.get(leaves..).unwrap_or_default();
    let fixed_ok = rest
        .iter()
        .zip(fixed.iter().chain(&[DTYPE_PRED]))
        .all(|(a, b)| a == b);
    if leaf_ok && fixed_ok {
        Ok(())
    } else {
        Err((
            Kind::Argument,
            "a result dtype does not match the layout".to_owned(),
        ))
    }
}

/// `Env._xla_open() -> (slot, generation)`: registers this `Env` as an XLA
/// FFI handle sharing its guard.
pub(crate) fn open<'py>(g: Gil<'py>, inner: &Inner) -> CResult<Mine<'py>> {
    if !cfg!(target_arch = "x86_64") {
        return Err(error(
            g,
            Kind::Capability,
            "the XLA FFI path is experimental and only enabled on x86-64 until it has native evidence on this architecture".to_owned(),
        ));
    }
    let refusal = inner.with(g, |ops| {
        let layout = ops.layout();
        Ok(if !layout.state_leaves.is_empty() {
            Some("the XLA FFI path exports no privileged state: make the Env without privileged_state")
        } else if layout.action_bounds.is_some() {
            Some("the XLA FFI path supports discrete actions only")
        } else if ops.split().is_some() || ops.batched().is_some() {
            Some("the XLA FFI path supports Sync mode only")
        } else {
            None
        })
    })?;
    if let Some(why) = refusal {
        return Err(error(g, Kind::Capability, why.to_owned()));
    }
    let h = handler::open::<NativeXla>(g, inner.shared())?;
    pair(g, h)
}

fn pair(g: Gil<'_>, h: Handle) -> CResult<Mine<'_>> {
    capi::tuple(
        g,
        vec![
            capi::int_u64(g, u64::from(h.slot))?,
            capi::int_u64(g, h.generation)?,
        ],
    )
}

/// The handle `(slot, generation)` of a call's arguments.
fn handle_arg(g: Gil<'_>, args: Ref<'_>) -> CResult<Handle> {
    let items = capi::tuple_items(g, args, "the arguments")?;
    let [slot, generation] = items.as_slice() else {
        return Err(error(
            g,
            Kind::Argument,
            "takes slot and generation".to_owned(),
        ));
    };
    Ok(Handle {
        slot: capi::to_u32(g, *slot, "slot")?,
        generation: capi::to_u64(g, *generation, "generation")?,
    })
}

/// `_xla_targets() -> {"step": capsule, "reset": capsule}`.
pub(crate) fn targets(g: Gil<'_>) -> CResult<Mine<'_>> {
    handler::targets::<NativeXla>(g)
}

/// `_xla_token(slot, generation) -> int`: the next expected token.
pub(crate) fn token<'py>(g: Gil<'py>, args: Ref<'py>) -> CResult<Mine<'py>> {
    let h = handle_arg(g, args)?;
    capi::int_u64(g, u64::from(handler::token::<NativeXla>(g, h)?))
}

/// `_xla_take_error(slot, generation) -> (class name, message) | None`.
pub(crate) fn take_error<'py>(g: Gil<'py>, args: Ref<'py>) -> CResult<Mine<'py>> {
    let h = handle_arg(g, args)?;
    match handler::take_last_error::<NativeXla>(g, h)? {
        None => Ok(capi::none(g)),
        Some((kind, message)) => capi::tuple(
            g,
            vec![
                capi::string(g, tessera_rl_pyffi::errors::kind_name(kind))?,
                capi::string(g, &message)?,
            ],
        ),
    }
}

/// `_xla_close(slot, generation)`: waits for a call in flight, then closes.
pub(crate) fn close<'py>(g: Gil<'py>, args: Ref<'py>) -> CResult<Mine<'py>> {
    let h = handle_arg(g, args)?;
    handler::close::<NativeXla>(g, h);
    Ok(capi::none(g))
}
