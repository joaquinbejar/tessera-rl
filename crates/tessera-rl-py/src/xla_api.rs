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
        // Every check before the pool is touched (REVIEW-B-r1 point 1).
        check_dtypes(pool, dtypes)?;
        check_sizes(pool, Some(actions.len()), outs)?;
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
        check_sizes(pool, None, outs)?;
        let seed = seed.map_or(ResetSeed::Keep, ResetSeed::Base);
        pool.reset(None, &seed).map_err(kind_of)?;
        pool.outputs_into(outs).map_err(kind_of)
    }

    fn poison(pool: &mut Self::Pool) {
        pool.poison();
    }
}

/// Bytes of one element of a layout dtype.
fn dtype_bytes(numpy: &str) -> Option<usize> {
    Some(match numpy {
        "float64" | "int64" | "uint64" => 8,
        "float32" | "int32" | "uint32" => 4,
        "int16" | "uint16" => 2,
        "int8" | "uint8" | "bool" => 1,
        _ => return None,
    })
}

/// The actions' length and every result buffer's byte length must match
/// the layout, in `outputs_into`'s order; checked before the pool is
/// touched, so a refused call leaves it as it was (REVIEW-B-r1 point 1).
fn check_sizes(
    pool: &dyn PoolOps,
    actions: Option<usize>,
    outs: &[&mut [u8]],
) -> Result<(), (Kind, String)> {
    let layout = pool.layout();
    let (n, a) = (layout.num_envs, layout.agents);
    let rows = n
        .checked_mul(a)
        .ok_or_else(|| (Kind::Argument, "layout too large".to_owned()))?;
    if let Some(len) = actions
        && len != rows
    {
        return Err((
            Kind::Action,
            format!("actions must be int32 [{n}, {a}] ({rows} values), got {len}"),
        ));
    }
    let leaf = |l: &crate::env::Leaf| -> Option<usize> {
        let elems = l
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(usize::try_from(d).ok()?))?;
        elems.checked_mul(dtype_bytes(l.dtype)?)?.checked_mul(rows)
    };
    // Compared one buffer at a time, with no collection: this runs on
    // every step and reset, so it must not allocate (REVIEW-B-r2).
    let mismatch = || {
        (
            Kind::Argument,
            "a result buffer's size does not match the layout".to_owned(),
        )
    };
    let mut got = outs.iter().map(|o| o.len());
    let mut expect = |want: Option<usize>| -> Result<(), (Kind, String)> {
        match (want, got.next()) {
            (Some(w), Some(g)) if w == g => Ok(()),
            _ => Err(mismatch()),
        }
    };
    for l in &layout.leaves {
        expect(leaf(l))?;
    }
    let four = rows.checked_mul(4);
    expect(four)?; // rewards, f32
    for _ in 0..4 {
        expect(Some(rows))?; // agent terminated, truncated, alive, acting
    }
    expect(four)?; // slot generations, u32
    expect(Some(n))?; // episode terminated
    expect(Some(n))?; // episode truncated
    if layout.mask_len > 0 {
        expect(rows.checked_mul(layout.mask_len))?;
    }
    if got.next().is_some() {
        return Err(mismatch());
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::{NativeXla, XlaHost};
    use crate::env::{PoolArgs, ResetSeed, cartpole};
    use tessera_rl_pyffi::xla::sys::{DTYPE_F32, DTYPE_PRED, DTYPE_U32};

    fn args() -> PoolArgs {
        PoolArgs {
            num_envs: 2,
            seed: 3,
            max_episode_steps: None,
            auto_reset: "next_step".to_owned(),
            chunk_size: None,
            schedule: "static".to_owned(),
            workers: Some(0),
            spin: None,
            wait: None,
            wait_cap_us: None,
            transforms: tessera_rl_engine::Transforms::new(),
            policy_map: None,
            pipelined: false,
            asynced: None,
            privileged_state: false,
            league: None,
            onnx_opponents: None,
            targets: false,
            output_allocator: None,
        }
    }

    /// REVIEW-B-r1 point 1: a wrong result size or actions length is
    /// refused before the pool is stepped (the state hash is unchanged),
    /// and a correct call then steps once.
    #[test]
    fn test_xla_host_wrong_sizes_refused_before_stepping() {
        let mut pool = cartpole(&args()).unwrap();
        pool.reset(None, &ResetSeed::Keep).unwrap();
        let before: Vec<u64> = (0..2).map(|i| pool.state_hash(i).unwrap()).collect();
        let dtypes = [
            DTYPE_F32, DTYPE_F32, DTYPE_PRED, DTYPE_PRED, DTYPE_PRED, DTYPE_PRED, DTYPE_U32,
            DTYPE_PRED, DTYPE_PRED,
        ];
        let sizes = [2 * 4 * 4, 2 * 4, 2, 2, 2, 2, 2 * 4, 2, 2];
        let make = |extra: usize| -> Vec<Vec<u8>> {
            sizes
                .iter()
                .enumerate()
                .map(|(i, &n)| vec![0; n + usize::from(i == 0) * extra])
                .collect()
        };
        let mut wrong = make(4);
        let mut views: Vec<&mut [u8]> = wrong.iter_mut().map(Vec::as_mut_slice).collect();
        assert!(NativeXla::step(pool.as_mut(), &[1, 1], &mut views, &dtypes).is_err());
        let mut right = make(0);
        let mut views: Vec<&mut [u8]> = right.iter_mut().map(Vec::as_mut_slice).collect();
        assert!(
            NativeXla::step(pool.as_mut(), &[1, 1, 1], &mut views, &dtypes).is_err(),
            "actions length"
        );
        let after: Vec<u64> = (0..2).map(|i| pool.state_hash(i).unwrap()).collect();
        assert_eq!(before, after, "nothing stepped");
        let mut views: Vec<&mut [u8]> = right.iter_mut().map(Vec::as_mut_slice).collect();
        NativeXla::step(pool.as_mut(), &[1, 1], &mut views, &dtypes).unwrap();
        let stepped: Vec<u64> = (0..2).map(|i| pool.state_hash(i).unwrap()).collect();
        assert_ne!(before, stepped, "one step");
    }

    /// REVIEW-B-r2 point 1: the host's checks and step allocate nothing on
    /// the success path (the engine's Inline Sync step allocates nothing,
    /// ADR-0004), measured with the binary's counting allocator.
    #[test]
    fn test_xla_host_step_success_path_does_not_allocate() {
        let mut pool = cartpole(&args()).unwrap();
        pool.reset(None, &ResetSeed::Keep).unwrap();
        let dtypes = [
            DTYPE_F32, DTYPE_F32, DTYPE_PRED, DTYPE_PRED, DTYPE_PRED, DTYPE_PRED, DTYPE_U32,
            DTYPE_PRED, DTYPE_PRED,
        ];
        let sizes = [2 * 4 * 4, 2 * 4, 2, 2, 2, 2, 2 * 4, 2, 2];
        let mut bufs: Vec<Vec<u8>> = sizes.iter().map(|&n| vec![0; n]).collect();
        let mut views: Vec<&mut [u8]> = bufs.iter_mut().map(Vec::as_mut_slice).collect();
        // Warm-up: the first step may touch lazily built state.
        NativeXla::step(pool.as_mut(), &[1, 1], &mut views, &dtypes).unwrap();
        for _ in 0..5 {
            let mut result = Ok(());
            let n = tessera_rl_pyffi::xla::testing::count(|| {
                result = NativeXla::step(pool.as_mut(), &[1, 1], &mut views, &dtypes);
            });
            result.unwrap();
            assert_eq!(
                n, 0,
                "allocations in check_dtypes, check_sizes, step or outputs_into"
            );
        }
    }

    /// REVIEW-B-r1 point 2: the host's poison poisons the Env's pool.
    #[test]
    fn test_xla_host_poison_poisons_the_pool() {
        let mut pool = cartpole(&args()).unwrap();
        pool.reset(None, &ResetSeed::Keep).unwrap();
        NativeXla::poison(pool.as_mut());
        assert!(pool.is_poisoned());
        assert!(pool.step(crate::env::discrete(&[1, 1])).is_err());
    }
}
