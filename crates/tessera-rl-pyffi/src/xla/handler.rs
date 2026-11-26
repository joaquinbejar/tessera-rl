// SPDX-License-Identifier: MIT OR Apache-2.0

//! The XLA FFI handlers of the token path (ADR-0095), experimental.
//!
//! XLA calls a handler directly, on one of its threads, without the
//! interpreter. The handler enters through the audited boundary thunk
//! (`ffi::thunks::xla_handler`), so it runs under canonical floating-point
//! controls; its body (`run`) contains every panic and returns an
//! `XLA_FFI_Error` for every failure. Per call:
//!
//! 1. a metadata query (the instantiate stage) is answered with API 0.3;
//!    any stage but execute does nothing;
//! 2. the floating-point policy: `reject` refuses a non-canonical caller,
//!    `canonicalize` defers the once-per-process warning to the next
//!    Python entry (`boundary::defer_warning`);
//! 3. every argument and result is checked (type, dtype, rank, dims) and
//!    the `slot` and `generation` attributes are read;
//! 4. the registry protocol (ADR-0095 point 3), then the `Env`'s own guard
//!    (`Shared::try_with`): a busy or closed handle is refused;
//! 5. the token: refused unless it is the handle's call counter
//!    (`OrderError`, nothing stepped);
//! 6. the step or reset, its outputs copied into XLA's result buffers by
//!    the host, the counter advanced and its new value returned as the
//!    next token.
//!
//! An extension (`tessera-rl-py`) supplies the pool type and its step and
//! reset through the safe [`XlaHost`] trait; all `unsafe` stays here.

#![expect(unsafe_code, reason = "audited module: the XLA FFI handler (ADR-0095)")]

use core::ffi::c_void;
use core::marker::PhantomData;
use std::sync::Arc;

use super::registry::{Handle, Registry, RegistryError};
use super::sys::{
    ARG_BUFFER, ATTR_SCALAR, Api, ApiVersion, Attrs, Buffer, CallFrame, DTYPE_PRED, DTYPE_S32,
    DTYPE_S64, DTYPE_U8, DTYPE_U32, DataType, ERROR_FAILED_PRECONDITION, ERROR_INTERNAL,
    ERROR_INVALID_ARGUMENT, EXTENSION_METADATA, Error, ErrorCode, ErrorCreateArgs, ExtensionBase,
    MetadataExtension, RET_BUFFER, STAGE_EXECUTE, Scalar,
};
use crate::errors::Kind;
use crate::ffi::boundary::{self, Policy};
use crate::ffi::capi::{self, CResult, Gil, Mine};
use crate::ffi::dispose;
use crate::ffi::inner::{Refused, Shared};
use crate::ffi::thunks::{ThunkFn, XlaHandlerEntry};

/// Most results a call returns after the token: up to 16 observation
/// leaves, 8 output arrays and the masks.
const MAX_RETS: usize = 32;
/// Most buffers in an argument or result list (the token plus results).
const MAX_BUFFERS: usize = MAX_RETS + 1;

/// A fixed list of buffers, so the success path never allocates.
struct Bufs<'a> {
    items: [Option<&'a Buffer>; MAX_BUFFERS],
    len: usize,
}

impl<'a> Bufs<'a> {
    fn get(&self, i: usize) -> Option<&'a Buffer> {
        if i < self.len {
            self.items.get(i).copied().flatten()
        } else {
            None
        }
    }
}

/// One registered handle: the `Env`'s guarded pool, the call counter the
/// next token must equal, and the last error, which Python reads (under
/// the interpreter) and raises as its class.
pub struct XlaEntry<P: ?Sized> {
    shared: Arc<Shared<P>>,
    counter: u32,
    last_error: Option<(Kind, String)>,
    /// A panic escaped the host's step or reset: the handle refuses every
    /// later call (ADR-0028 applied to the handle).
    poisoned: bool,
}

impl<P: ?Sized> XlaEntry<P> {
    /// A new entry over `shared`, the `Env`'s guard (`Inner::shared`).
    #[must_use]
    pub fn new(shared: Arc<Shared<P>>) -> Self {
        Self {
            shared,
            counter: 0,
            last_error: None,
            poisoned: false,
        }
    }
}

/// What an extension supplies to the handlers. Safe: the handlers do every
/// unsafe step and hand the host checked slices.
pub trait XlaHost: 'static {
    /// The pool type behind the `Env` guard.
    type Pool: ?Sized + Send;

    /// The process-wide registry of handles.
    fn registry() -> &'static Registry<XlaEntry<Self::Pool>>;

    /// Steps `pool` with `actions` and copies its outputs into `outs`,
    /// whose XLA dtypes are `dtypes` (one per buffer).
    ///
    /// Contract: the host checks the actions' length and every result
    /// buffer's dtype and byte length **before** touching the pool, so a
    /// refusal leaves the pool exactly as it was (REVIEW-B-r1 point 1):
    /// the token does not advance on an error, and a retry must not step
    /// twice.
    ///
    /// # Errors
    ///
    /// The error kind and message to record as the handle's last error.
    fn step(
        pool: &mut Self::Pool,
        actions: &[i32],
        outs: &mut [&mut [u8]],
        dtypes: &[DataType],
    ) -> Result<(), (Kind, String)>;

    /// Resets every instance of `pool`, with `seed` as the base seed when
    /// given, and copies its outputs into `outs`.
    ///
    /// # Errors
    ///
    /// As [`XlaHost::step`].
    fn reset(
        pool: &mut Self::Pool,
        seed: Option<u64>,
        outs: &mut [&mut [u8]],
        dtypes: &[DataType],
    ) -> Result<(), (Kind, String)>;

    /// Poisons `pool` (ADR-0028) after a panic escaped [`XlaHost::step`]
    /// or [`XlaHost::reset`]: the pool may be half-stepped.
    fn poison(pool: &mut Self::Pool);
}

#[derive(Clone, Copy)]
enum Op {
    Step,
    Reset,
}

/// The step handler of host `H`, for [`targets`].
pub struct XlaStep<H>(PhantomData<H>);
/// The reset handler of host `H`, for [`targets`].
pub struct XlaReset<H>(PhantomData<H>);

impl<H: XlaHost> XlaHandlerEntry for XlaStep<H> {
    unsafe extern "C" fn call(frame: *mut CallFrame) -> *mut Error {
        run::<H>(frame, Op::Step)
    }
}

impl<H: XlaHost> XlaHandlerEntry for XlaReset<H> {
    unsafe extern "C" fn call(frame: *mut CallFrame) -> *mut Error {
        run::<H>(frame, Op::Reset)
    }
}

/// A failure: the XLA error code and its message.
type Fail = (ErrorCode, String);

/// The handler body: panics contained, every failure an `XLA_FFI_Error`.
fn run<H: XlaHost>(frame: *mut CallFrame, op: Op) -> *mut Error {
    // SAFETY: XLA passes a valid, exclusive call frame for the duration of
    // the call (`XLA_FFI_Handler`); a null frame is handled below.
    let Some(frame) = (unsafe { frame.as_mut() }) else {
        // No frame means no API table, so no way to create an error, and
        // returning null would report success (REVIEW-B-r1): stop the
        // process, as `make_error` does without an `error_create`.
        std::process::abort();
    };
    let api = frame.api;
    let outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute::<H>(frame, op)));
    match outcome {
        Ok(Ok(())) => core::ptr::null_mut(),
        Ok(Err((code, message))) => make_error(api, code, &message),
        Err(payload) => {
            dispose(payload);
            make_error(api, ERROR_INTERNAL, "internal panic in Tessera")
        }
    }
}

/// An `XLA_FFI_Error` made by XLA's own `error_create`; XLA owns it.
fn make_error(api: *const Api, code: ErrorCode, message: &str) -> *mut Error {
    let text = std::ffi::CString::new(message.replace('\0', " ")).unwrap_or_default();
    // SAFETY: `api` comes from the call frame XLA passed, valid for the
    // call; a null table is handled below.
    let create = unsafe { api.as_ref() }.and_then(|a| a.error_create);
    let Some(create) = create else {
        // No way to report an error to XLA: returning null would mean
        // success, so stop the process instead.
        std::process::abort();
    };
    let mut args = ErrorCreateArgs {
        struct_size: core::mem::size_of::<ErrorCreateArgs>(),
        extension_start: core::ptr::null_mut(),
        message: text.as_ptr(),
        errc: code,
    };
    // SAFETY: `create` is XLA's `XLA_FFI_Error_Create`; `args` is a
    // complete, live struct whose message outlives the call (XLA copies
    // it into the error).
    unsafe { create(&raw mut args) }
}

fn execute<H: XlaHost>(frame: &mut CallFrame, op: Op) -> Result<(), Fail> {
    if answer_metadata(frame.extension_start) {
        return Ok(());
    }
    if frame.stage != STAGE_EXECUTE {
        return Ok(());
    }
    let reject = match boundary::frame() {
        Some((true, _)) if boundary::policy() == Policy::Reject => true,
        Some((true, _)) => {
            boundary::defer_warning();
            false
        }
        _ => false,
    };
    let handle = handle_of(&frame.attrs)?;
    match execute_for::<H>(frame, op, handle, reject) {
        Ok(()) => Ok(()),
        Err((code, message)) => {
            // Every refusal is recorded as the handle's last error
            // (REVIEW-B-r1), except when the handle itself is busy or
            // closed: then the entry cannot be reached.
            if code == ERROR_INVALID_ARGUMENT {
                let recorded = message.clone();
                let _ = H::registry().call(handle, |e| {
                    e.last_error = Some((Kind::Argument, recorded));
                });
            }
            Err((code, message))
        }
    }
}

fn execute_for<H: XlaHost>(
    frame: &mut CallFrame,
    op: Op,
    handle: Handle,
    reject: bool,
) -> Result<(), Fail> {
    let args = buffers(&frame.args)?;
    let rets = buffers(&frame.rets)?;
    let (Some(token_in), Some(token_out)) = (args.get(0), rets.get(0)) else {
        return Err(invalid("the token argument and result are missing"));
    };
    let token = scalar_u32(token_in, "token")?;
    expect_scalar(token_out, DTYPE_U32, "the token result")?;
    let input = Input::of(op, &args)?;
    let mut outs: [&mut [u8]; MAX_RETS] = Default::default();
    let mut dtypes: [DataType; MAX_RETS] = [0; MAX_RETS];
    let n = rets.len.saturating_sub(1);
    for (i, (slot, dtype)) in outs.iter_mut().zip(dtypes.iter_mut()).enumerate().take(n) {
        let buf = rets.get(i + 1).ok_or_else(|| invalid("a missing result"))?;
        *slot = bytes_mut(buf)?;
        *dtype = buf.dtype;
    }
    let (outs, dtypes) = (
        outs.get_mut(..n).unwrap_or_default(),
        dtypes.get(..n).unwrap_or_default(),
    );
    let next = H::registry()
        .call(handle, |entry| {
            if reject {
                let message = "the caller's floating-point controls are not the default and the policy is \"reject\" (tessera_rl.set_fp_policy)".to_owned();
                entry.last_error = Some((Kind::FloatingPoint, message.clone()));
                return Err((ERROR_FAILED_PRECONDITION, message));
            }
            if entry.poisoned {
                let message = "the handle is poisoned by an earlier panic in its step (ADR-0028); close it".to_owned();
                entry.last_error = Some((Kind::Poisoned, message.clone()));
                return Err((ERROR_FAILED_PRECONDITION, message));
            }
            if token != entry.counter {
                let message = format!(
                    "ordering token {token} is not the handle's next one ({}): a stale, replayed or reordered call; nothing was stepped",
                    entry.counter
                );
                entry.last_error = Some((Kind::Order, message.clone()));
                return Err((ERROR_FAILED_PRECONDITION, message));
            }
            // A panic in the host's step or reset (one the engine did not
            // contain as `Poisoned`) poisons the handle and the `Env`'s pool,
            // per ADR-0028 (REVIEW-B-r1 point 2): the pool may be
            // half-stepped and the results half-written. The `Env` guard was
            // released by its own guard on the unwind, so it is taken again
            // to poison the pool; Python then sees `PoisonedError` as after
            // any worker panic, and a full reset clears it.
            let shared = &entry.shared;
            let done = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                shared.try_with(|pool| match input {
                    Input::Step(actions) => H::step(pool, actions, outs, dtypes),
                    Input::Reset(seed) => H::reset(pool, seed, outs, dtypes),
                })
            }));
            let done = match done {
                Ok(done) => done,
                Err(payload) => {
                    dispose(payload);
                    let _ = shared.try_with(|pool| H::poison(pool));
                    let message = "a panic in the XLA FFI step poisoned this handle and its environment (ADR-0028); close the handle".to_owned();
                    entry.poisoned = true;
                    entry.last_error = Some((Kind::Poisoned, message.clone()));
                    return Err((ERROR_INTERNAL, message));
                }
            };
            match done {
                Err(Refused::Busy) => {
                    let message = "another call on this environment is running".to_owned();
                    entry.last_error = Some((Kind::Concurrency, message.clone()));
                    Err((ERROR_FAILED_PRECONDITION, message))
                }
                Err(Refused::Closed) => {
                    let message = "the environment is closed".to_owned();
                    entry.last_error = Some((Kind::Argument, message.clone()));
                    Err((ERROR_FAILED_PRECONDITION, message))
                }
                Ok(Err((kind, message))) => {
                    entry.last_error = Some((kind, message.clone()));
                    Err((ERROR_INTERNAL, message))
                }
                Ok(Ok(())) => {
                    // Wraps after 2^32 calls: the token is compared for
                    // equality only, so the order check holds across the
                    // wrap (a token 2^32 calls stale is not detectable).
                    entry.counter = entry.counter.wrapping_add(1);
                    Ok(entry.counter)
                }
            }
        })
        .map_err(|e| {
            let what = match e {
                RegistryError::Busy => "another call on this handle is running",
                RegistryError::Closed | RegistryError::Full => "the handle is closed",
            };
            (ERROR_FAILED_PRECONDITION, what.to_owned())
        })??;
    write_u32(token_out, next)
}

/// The decoded inputs after the token.
#[derive(Clone, Copy)]
enum Input<'a> {
    Step(&'a [i32]),
    Reset(Option<u64>),
}

impl<'a> Input<'a> {
    fn of(op: Op, args: &Bufs<'a>) -> Result<Self, Fail> {
        match op {
            Op::Step => {
                let (Some(actions), 2) = (args.get(1), args.len) else {
                    return Err(invalid("step takes the token and the actions"));
                };
                if actions.dtype != DTYPE_S32 {
                    return Err(invalid("actions must be int32"));
                }
                Ok(Self::Step(i32s(actions)?))
            }
            Op::Reset => {
                let (Some(seed), Some(use_seed), 3) = (args.get(1), args.get(2), args.len) else {
                    return Err(invalid("reset takes the token, the seed and its flag"));
                };
                if seed.dtype != DTYPE_U32 || dims(seed)? != [2] {
                    return Err(invalid("the seed must be uint32[2]"));
                }
                let flag = scalar_flag(use_seed)?;
                let words = u32s(seed)?;
                let seed = match (flag, words) {
                    (true, [lo, hi]) => Some(u64::from(*lo) | (u64::from(*hi) << 32)),
                    _ => None,
                };
                Ok(Self::Reset(seed))
            }
        }
    }
}

/// Answers XLA's metadata query (ADR-0095 point 4): API 0.3, no traits.
fn answer_metadata(mut ext: *mut ExtensionBase) -> bool {
    loop {
        // SAFETY: XLA's extension chain is a null-terminated linked list of
        // live `XLA_FFI_Extension_Base` headers, valid for the call.
        let Some(base) = (unsafe { ext.as_ref() }) else {
            return false;
        };
        if base.kind == EXTENSION_METADATA {
            let meta_ext = ext.cast::<MetadataExtension>();
            // SAFETY: an extension of type `XLA_FFI_Extension_Metadata` is
            // an `XLA_FFI_Metadata_Extension` (c_api.h), live for the call.
            let metadata = unsafe { (*meta_ext).metadata };
            // SAFETY: XLA points `metadata` at a live, writable struct for
            // the handler to fill.
            if let Some(m) = unsafe { metadata.as_mut() } {
                m.api_version = ApiVersion {
                    struct_size: core::mem::size_of::<ApiVersion>(),
                    extension_start: core::ptr::null_mut(),
                    major_version: super::sys::API_MAJOR,
                    minor_version: super::sys::API_MINOR,
                };
                m.traits = 0;
            }
            return true;
        }
        ext = base.next;
    }
}

fn invalid(what: &str) -> Fail {
    (ERROR_INVALID_ARGUMENT, what.to_owned())
}

/// The buffers of an argument or result list, all of buffer type.
fn buffers<'a, L: BufferList>(list: &L) -> Result<Bufs<'a>, Fail> {
    let n = usize::try_from(list.size()).map_err(|_| invalid("negative size"))?;
    if n > MAX_BUFFERS {
        return Err(invalid("too many buffers"));
    }
    let mut out = Bufs {
        items: [None; MAX_BUFFERS],
        len: n,
    };
    for i in 0..n {
        // SAFETY: `types` holds `size` entries (c_api.h), so `i` is in bounds.
        let ty_at = unsafe { list.types().add(i) };
        // SAFETY: in bounds, per above, and live for the call.
        let ty = unsafe { *ty_at };
        if ty != L::BUFFER {
            return Err(invalid("every argument and result must be a buffer"));
        }
        // SAFETY: the pointer array holds `size` entries, so `i` is in bounds.
        let item_at = unsafe { list.items().add(i) };
        // SAFETY: in bounds and live; a buffer entry points at an
        // `XLA_FFI_Buffer` for the call.
        let ptr = unsafe { *item_at }.cast::<Buffer>();
        // SAFETY: live for the call, per above; null is refused.
        let buf = unsafe { ptr.as_ref() }.ok_or_else(|| invalid("a null buffer"))?;
        if let Some(item) = out.items.get_mut(i) {
            *item = Some(buf);
        }
    }
    Ok(out)
}

/// An argument or result list: the same shape in `XLA_FFI_Args` and
/// `XLA_FFI_Rets`.
trait BufferList {
    const BUFFER: i32;
    fn size(&self) -> i64;
    fn types(&self) -> *const i32;
    fn items(&self) -> *const *mut c_void;
}

impl BufferList for super::sys::Args {
    const BUFFER: i32 = ARG_BUFFER;
    fn size(&self) -> i64 {
        self.size
    }
    fn types(&self) -> *const i32 {
        self.types
    }
    fn items(&self) -> *const *mut c_void {
        self.args
    }
}

impl BufferList for super::sys::Rets {
    const BUFFER: i32 = RET_BUFFER;
    fn size(&self) -> i64 {
        self.size
    }
    fn types(&self) -> *const i32 {
        self.types
    }
    fn items(&self) -> *const *mut c_void {
        self.rets
    }
}

fn dims(buf: &Buffer) -> Result<&[i64], Fail> {
    let rank = usize::try_from(buf.rank).map_err(|_| invalid("a negative rank"))?;
    if rank == 0 {
        return Ok(&[]);
    }
    if buf.dims.is_null() {
        return Err(invalid("null dims with a nonzero rank"));
    }
    // SAFETY: `dims` holds `rank` entries for the call (c_api.h), non-null.
    Ok(unsafe { core::slice::from_raw_parts(buf.dims, rank) })
}

fn dtype_size(dtype: DataType) -> Option<usize> {
    match dtype {
        DTYPE_PRED | DTYPE_U8 | super::sys::DTYPE_S8 => Some(1),
        DTYPE_S32 | DTYPE_U32 | super::sys::DTYPE_F32 => Some(4),
        DTYPE_S64 | super::sys::DTYPE_U64 | super::sys::DTYPE_F64 => Some(8),
        _ => None,
    }
}

/// The buffer's byte length from its dtype and dims.
fn byte_len(buf: &Buffer) -> Result<usize, Fail> {
    let size = dtype_size(buf.dtype).ok_or_else(|| invalid("an unsupported dtype"))?;
    let mut n = size;
    for &d in dims(buf)? {
        let d = usize::try_from(d).map_err(|_| invalid("a negative dimension"))?;
        n = n
            .checked_mul(d)
            .ok_or_else(|| invalid("a buffer too large"))?;
    }
    Ok(n)
}

fn bytes_mut<'a>(buf: &Buffer) -> Result<&'a mut [u8], Fail> {
    let len = byte_len(buf)?;
    if len == 0 {
        return Ok(&mut []);
    }
    if buf.data.is_null() {
        return Err(invalid("a null result buffer"));
    }
    // SAFETY: XLA allocates each result buffer with the dims and dtype it
    // declares, exclusively for this call; distinct results do not alias.
    Ok(unsafe { core::slice::from_raw_parts_mut(buf.data.cast::<u8>(), len) })
}

fn typed<'a, T>(buf: &Buffer, what: &str) -> Result<&'a [T], Fail> {
    let len = byte_len(buf)?;
    if len == 0 {
        return Ok(&[]);
    }
    let ptr = buf.data.cast::<T>();
    if ptr.is_null() || !ptr.is_aligned() {
        return Err(invalid(what));
    }
    // SAFETY: a live, aligned input buffer of `len` bytes for the call
    // (checked above), read-only.
    Ok(unsafe { core::slice::from_raw_parts(ptr, len / core::mem::size_of::<T>()) })
}

fn i32s<'a>(buf: &Buffer) -> Result<&'a [i32], Fail> {
    typed(buf, "actions are not aligned int32")
}

fn u32s<'a>(buf: &Buffer) -> Result<&'a [u32], Fail> {
    typed(buf, "a uint32 buffer is not aligned")
}

fn expect_scalar(buf: &Buffer, dtype: DataType, what: &str) -> Result<(), Fail> {
    if buf.dtype != dtype || !dims(buf)?.is_empty() {
        return Err((ERROR_INVALID_ARGUMENT, format!("{what} must be a scalar")));
    }
    Ok(())
}

fn scalar_u32(buf: &Buffer, what: &str) -> Result<u32, Fail> {
    expect_scalar(buf, DTYPE_U32, what)?;
    u32s(buf)?
        .first()
        .copied()
        .ok_or_else(|| invalid("an empty token"))
}

fn scalar_flag(buf: &Buffer) -> Result<bool, Fail> {
    if !dims(buf)?.is_empty() || (buf.dtype != DTYPE_PRED && buf.dtype != DTYPE_U8) {
        return Err(invalid("the seed flag must be a bool scalar"));
    }
    let byte: &[u8] = typed(buf, "the seed flag")?;
    Ok(byte.first().is_some_and(|&b| b != 0))
}

fn write_u32(buf: &Buffer, value: u32) -> Result<(), Fail> {
    let out = bytes_mut(buf)?;
    let Some(dst) = out.get_mut(..4) else {
        return Err(invalid("the token result is too small"));
    };
    dst.copy_from_slice(&value.to_ne_bytes());
    Ok(())
}

/// The `slot` and `generation` attributes (sorted by name, `int64`).
fn handle_of(attrs: &Attrs) -> Result<Handle, Fail> {
    let n = usize::try_from(attrs.size).map_err(|_| invalid("negative attribute count"))?;
    let (mut slot, mut generation) = (None, None);
    for i in 0..n {
        // SAFETY: the three arrays hold `size` entries, so `i` is in bounds.
        let ty_at = unsafe { attrs.types.add(i) };
        // SAFETY: as above.
        let name_at = unsafe { attrs.names.add(i) };
        // SAFETY: as above.
        let value_at = unsafe { attrs.attrs.add(i) };
        // SAFETY: in bounds and live for the call.
        let ty = unsafe { *ty_at };
        // SAFETY: as above.
        let name_ptr = unsafe { *name_at };
        // SAFETY: as above.
        let value_ptr = unsafe { *value_at };
        // SAFETY: a live `XLA_FFI_ByteSpan` for the call; null is skipped.
        let Some(name) = (unsafe { name_ptr.as_ref() }) else {
            continue;
        };
        if name.ptr.is_null() || ty != ATTR_SCALAR {
            continue;
        }
        // SAFETY: the byte span holds `len` bytes for the call.
        let name = unsafe { core::slice::from_raw_parts(name.ptr.cast::<u8>(), name.len) };
        // SAFETY: a scalar attribute points at a live `XLA_FFI_Scalar`.
        let Some(scalar) = (unsafe { value_ptr.cast::<Scalar>().as_ref() }) else {
            continue;
        };
        if scalar.dtype != DTYPE_S64 || scalar.value.is_null() {
            continue;
        }
        // SAFETY: an `int64` scalar's value points at a live `i64`; read
        // unaligned since nothing documents its alignment.
        let value = unsafe { scalar.value.cast::<i64>().read_unaligned() };
        let target = match name {
            b"slot" => &mut slot,
            b"generation" => &mut generation,
            _ => continue,
        };
        if target.is_some() {
            return Err(invalid("a duplicate slot or generation attribute"));
        }
        *target = Some(value);
    }
    // A handle is not a capability (#738): it is built here from the call's
    // attributes, and `_xla_token`, `_xla_take_error` and `_xla_close` build
    // it from Python integers, so any caller can name any slot and
    // generation. The registry checks every use against the slot's current
    // generation and entry; a handle that names nothing live is refused or,
    // in `close`, does nothing to a slot that holds or is getting an entry.
    match (slot.map(u32::try_from), generation.map(u64::try_from)) {
        (Some(Ok(slot)), Some(Ok(generation))) => Ok(Handle { slot, generation }),
        _ => Err(invalid(
            "the slot and generation attributes are missing or out of range",
        )),
    }
}

// -- the Python side (under the interpreter) ----------------------------------------------

/// Registers the `Env`'s guard as a new handle.
///
/// # Errors
///
/// `CapabilityError` when every slot is taken.
pub fn open<H: XlaHost>(g: Gil<'_>, shared: Arc<Shared<H::Pool>>) -> CResult<Handle> {
    H::registry().open(XlaEntry::new(shared)).map_err(|_| {
        crate::errors::error(
            g,
            Kind::Capability,
            "every XLA FFI handle slot is taken".to_owned(),
        )
    })
}

/// The handle's next token.
///
/// # Errors
///
/// `ConcurrencyError` while a call runs, `ArgumentError` once closed.
pub fn token<H: XlaHost>(g: Gil<'_>, handle: Handle) -> CResult<u32> {
    H::registry()
        .call(handle, |e| e.counter)
        .map_err(|e| registry_error(g, e))
}

/// Takes the handle's last error, raised as its class by the caller.
///
/// # Errors
///
/// As [`token`].
pub fn take_last_error<H: XlaHost>(g: Gil<'_>, handle: Handle) -> CResult<Option<(Kind, String)>> {
    H::registry()
        .call(handle, |e| e.last_error.take())
        .map_err(|e| registry_error(g, e))
}

/// Closes the handle: an in-flight call finishes first (ADR-0095 point 3).
pub fn close<H: XlaHost>(g: Gil<'_>, handle: Handle) {
    g.detach(|| H::registry().close(handle));
}

fn registry_error(g: Gil<'_>, e: RegistryError) -> capi::Raised {
    match e {
        RegistryError::Busy => crate::errors::error(
            g,
            Kind::Concurrency,
            "another call on this handle is running".to_owned(),
        ),
        RegistryError::Closed | RegistryError::Full => {
            crate::errors::error(g, Kind::Argument, "the handle is closed".to_owned())
        }
    }
}

/// `{"step": capsule, "reset": capsule}`: the handlers' boundary thunks as
/// the capsules `jax.ffi.register_ffi_target` takes (a null name and no
/// destructor, as `jax.ffi.pycapsule` builds them in the pinned JAX).
///
/// # Errors
///
/// A Python exception if the dict or a capsule cannot be made.
pub fn targets<H: XlaHost>(g: Gil<'_>) -> CResult<Mine<'_>> {
    let step = ThunkFn::<super::sys::Handler>::of::<XlaStep<H>>().get();
    let reset = ThunkFn::<super::sys::Handler>::of::<XlaReset<H>>().get();
    let dict = capi::dict(g)?;
    for (name, f) in [(c"step", step), (c"reset", reset)] {
        let cap = capi::fn_capsule(g, f as *const c_void)?;
        capi::dict_set(g, &dict, name, &cap)?;
    }
    Ok(dict)
}

#[cfg(test)]
#[cfg(not(tessera_xla_loom))]
mod tests {
    //! The handlers against hand-built call frames (no XLA): every refusal,
    //! a successful step, the token, a panic, and the boundary thunk under
    //! hostile floating-point controls (ADR-0041, ADR-0095).

    use core::cell::Cell;
    use core::ffi::c_void;

    use tessera_rl_num::fp::{self, hostile};

    use super::{Bufs, Kind, Op, Registry, XlaEntry, XlaHost, XlaReset, XlaStep, run};
    use crate::ffi::inner::Inner;
    use crate::ffi::thunks::ThunkFn;
    use crate::xla::registry::Handle;
    use crate::xla::sys::{
        ARG_BUFFER, ATTR_SCALAR, Api, ApiVersion, Args, Attrs, Buffer, ByteSpan, CallFrame,
        DTYPE_PRED, DTYPE_S32, DTYPE_S64, DTYPE_U32, DataType, EXTENSION_METADATA, Error,
        ErrorCode, ErrorCreateArgs, ExtensionBase, Handler, Metadata, MetadataExtension,
        RET_BUFFER, Rets, STAGE_EXECUTE, STAGE_INSTANTIATE, Scalar, TypeId,
    };

    /// The toy pool: a step counter; a step writes it into the first
    /// result and the sum of the actions into the second.
    struct Toy {
        steps: u32,
        panic_next: bool,
        poisoned: bool,
    }

    thread_local! {
        static LAST_CODE: Cell<ErrorCode> = const { Cell::new(0) };
        static CANONICAL_INSIDE: Cell<bool> = const { Cell::new(false) };
        /// Allocations made by the last direct `run` (feature `test-hooks`).
        static RUN_ALLOCATIONS: Cell<u64> = const { Cell::new(u64::MAX) };
    }

    struct ToyHost;

    static TOYS: Registry<XlaEntry<Toy>> = Registry::new();

    impl XlaHost for ToyHost {
        type Pool = Toy;

        fn registry() -> &'static Registry<XlaEntry<Toy>> {
            &TOYS
        }

        fn step(
            pool: &mut Toy,
            actions: &[i32],
            outs: &mut [&mut [u8]],
            dtypes: &[DataType],
        ) -> Result<(), (Kind, String)> {
            CANONICAL_INSIDE.with(|c| c.set(fp::is_canonical_bits(fp::current_bits())));
            if pool.panic_next {
                pool.panic_next = false;
                panic!("toy panic");
            }
            if dtypes != [DTYPE_U32, DTYPE_S32] {
                return Err((Kind::Argument, "toy dtypes".to_owned()));
            }
            pool.steps += 1;
            let [steps, sum] = outs else {
                return Err((Kind::Argument, "toy results".to_owned()));
            };
            steps.copy_from_slice(&pool.steps.to_ne_bytes());
            sum.copy_from_slice(&actions.iter().sum::<i32>().to_ne_bytes());
            Ok(())
        }

        fn poison(pool: &mut Toy) {
            pool.poisoned = true;
        }

        fn reset(
            pool: &mut Toy,
            seed: Option<u64>,
            outs: &mut [&mut [u8]],
            _dtypes: &[DataType],
        ) -> Result<(), (Kind, String)> {
            pool.steps = 0;
            let [steps, sum] = outs else {
                return Err((Kind::Argument, "toy results".to_owned()));
            };
            steps.copy_from_slice(&0u32.to_ne_bytes());
            let low = u32::try_from(seed.unwrap_or(0) & 0x7fff_ffff).unwrap_or(0);
            sum.copy_from_slice(&i32::try_from(low).unwrap_or(0).to_ne_bytes());
            Ok(())
        }
    }

    unsafe extern "C" fn record_error(args: *mut ErrorCreateArgs) -> *mut Error {
        // SAFETY: the handler passes a live struct.
        let code = unsafe { (*args).errc };
        LAST_CODE.with(|c| c.set(code));
        core::ptr::NonNull::<Error>::dangling().as_ptr()
    }

    fn api() -> Api {
        Api {
            struct_size: core::mem::size_of::<Api>(),
            extension_start: core::ptr::null_mut(),
            api_version: ApiVersion {
                struct_size: core::mem::size_of::<ApiVersion>(),
                extension_start: core::ptr::null_mut(),
                major_version: 0,
                minor_version: 3,
            },
            internal_api: core::ptr::null(),
            error_create: Some(record_error),
        }
    }

    /// One owned buffer: data, dims and the `XLA_FFI_Buffer` over them.
    struct Owned {
        data: Vec<u8>,
        dims: Vec<i64>,
        dtype: DataType,
    }

    impl Owned {
        fn new(dtype: DataType, dims: &[i64], data: Vec<u8>) -> Self {
            Self {
                data,
                dims: dims.to_vec(),
                dtype,
            }
        }

        fn view(&mut self) -> Buffer {
            Buffer {
                struct_size: core::mem::size_of::<Buffer>(),
                extension_start: core::ptr::null_mut(),
                dtype: self.dtype,
                data: self.data.as_mut_ptr().cast::<c_void>(),
                rank: i64::try_from(self.dims.len()).unwrap(),
                dims: self.dims.as_mut_ptr(),
            }
        }
    }

    fn u32_scalar(v: u32) -> Owned {
        Owned::new(DTYPE_U32, &[], v.to_ne_bytes().to_vec())
    }

    /// Runs `op` on a frame built from `args`, `rets` and `handle` at
    /// `stage`; returns the error code (0 for success) and the results.
    fn call(
        op: Op,
        stage: i32,
        args: Vec<Owned>,
        rets: Vec<Owned>,
        handle: Option<Handle>,
        via_thunk: Option<Handler>,
    ) -> (ErrorCode, Vec<Vec<u8>>) {
        call_tweak(op, stage, args, rets, handle, via_thunk, |_, _| {})
    }

    fn call_with_rank(args: Vec<Owned>, rank: i64, h: Option<Handle>) -> (ErrorCode, Vec<Vec<u8>>) {
        call_tweak(
            Op::Step,
            STAGE_EXECUTE,
            args,
            step_rets(),
            h,
            None,
            |a, _| a[1].rank = rank,
        )
    }

    fn call_with_null_dims(args: Vec<Owned>, h: Option<Handle>) -> (ErrorCode, Vec<Vec<u8>>) {
        call_tweak(
            Op::Step,
            STAGE_EXECUTE,
            args,
            step_rets(),
            h,
            None,
            |a, _| {
                a[1].dims = core::ptr::null_mut();
            },
        )
    }

    fn call_tweak(
        op: Op,
        stage: i32,
        args: Vec<Owned>,
        rets: Vec<Owned>,
        handle: Option<Handle>,
        via_thunk: Option<Handler>,
        tweak: impl FnOnce(&mut [Buffer], &mut [ByteSpan; 2]),
    ) -> (ErrorCode, Vec<Vec<u8>>) {
        let mut args = args;
        let mut rets = rets;
        let mut arg_views: Vec<Buffer> = args.iter_mut().map(Owned::view).collect();
        let mut ret_views: Vec<Buffer> = rets.iter_mut().map(Owned::view).collect();
        let mut arg_types = vec![ARG_BUFFER; arg_views.len()];
        let mut ret_types = vec![RET_BUFFER; ret_views.len()];
        let (slot, generation) = handle.map_or((0i64, 0i64), |h| {
            (i64::from(h.slot), i64::try_from(h.generation).unwrap())
        });
        let mut values = [generation, slot];
        let mut scalars = [
            Scalar {
                dtype: DTYPE_S64,
                value: (&raw mut values[0]).cast(),
            },
            Scalar {
                dtype: DTYPE_S64,
                value: (&raw mut values[1]).cast(),
            },
        ];
        let mut names = [
            ByteSpan {
                ptr: b"generation".as_ptr().cast(),
                len: 10,
            },
            ByteSpan {
                ptr: b"slot".as_ptr().cast(),
                len: 4,
            },
        ];
        tweak(&mut arg_views, &mut names);
        // Raw pointers only after the tweak: a later `&mut` to the views
        // would invalidate them (Stacked Borrows, Miri).
        let mut arg_ptrs: Vec<*mut c_void> = arg_views
            .iter_mut()
            .map(|b| core::ptr::from_mut(b).cast())
            .collect();
        let mut ret_ptrs: Vec<*mut c_void> = ret_views
            .iter_mut()
            .map(|b| core::ptr::from_mut(b).cast())
            .collect();
        let mut name_ptrs = [&raw mut names[0], &raw mut names[1]];
        let mut attr_ptrs: [*mut c_void; 2] =
            [(&raw mut scalars[0]).cast(), (&raw mut scalars[1]).cast()];
        let mut attr_types = [ATTR_SCALAR; 2];
        let n_attrs = if handle.is_some() { 2 } else { 0 };
        let api = api();
        let mut frame = CallFrame {
            struct_size: core::mem::size_of::<CallFrame>(),
            extension_start: core::ptr::null_mut(),
            api: &raw const api,
            ctx: core::ptr::null_mut(),
            stage,
            args: Args {
                struct_size: core::mem::size_of::<Args>(),
                extension_start: core::ptr::null_mut(),
                size: i64::try_from(arg_ptrs.len()).unwrap(),
                types: arg_types.as_mut_ptr(),
                args: arg_ptrs.as_mut_ptr(),
            },
            rets: Rets {
                struct_size: core::mem::size_of::<Rets>(),
                extension_start: core::ptr::null_mut(),
                size: i64::try_from(ret_ptrs.len()).unwrap(),
                types: ret_types.as_mut_ptr(),
                rets: ret_ptrs.as_mut_ptr(),
            },
            attrs: Attrs {
                struct_size: core::mem::size_of::<Attrs>(),
                extension_start: core::ptr::null_mut(),
                size: n_attrs,
                types: attr_types.as_mut_ptr(),
                names: name_ptrs.as_mut_ptr(),
                attrs: attr_ptrs.as_mut_ptr(),
            },
            future: core::ptr::null_mut(),
        };
        LAST_CODE.with(|c| c.set(0));
        let err = match via_thunk {
            // SAFETY: the thunk of a test handler, with a live frame.
            Some(thunk) => unsafe { thunk(&raw mut frame) },
            None => run_counted(&raw mut frame, op),
        };
        let code = if err.is_null() {
            0
        } else {
            LAST_CODE.with(Cell::get)
        };
        drop(arg_views);
        drop(ret_views);
        (code, rets.into_iter().map(|r| r.data).collect())
    }

    #[cfg(feature = "test-hooks")]
    fn run_counted(frame: *mut CallFrame, op: Op) -> *mut Error {
        let mut err = core::ptr::null_mut();
        let n = crate::xla::testing::count(|| err = run::<ToyHost>(frame, op));
        RUN_ALLOCATIONS.with(|c| c.set(n));
        err
    }

    #[cfg(not(feature = "test-hooks"))]
    fn run_counted(frame: *mut CallFrame, op: Op) -> *mut Error {
        run::<ToyHost>(frame, op)
    }

    /// ADR-0095 point 5: the success path allocates nothing.
    #[cfg(feature = "test-hooks")]
    #[test]
    fn test_handler_success_path_does_not_allocate() {
        let h = open_toy();
        for token in 0..3 {
            let (code, _) = call(
                Op::Step,
                STAGE_EXECUTE,
                step_args(token, &[1]),
                step_rets(),
                Some(h),
                None,
            );
            assert_eq!(code, 0);
            assert_eq!(RUN_ALLOCATIONS.with(Cell::get), 0, "step {token}");
        }
        TOYS.close(h);
    }

    fn step_args(token: u32, actions: &[i32]) -> Vec<Owned> {
        let bytes = actions.iter().flat_map(|a| a.to_ne_bytes()).collect();
        let n = i64::try_from(actions.len()).unwrap();
        vec![u32_scalar(token), Owned::new(DTYPE_S32, &[n], bytes)]
    }

    fn step_rets() -> Vec<Owned> {
        vec![
            u32_scalar(0),
            u32_scalar(0),
            Owned::new(DTYPE_S32, &[], vec![0; 4]),
        ]
    }

    fn open_toy() -> Handle {
        let inner = Inner::new(
            Box::new(Toy {
                steps: 0,
                panic_next: false,
                poisoned: false,
            }),
            false,
        );
        TOYS.open(XlaEntry::new(inner.shared()))
            .map_err(|_| ())
            .unwrap()
    }

    fn u32_of(b: &[u8]) -> u32 {
        u32::from_ne_bytes(b.try_into().unwrap())
    }

    #[test]
    fn test_handler_step_advances_the_token_and_writes_outputs() {
        let h = open_toy();
        let (code, out) = call(
            Op::Step,
            STAGE_EXECUTE,
            step_args(0, &[2, 3]),
            step_rets(),
            Some(h),
            None,
        );
        assert_eq!(code, 0);
        assert_eq!(u32_of(&out[0]), 1, "the next token");
        assert_eq!(u32_of(&out[1]), 1, "one step");
        assert_eq!(i32::from_ne_bytes(out[2].as_slice().try_into().unwrap()), 5);
        let (code, out) = call(
            Op::Step,
            STAGE_EXECUTE,
            step_args(1, &[1]),
            step_rets(),
            Some(h),
            None,
        );
        assert_eq!((code, u32_of(&out[0]), u32_of(&out[1])), (0, 2, 2));
        TOYS.close(h);
    }

    #[test]
    fn test_handler_stale_or_replayed_token_is_refused_without_stepping() {
        let h = open_toy();
        assert_eq!(
            call(
                Op::Step,
                STAGE_EXECUTE,
                step_args(0, &[1]),
                step_rets(),
                Some(h),
                None
            )
            .0,
            0
        );
        for stale in [0, 5] {
            let (code, _) = call(
                Op::Step,
                STAGE_EXECUTE,
                step_args(stale, &[1]),
                step_rets(),
                Some(h),
                None,
            );
            assert_eq!(code, crate::xla::sys::ERROR_FAILED_PRECONDITION);
        }
        let err = TOYS.call(h, |e| e.last_error.take()).unwrap();
        assert!(matches!(err, Some((Kind::Order, _))), "{err:?}");
        // Nothing was stepped: the next good call still sees one step.
        let (code, out) = call(
            Op::Step,
            STAGE_EXECUTE,
            step_args(1, &[1]),
            step_rets(),
            Some(h),
            None,
        );
        assert_eq!((code, u32_of(&out[1])), (0, 2));
        TOYS.close(h);
    }

    #[test]
    fn test_handler_refuses_closed_missing_attrs_and_bad_buffers() {
        let h = open_toy();
        TOYS.close(h);
        let pre = crate::xla::sys::ERROR_FAILED_PRECONDITION;
        let inv = crate::xla::sys::ERROR_INVALID_ARGUMENT;
        assert_eq!(
            call(
                Op::Step,
                STAGE_EXECUTE,
                step_args(0, &[1]),
                step_rets(),
                Some(h),
                None
            )
            .0,
            pre
        );
        let h = open_toy();
        assert_eq!(
            call(
                Op::Step,
                STAGE_EXECUTE,
                step_args(0, &[1]),
                step_rets(),
                None,
                None
            )
            .0,
            inv
        );
        let mut bad = step_args(0, &[1]);
        bad[1].dtype = DTYPE_U32;
        assert_eq!(
            call(Op::Step, STAGE_EXECUTE, bad, step_rets(), Some(h), None).0,
            inv
        );
        let mut bad_token = step_args(0, &[1]);
        bad_token[0].dims = vec![1];
        assert_eq!(
            call(
                Op::Step,
                STAGE_EXECUTE,
                bad_token,
                step_rets(),
                Some(h),
                None
            )
            .0,
            inv
        );
        TOYS.close(h);
    }

    #[test]
    fn test_handler_busy_env_guard_is_refused() {
        let inner = Inner::new(
            Box::new(Toy {
                steps: 0,
                panic_next: false,
                poisoned: false,
            }),
            false,
        );
        let shared = inner.shared();
        let h = TOYS
            .open(XlaEntry::new(inner.shared()))
            .map_err(|_| ())
            .unwrap();
        let code = shared
            .try_with(|_| {
                call(
                    Op::Step,
                    STAGE_EXECUTE,
                    step_args(0, &[1]),
                    step_rets(),
                    Some(h),
                    None,
                )
                .0
            })
            .unwrap();
        assert_eq!(code, crate::xla::sys::ERROR_FAILED_PRECONDITION);
        TOYS.close(h);
    }

    /// A panic in the step (ADR-0028, REVIEW-plan-B): an internal error,
    /// not an unwind; the handle is poisoned (later calls refused, the
    /// error is `PoisonedError`); the `Env` guard is released, so the
    /// `Env` stays usable; closing the handle works.
    #[test]
    fn test_handler_panic_poisons_the_handle_and_frees_the_env() {
        let inner = Inner::new(
            Box::new(Toy {
                steps: 0,
                panic_next: true,
                poisoned: false,
            }),
            false,
        );
        let shared = inner.shared();
        let h = TOYS
            .open(XlaEntry::new(inner.shared()))
            .map_err(|_| ())
            .unwrap();
        let (code, _) = call(
            Op::Step,
            STAGE_EXECUTE,
            step_args(0, &[1]),
            step_rets(),
            Some(h),
            None,
        );
        assert_eq!(code, crate::xla::sys::ERROR_INTERNAL);
        let err = TOYS.call(h, |e| e.last_error.take()).unwrap();
        assert!(matches!(err, Some((Kind::Poisoned, _))), "{err:?}");
        // Refused from now on, even with the right token.
        let (code, _) = call(
            Op::Step,
            STAGE_EXECUTE,
            step_args(0, &[1]),
            step_rets(),
            Some(h),
            None,
        );
        assert_eq!(code, crate::xla::sys::ERROR_FAILED_PRECONDITION);
        // The Env's guard was released by the unwind, and its pool is
        // poisoned (ADR-0028, REVIEW-B-r1 point 2).
        assert_eq!(shared.try_with(|t| (t.steps, t.poisoned)), Ok((0, true)));
        TOYS.close(h);
    }

    #[test]
    fn test_handler_other_stages_do_nothing() {
        let h = open_toy();
        let (code, out) = call(
            Op::Step,
            STAGE_INSTANTIATE,
            step_args(9, &[1]),
            step_rets(),
            Some(h),
            None,
        );
        assert_eq!(
            (code, u32_of(&out[0])),
            (0, 0),
            "nothing written outside execute"
        );
        TOYS.close(h);
    }

    #[test]
    fn test_handler_reset_takes_a_seed_and_returns_the_next_token() {
        let h = open_toy();
        let seed = [7u32, 0].iter().flat_map(|w| w.to_ne_bytes()).collect();
        let args = vec![
            u32_scalar(0),
            Owned::new(DTYPE_U32, &[2], seed),
            Owned::new(DTYPE_PRED, &[], vec![1]),
        ];
        let (code, out) = call(Op::Reset, STAGE_EXECUTE, args, step_rets(), Some(h), None);
        assert_eq!(code, 0);
        assert_eq!(u32_of(&out[0]), 1);
        assert_eq!(i32::from_ne_bytes(out[2].as_slice().try_into().unwrap()), 7);
        TOYS.close(h);
    }

    #[test]
    fn test_handler_metadata_query_answers_api_0_3() {
        let mut metadata = Metadata {
            struct_size: core::mem::size_of::<Metadata>(),
            api_version: ApiVersion {
                struct_size: core::mem::size_of::<ApiVersion>(),
                extension_start: core::ptr::null_mut(),
                major_version: -1,
                minor_version: -1,
            },
            traits: 7,
            state_type_id: TypeId { type_id: 0 },
        };
        let mut ext = MetadataExtension {
            extension_base: ExtensionBase {
                struct_size: core::mem::size_of::<MetadataExtension>(),
                kind: EXTENSION_METADATA,
                next: core::ptr::null_mut(),
            },
            metadata: &raw mut metadata,
        };
        assert!(super::answer_metadata((&raw mut ext).cast()));
        assert_eq!(
            (
                metadata.api_version.major_version,
                metadata.api_version.minor_version,
                metadata.traits
            ),
            (0, 3, 0)
        );
        assert!(!super::answer_metadata(core::ptr::null_mut()));
    }

    /// ADR-0041 through the handler's own thunk: canonical controls inside,
    /// the caller's restored, the result unchanged, no unwind.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "the boundary thunk is naked assembly and writes the FP control registers, outside Miri"
    )]
    fn test_handler_thunk_canonicalizes_hostile_controls() {
        let thunk = ThunkFn::<Handler>::of::<XlaStep<ToyHost>>().get();
        let _reset = ThunkFn::<Handler>::of::<XlaReset<ToyHost>>().get();
        for mode in [hostile::flush_to_zero, hostile::toward_zero] {
            let h = open_toy();
            CANONICAL_INSIDE.with(|c| c.set(false));
            let ((code, out), after) = hostile::run(mode, || {
                call(
                    Op::Step,
                    STAGE_EXECUTE,
                    step_args(0, &[4]),
                    step_rets(),
                    Some(h),
                    Some(thunk),
                )
            });
            assert_eq!(code, 0);
            assert_eq!(u32_of(&out[0]), 1);
            assert!(
                CANONICAL_INSIDE.with(Cell::get),
                "canonical inside the handler"
            );
            let ((), latched) = hostile::run(mode, || ());
            assert_eq!(
                hostile::control_bits(after),
                hostile::control_bits(latched),
                "the caller's controls are restored"
            );
            TOYS.close(h);
        }
    }

    /// REVIEW-B-r1 non-blocking points: a negative rank, null dims with a
    /// nonzero rank and a duplicate attribute are refused, and an argument
    /// refusal is recorded as the handle's last error.
    #[test]
    fn test_handler_refuses_bad_rank_null_dims_and_records_the_error() {
        let h = open_toy();
        let inv = crate::xla::sys::ERROR_INVALID_ARGUMENT;
        let mut neg = step_args(0, &[1]);
        neg[1].dims = vec![];
        let (code, _) = call_with_rank(neg, -1, Some(h));
        assert_eq!(code, inv);
        let err = TOYS.call(h, |e| e.last_error.take()).unwrap();
        assert!(matches!(err, Some((Kind::Argument, _))), "{err:?}");
        let (code, _) = call_with_null_dims(step_args(0, &[1]), Some(h));
        assert_eq!(code, inv);
        // Nothing was stepped by either refusal.
        let (code, out) = call(
            Op::Step,
            STAGE_EXECUTE,
            step_args(0, &[1]),
            step_rets(),
            Some(h),
            None,
        );
        assert_eq!((code, u32_of(&out[1])), (0, 1));
        TOYS.close(h);
    }

    #[test]
    fn test_handler_duplicate_attribute_is_refused() {
        let h = open_toy();
        let (code, _) = call_tweak(
            Op::Step,
            STAGE_EXECUTE,
            step_args(0, &[1]),
            step_rets(),
            Some(h),
            None,
            |_, n| {
                n[0] = ByteSpan {
                    ptr: b"slot".as_ptr().cast(),
                    len: 4,
                };
            },
        );
        assert_eq!(code, crate::xla::sys::ERROR_INVALID_ARGUMENT);
        TOYS.close(h);
    }

    #[test]
    fn test_handler_bufs_get_bounds() {
        let b = Bufs {
            items: [None; super::MAX_BUFFERS],
            len: 0,
        };
        assert!(b.get(0).is_none());
    }
}
