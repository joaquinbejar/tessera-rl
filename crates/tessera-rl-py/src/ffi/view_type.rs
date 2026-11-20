// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native.OutputView` type: one borrowed output region
//! (ADR-0003), exported read-only through the buffer protocol and DLPack.
//! A view holds an [`Export`], an `Arc` of the output storage, so its memory
//! outlives the environment; its contents are only guaranteed until the
//! next engine call that writes outputs, which `is_stale` reports.
//!
//! The exports and their memory are `tessera_rl_pyffi`'s (ADR-0062 item 8):
//! this module decides what a view covers and when it is stale.

use std::sync::Arc;

use core::sync::atomic::AtomicBool;

use tessera_rl_engine::{HalfGenerations, IdGenerations, OutputStorage};
use tessera_rl_pyffi::ffi::buffer::{Buffer, Export, ExportError, Rows};
use tessera_rl_pyffi::ffi::capi::{self, CResult, Gil, Mine, Ref};
use tessera_rl_pyffi::ffi::gather::GatherStorage;
use tessera_rl_pyffi::ffi::object::{
    Body, MethodKw, MethodNoArgs, MethodRow, MethodTable, TypeObject, TypeSpec,
};

use crate::batched::{BatchId, ReleaseEpochs, gather_stale, ids_stale};
use crate::errors::{Kind, error};
use crate::split::half_stale;
use crate::view::ViewSpec;

/// An `OutputView` object's body.
pub(crate) struct View {
    /// The exported rows; keeps their memory alive. It owns memory, never
    /// threads, so dropping it in a forked child is safe.
    export: Export,
    /// The fork generation this view was taken in: in a child it reads stale,
    /// as the copied rows may hold a step that was running at the fork.
    born: u64,
    stale: Stale,
    generation: u64,
}

impl Body for View {}

/// What ends a view's borrow window.
enum Stale {
    /// A write of the output storage: Sync mode.
    Storage(Arc<OutputStorage>),
    /// A send of the view's half, or poison. It ignores the storage's
    /// generation, which a refused call can move (ADR-0047 section 5).
    Half(HalfGenerations, usize, u64, Arc<AtomicBool>),
    /// A send or mutation of any of the view's ids, a successful writer that
    /// dropped its token, or poison.
    Ids(
        IdGenerations,
        Arc<[BatchId]>,
        ReleaseEpochs,
        Arc<AtomicBool>,
    ),
    /// The next write of the gather storage (the next `recv`), or poison.
    Gather(Arc<GatherStorage>, Arc<AtomicBool>),
}

/// How a storage view tracks staleness.
#[derive(Clone, Copy)]
pub(crate) enum Track<'a> {
    Storage,
    Half(&'a HalfGenerations, usize, u64, &'a Arc<AtomicBool>),
    Ids(
        &'a IdGenerations,
        &'a Arc<[BatchId]>,
        &'a ReleaseEpochs,
        &'a Arc<AtomicBool>,
    ),
}

static TYPE: TypeObject<View> = TypeObject::new(&SPEC);

static SPEC: TypeSpec<View> = TypeSpec::new(
    c"tessera_rl._native.OutputView",
    c"A borrowed, read-only view of one output buffer (valid until the next call).",
    &METHODS,
)
.with_buffer::<ViewBuffer>();

/// Creates the `OutputView` type and adds it to `module`.
pub(super) fn register(g: Gil<'_>, module: Ref<'_>) -> CResult<()> {
    TYPE.create(g, module, c"OutputView")
}

/// The rows of `spec`.
fn rows(spec: &ViewSpec) -> Rows<'_> {
    Rows {
        total_rows: spec.total_rows,
        first_row: spec.first_row,
        shape: &spec.shape,
    }
}

/// The Python error of a refused export of `spec`.
fn refused(g: Gil<'_>, spec: &ViewSpec, e: ExportError) -> capi::Raised {
    let field = spec.field;
    match e {
        ExportError::NoField => error(g, Kind::Argument, format!("no output {field:?}")),
        ExportError::Region => error(g, Kind::Argument, format!("output {field:?} too large")),
        ExportError::Layout => error(
            g,
            Kind::Argument,
            format!("output {field:?} does not match its layout"),
        ),
        ExportError::Undescribable => capi::raise(
            g,
            capi::buffer_error(),
            &format!("output {field:?} is too large to export"),
        ),
    }
}

/// A new view of `spec` in `storage`, at the storage's current generation.
pub(crate) fn new_view<'py>(
    g: Gil<'py>,
    storage: &Arc<OutputStorage>,
    spec: &ViewSpec,
    track: Track<'_>,
) -> CResult<Mine<'py>> {
    let export = Export::output(storage, spec.field, rows(spec), spec.elem, true)
        .map_err(|e| refused(g, spec, e))?;
    let stale = match track {
        Track::Storage => Stale::Storage(Arc::clone(storage)),
        Track::Half(gens, half, taken, unsettled) => {
            Stale::Half(gens.clone(), half, taken, Arc::clone(unsettled))
        }
        Track::Ids(gens, ids, epochs, unsettled) => Stale::Ids(
            gens.clone(),
            Arc::clone(ids),
            Arc::clone(epochs),
            Arc::clone(unsettled),
        ),
    };
    build_view(g, export, stale, storage.generation())
}

/// A new view of `spec` in the gather storage, written at generation
/// `taken`.
pub(crate) fn new_gather_view<'py>(
    g: Gil<'py>,
    storage: &Arc<GatherStorage>,
    spec: &ViewSpec,
    taken: u64,
    unsettled: &Arc<AtomicBool>,
) -> CResult<Mine<'py>> {
    let export = Export::gather(storage, spec.field, rows(spec), spec.elem, true)
        .map_err(|e| refused(g, spec, e))?;
    let stale = Stale::Gather(Arc::clone(storage), Arc::clone(unsettled));
    build_view(g, export, stale, taken)
}

fn build_view(g: Gil<'_>, export: Export, stale: Stale, generation: u64) -> CResult<Mine<'_>> {
    TYPE.instantiate(
        g,
        View {
            export,
            born: super::fork::generation(),
            stale,
            generation,
        },
    )
}

/// The buffer protocol export of a view: read-only, C-contiguous
/// (`tessera_rl_pyffi::ffi::buffer`).
struct ViewBuffer;

impl Buffer for ViewBuffer {
    type Body = View;
    fn export(_g: Gil<'_>, body: &View) -> CResult<Export> {
        Ok(body.export.clone())
    }
}

/// `generation()`: the write generation the view was taken at.
struct GenerationM;

impl MethodNoArgs for GenerationM {
    type Body = View;
    fn call<'py>(g: Gil<'py>, v: &View) -> CResult<Mine<'py>> {
        capi::int_u64(g, v.generation)
    }
}

/// `is_stale()`: whether an engine call has written outputs since the view
/// was taken (its borrow window ended).
struct IsStaleM;

impl MethodNoArgs for IsStaleM {
    type Body = View;
    fn call<'py>(g: Gil<'py>, v: &View) -> CResult<Mine<'py>> {
        if v.born != super::fork::generation() {
            return Ok(capi::boolean(g, true));
        }
        let stale = match &v.stale {
            Stale::Storage(storage) => storage.generation() != v.generation,
            Stale::Half(gens, half, taken, unsettled) => half_stale(gens, *half, *taken, unsettled),
            Stale::Ids(gens, ids, epochs, unsettled) => ids_stale(gens, ids, epochs, unsettled),
            Stale::Gather(storage, unsettled) => gather_stale(storage, v.generation, unsettled),
        };
        Ok(capi::boolean(g, stale))
    }
}

/// `__dlpack_device__()`: always the CPU, `(kDLCPU, 0)`.
struct DlpackDeviceM;

impl MethodNoArgs for DlpackDeviceM {
    type Body = View;
    fn call<'py>(g: Gil<'py>, _v: &View) -> CResult<Mine<'py>> {
        capi::tuple(g, vec![capi::int_u64(g, 1)?, capi::int_u64(g, 0)?])
    }
}

/// `__dlpack__(*, stream=None, max_version=None, dl_device=None,
/// copy=None)`: a capsule of the view, versioned and read-only when the
/// consumer accepts DLPack 1.x, legacy otherwise. A legacy capsule has no
/// read-only flag: its consumer gets a writable alias, and writing through
/// it corrupts the outputs until the next call overwrites them. The
/// argument rules are `tessera_rl_pyffi`'s, shared with host views
/// (`Export::dlpack_call`).
struct DlpackM;

impl MethodKw for DlpackM {
    type Body = View;
    fn call<'py>(
        g: Gil<'py>,
        v: &View,
        args: Ref<'py>,
        kwargs: Option<Ref<'py>>,
    ) -> CResult<Mine<'py>> {
        v.export
            .dlpack_call(g, args, kwargs, "; use outputs=\"owned\"")
    }
}

static METHODS: MethodTable<View> = MethodTable::new(&[
    MethodRow::kw::<DlpackM>(
        c"__dlpack__",
        c"__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None): a read-only DLPack 1.0 capsule when max_version >= (1, 0); otherwise a legacy capsule, which cannot be marked read-only (writing through it corrupts the outputs until the next call).",
    ),
    MethodRow::noargs::<DlpackDeviceM>(c"__dlpack_device__", c"(kDLCPU, 0)"),
    MethodRow::noargs::<GenerationM>(
        c"generation",
        c"The write generation this view was taken at.",
    ),
    MethodRow::noargs::<IsStaleM>(
        c"is_stale",
        c"Whether outputs were written since this view was taken.",
    ),
    MethodRow::END,
]);
