// SPDX-License-Identifier: MIT OR Apache-2.0

//! The outputs of a host object (SPEC 14.4.4 draft, ADR-0065 item 5): one
//! view per output region, exported read-only through the buffer protocol
//! and DLPack. A view holds an `Arc` of the output storage, so its memory
//! outlives `close` (ADR-0003 rule 1); its contents are guaranteed until
//! the next `reset`, `step` or `close`, which its generations report.

use std::ffi::CString;
use std::sync::Arc;

use tessera_rl_core::space::DType;
use tessera_rl_engine::{HostLayout, OutputDtype, OutputField, OutputStorage};

use crate::errors::{self, Kind};
use crate::ffi::buffer::{Buffer, Element, Export, ExportError, Rows};
use crate::ffi::capi::{self, CResult, Gil, Mine, Ref};
use crate::ffi::object::{
    Body, MethodKw, MethodNoArgs, MethodRow, MethodTable, TypeObject, TypeSpec,
};

/// A view's body.
pub struct HostView {
    /// The exported rows; keeps their memory alive. It owns memory, never
    /// threads, so dropping it in a forked child is safe.
    export: Export,
    /// The storage's generation is the pool's current generation.
    storage: Arc<OutputStorage>,
    /// The generation of the window this view was taken in.
    generation: u64,
}

impl Body for HostView {}

/// One view to export: a region, its element type and its shape.
#[derive(Debug, Clone)]
struct Spec {
    field: OutputField,
    element: Element,
    shape: Vec<usize>,
}

/// The element of an observation leaf.
const fn leaf_element(dtype: DType) -> Element {
    match dtype {
        DType::Bool => Element::Bool,
        DType::I8 => Element::I8,
        DType::I16 => Element::I16,
        DType::I32 => Element::I32,
        DType::I64 => Element::I64,
        DType::U8 => Element::U8,
        DType::U16 => Element::U16,
        DType::U32 => Element::U32,
        DType::F32 => Element::F32,
        DType::F64 => Element::F64,
    }
}

/// The element of a non-observation region.
const fn region_element(dtype: OutputDtype) -> Option<Element> {
    match dtype {
        OutputDtype::Bool8 => Some(Element::Bool),
        OutputDtype::F32 => Some(Element::F32),
        OutputDtype::U32 => Some(Element::U32),
        OutputDtype::U8 => Some(Element::U8),
        _ => None,
    }
}

/// The views of one call, keyed as SPEC 14.4.4 (draft) lists them.
struct Set {
    observations: Vec<Spec>,
    terminal_observations: Vec<Spec>,
    others: Vec<(&'static str, Spec)>,
}

fn usize_shape(shape: &[u32]) -> Vec<usize> {
    shape
        .iter()
        .map(|&d| usize::try_from(d).unwrap_or(usize::MAX))
        .collect()
}

/// The view set of `l`; `None` for a region of a dtype no protocol names.
fn set(l: &HostLayout) -> Option<Set> {
    let (n, a) = (l.num_envs, l.agents);
    let leaves = |field: fn(usize) -> OutputField| {
        l.leaves
            .iter()
            .enumerate()
            .map(|(i, leaf)| {
                let mut shape = vec![n, a];
                shape.extend(usize_shape(&leaf.shape));
                Spec {
                    field: field(i),
                    element: leaf_element(leaf.dtype),
                    shape,
                }
            })
            .collect::<Vec<_>>()
    };
    let spec = |field, dtype, shape: &[usize]| {
        Some(Spec {
            field,
            element: region_element(dtype)?,
            shape: shape.to_vec(),
        })
    };
    let per_agent = [n, a];
    let per_env = [n];
    Some(Set {
        observations: leaves(OutputField::Observation),
        terminal_observations: if l.terminal {
            leaves(OutputField::TerminalObservation)
        } else {
            Vec::new()
        },
        others: vec![
            (
                "rewards",
                spec(OutputField::Rewards, OutputDtype::F32, &per_agent)?,
            ),
            (
                "agent_terminated",
                spec(OutputField::AgentTerminated, OutputDtype::Bool8, &per_agent)?,
            ),
            (
                "agent_truncated",
                spec(OutputField::AgentTruncated, OutputDtype::Bool8, &per_agent)?,
            ),
            (
                "alive",
                spec(OutputField::Alive, OutputDtype::Bool8, &per_agent)?,
            ),
            (
                "acting",
                spec(OutputField::Acting, OutputDtype::Bool8, &per_agent)?,
            ),
            (
                "episode_terminated",
                spec(OutputField::EpisodeTerminated, OutputDtype::Bool8, &per_env)?,
            ),
            (
                "episode_truncated",
                spec(OutputField::EpisodeTruncated, OutputDtype::Bool8, &per_env)?,
            ),
            (
                "masks",
                spec(OutputField::Masks, OutputDtype::Bool8, &[n, a, l.mask_len])?,
            ),
            (
                "slot_generations",
                spec(OutputField::Generations, OutputDtype::U32, &per_agent)?,
            ),
            (
                "episode_return",
                spec(OutputField::EpisodeReturn, OutputDtype::F32, &per_env)?,
            ),
            (
                "episode_length",
                spec(OutputField::EpisodeLength, OutputDtype::F32, &per_env)?,
            ),
            // ADR-0065 item 5: the host-only outputs.
            (
                "aborted",
                spec(OutputField::Aborted, OutputDtype::Bool8, &per_env)?,
            ),
            (
                "abort_code",
                spec(OutputField::AbortCode, OutputDtype::U32, &per_env)?,
            ),
            (
                "pending_reset",
                spec(OutputField::PendingReset, OutputDtype::Bool8, &per_env)?,
            ),
        ],
    })
}

static TYPE: TypeObject<HostView> = TypeObject::new(&SPEC);

static SPEC: TypeSpec<HostView> = TypeSpec::new(
    c"_tessera_host.HostView",
    c"A borrowed, read-only view of one output of a host object (valid until its next call).",
    &METHODS,
)
.with_buffer::<ViewBuffer>();

/// Adds the view type to `module`, as `name` (`<module>.HostView`).
pub(super) fn register(g: Gil<'_>, module: Ref<'_>, name: &'static core::ffi::CStr) -> CResult<()> {
    TYPE.create_as(g, module, c"HostView", name)
}

fn refused(g: Gil<'_>, field: OutputField, e: ExportError) -> capi::Raised {
    match e {
        ExportError::NoField => errors::error(g, Kind::Argument, format!("no output {field:?}")),
        ExportError::Region => {
            errors::error(g, Kind::Argument, format!("output {field:?} too large"))
        }
        ExportError::Layout => errors::error(
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

fn view<'py>(g: Gil<'py>, storage: &Arc<OutputStorage>, spec: &Spec) -> CResult<Mine<'py>> {
    let rows = Rows {
        total_rows: spec.shape.first().copied().unwrap_or(0),
        first_row: 0,
        shape: &spec.shape,
    };
    let export = Export::output(storage, spec.field, rows, spec.element, true)
        .map_err(|e| refused(g, spec.field, e))?;
    TYPE.instantiate(
        g,
        HostView {
            export,
            storage: Arc::clone(storage),
            generation: storage.generation(),
        },
    )
}

fn list_of<'py>(
    g: Gil<'py>,
    items: impl IntoIterator<Item = CResult<Mine<'py>>>,
) -> CResult<Mine<'py>> {
    let list = capi::list(g)?;
    for item in items {
        capi::list_append(g, &list, &item?)?;
    }
    Ok(list)
}

/// The outputs dict of SPEC 14.4.4 (draft): views of `storage` at its
/// current generation, laid out as `l` says.
pub(super) fn outputs<'py>(
    g: Gil<'py>,
    l: &HostLayout,
    storage: &Arc<OutputStorage>,
) -> CResult<Mine<'py>> {
    let set = set(l).ok_or_else(|| {
        errors::error(
            g,
            Kind::Argument,
            "an output dtype has no buffer format".to_owned(),
        )
    })?;
    let d = capi::dict(g)?;
    let obs = list_of(g, set.observations.iter().map(|s| view(g, storage, s)))?;
    capi::dict_set(g, &d, c"observations", &obs)?;
    let term = list_of(
        g,
        set.terminal_observations
            .iter()
            .map(|s| view(g, storage, s)),
    )?;
    capi::dict_set(g, &d, c"terminal_observations", &term)?;
    for (key, spec) in &set.others {
        let key =
            CString::new(*key).map_err(|_| errors::error(g, Kind::Argument, "key".to_owned()))?;
        capi::dict_set(g, &d, &key, &view(g, storage, spec)?)?;
    }
    Ok(d)
}

/// The buffer protocol export of a view: read-only, C-contiguous.
struct ViewBuffer;

impl Buffer for ViewBuffer {
    type Body = HostView;
    fn export(_g: Gil<'_>, body: &HostView) -> CResult<Export> {
        Ok(body.export.clone())
    }
}

/// `_tessera_view_generation()`: the generation the view was taken in.
struct ViewGenerationM;

impl MethodNoArgs for ViewGenerationM {
    type Body = HostView;
    fn call<'py>(g: Gil<'py>, v: &HostView) -> CResult<Mine<'py>> {
        capi::int_u64(g, v.generation)
    }
}

/// `_tessera_current_generation()`: the pool's current generation.
struct CurrentGenerationM;

impl MethodNoArgs for CurrentGenerationM {
    type Body = HostView;
    fn call<'py>(g: Gil<'py>, v: &HostView) -> CResult<Mine<'py>> {
        capi::int_u64(g, v.storage.generation())
    }
}

/// `__dlpack_device__()`: always the CPU, `(kDLCPU, 0)`.
struct DlpackDeviceM;

impl MethodNoArgs for DlpackDeviceM {
    type Body = HostView;
    fn call<'py>(g: Gil<'py>, _v: &HostView) -> CResult<Mine<'py>> {
        capi::tuple(g, vec![capi::int_u64(g, 1)?, capi::int_u64(g, 0)?])
    }
}

/// `__dlpack__(*, stream=None, max_version=None, dl_device=None,
/// copy=None)`: as the first-party `OutputView` (SPEC 5.3 rule 6).
struct DlpackM;

impl MethodKw for DlpackM {
    type Body = HostView;
    fn call<'py>(
        g: Gil<'py>,
        v: &HostView,
        args: Ref<'py>,
        kwargs: Option<Ref<'py>>,
    ) -> CResult<Mine<'py>> {
        v.export.dlpack_call(g, args, kwargs)
    }
}

static METHODS: MethodTable<HostView> = MethodTable::new(&[
    MethodRow::kw::<DlpackM>(
        c"__dlpack__",
        c"__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None): a read-only DLPack 1.0 capsule when max_version >= (1, 0); otherwise a legacy capsule.",
    ),
    MethodRow::noargs::<DlpackDeviceM>(c"__dlpack_device__", c"(kDLCPU, 0)"),
    MethodRow::noargs::<ViewGenerationM>(
        c"_tessera_view_generation",
        c"The generation of the window this view was taken in.",
    ),
    MethodRow::noargs::<CurrentGenerationM>(
        c"_tessera_current_generation",
        c"The pool's current generation.",
    ),
    MethodRow::END,
]);
