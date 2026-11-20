// SPDX-License-Identifier: MIT OR Apache-2.0

//! Read-only exports of output memory (ADR-0003, SPEC 5.3): the buffer
//! protocol and DLPack.
//!
//! The exported memory is always named by an `Arc` of storage this crate
//! can name: the engine's [`OutputStorage`] or the [`GatherStorage`]. An
//! [`Export`] derives its pointer and length from that storage's own region
//! and checks the requested rows and shape against it, so no caller passes
//! a pointer or a length. There is no "any memory owner" abstraction.
//!
//! An export is reference-counted. A buffer-protocol view keeps a clone in
//! its `internal` field, released by `bf_releasebuffer`, so the memory, the
//! shape and the strides it describes outlive whatever the object's body
//! does after `getbuffer` returns. A DLPack capsule keeps a clone the same
//! way (`dlpack`).
//!
//! The buffer protocol always exports read-only (`readonly = 1`, and a
//! `PyBUF_WRITABLE` request raises `BufferError`). The `read_only` flag of
//! an export applies to DLPack only (SPEC 5.3 rule 6): it sets the
//! read-only flag of a versioned capsule.

use core::ffi::{CStr, c_int, c_void};
use core::marker::PhantomData;
use core::ptr;
use std::sync::Arc;

use pyo3::ffi;
use tessera_rl_engine::{OutputDtype, OutputField, OutputStorage};

use super::capi::{self, CResult, Gil, Mine, Ref};
use super::dlpack;
use super::entry_status;
use super::gather::GatherStorage;
use super::object::{Body, instance_body};
use super::thunks::{GetBuffer, GetBufferEntry, ReleaseBuffer, ReleaseBufferEntry, ThunkFn};

type Obj = *mut ffi::PyObject;

/// A type whose objects export a buffer.
pub trait Buffer {
    /// The body of the exporting type.
    type Body: Body;
    /// The export of `body`, for one buffer request.
    ///
    /// # Errors
    ///
    /// `Err` with a Python exception set (`BufferError` is customary).
    fn export(g: Gil<'_>, body: &Self::Body) -> CResult<Export>;
}

/// The element type of an export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Element {
    /// `bool`, one byte.
    Bool,
    /// `int8`.
    I8,
    /// `int16`, little-endian.
    I16,
    /// `int32`, little-endian.
    I32,
    /// `int64`, little-endian.
    I64,
    /// `uint8`.
    U8,
    /// `uint16`, little-endian.
    U16,
    /// `uint32`, little-endian.
    U32,
    /// `float32`, little-endian.
    F32,
    /// `float64`, little-endian.
    F64,
}

impl Element {
    /// Bytes per element.
    #[must_use]
    pub const fn itemsize(self) -> usize {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    /// The buffer protocol format (struct module syntax).
    #[must_use]
    pub const fn format(self) -> &'static CStr {
        match self {
            Self::Bool => c"?",
            Self::I8 => c"<b",
            Self::I16 => c"<h",
            Self::I32 => c"<i",
            Self::I64 => c"<q",
            Self::U8 => c"<B",
            Self::U16 => c"<H",
            Self::U32 => c"<I",
            Self::F32 => c"<f",
            Self::F64 => c"<d",
        }
    }

    /// The DLPack type code (`kDLInt` 0, `kDLUInt` 1, `kDLFloat` 2,
    /// `kDLBool` 6).
    #[must_use]
    pub const fn dl_code(self) -> u8 {
        match self {
            Self::Bool => 6,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 0,
            Self::U8 | Self::U16 | Self::U32 => 1,
            Self::F32 | Self::F64 => 2,
        }
    }
}

/// Which rows of a region an export covers: the region holds `total_rows`
/// rows of `shape[1..]` elements each, and the export is `shape[0]` rows
/// from `first_row`.
#[derive(Debug, Clone, Copy)]
pub struct Rows<'a> {
    /// The rows the region holds.
    pub total_rows: usize,
    /// The first exported row.
    pub first_row: usize,
    /// The exported shape: rows, then each row's dimensions.
    pub shape: &'a [usize],
}

/// Why an export was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportError {
    /// The storage has no such field.
    NoField,
    /// The field's region is larger than memory can address, or its type is
    /// unknown to this crate.
    Region,
    /// The rows and shape do not fit the region exactly.
    Layout,
    /// The export cannot be described to a consumer (`Py_ssize_t` sizes,
    /// `int` dimensions).
    Undescribable,
}

/// What keeps an export's memory alive.
enum Owner {
    Output(#[expect(dead_code, reason = "held for its memory")] Arc<OutputStorage>),
    Gather(#[expect(dead_code, reason = "held for its memory")] Arc<GatherStorage>),
}

struct Inner {
    /// Keeps `data` alive; owns memory, never threads.
    _owner: Owner,
    /// Inside `_owner`'s region, `bytes` long (checked in `Export::new`).
    data: *const u8,
    bytes: usize,
    element: Element,
    shape: Box<[isize]>,
    strides: Box<[isize]>,
    read_only: bool,
}

// SAFETY: `data` points into the storage `_owner` keeps alive, and is only
// read through (by the consumers of the export, under SPEC 5.3's borrow
// window); every other field is plain data.
unsafe impl Send for Inner {}
// SAFETY: as above; nothing is written through `&Inner`.
unsafe impl Sync for Inner {}

/// A read-only export of output rows: cheap to clone.
#[derive(Clone)]
pub struct Export(Arc<Inner>);

impl Export {
    /// An export of `rows` of `field` in the engine's output storage.
    ///
    /// For a raw byte region (an observation leaf), the caller names the
    /// leaf's element: a wrong name gives wrong values, never memory
    /// outside the region. Bounds and alignment are always checked.
    ///
    /// # Errors
    ///
    /// The field is absent, or the rows and shape do not fit its region
    /// exactly ([`ExportError`]).
    pub fn output(
        storage: &Arc<OutputStorage>,
        field: OutputField,
        rows: Rows<'_>,
        element: Element,
        read_only: bool,
    ) -> Result<Self, ExportError> {
        let region = storage.region(field).ok_or(ExportError::NoField)?;
        if !element_fits(region.dtype, element) {
            return Err(ExportError::Layout);
        }
        let bytes = dtype_itemsize(region.dtype)
            .and_then(|size| region.len.checked_mul(size))
            .ok_or(ExportError::Region)?;
        Self::new(
            Owner::Output(Arc::clone(storage)),
            region.ptr,
            bytes,
            rows,
            element,
            read_only,
        )
    }

    /// An export of `rows` of `field` in the gather storage.
    ///
    /// # Errors
    ///
    /// As [`Export::output`].
    pub fn gather(
        storage: &Arc<GatherStorage>,
        field: OutputField,
        rows: Rows<'_>,
        element: Element,
        read_only: bool,
    ) -> Result<Self, ExportError> {
        let (start, bytes) = storage.region(field).ok_or(ExportError::NoField)?;
        Self::new(
            Owner::Gather(Arc::clone(storage)),
            start,
            bytes,
            rows,
            element,
            read_only,
        )
    }

    fn new(
        owner: Owner,
        start: *const u8,
        region_bytes: usize,
        rows: Rows<'_>,
        element: Element,
        read_only: bool,
    ) -> Result<Self, ExportError> {
        // A 0-d export would describe one element at `data` with a length
        // of 0 bytes: refused.
        if rows.shape.is_empty() {
            return Err(ExportError::Layout);
        }
        let count = rows.shape.first().copied().unwrap_or(0);
        let (offset, bytes) = rows
            .shape
            .iter()
            .skip(1)
            .try_fold(element.itemsize(), |acc, &d| acc.checked_mul(d))
            .and_then(|row_bytes| {
                row_window(
                    region_bytes,
                    row_bytes,
                    rows.total_rows,
                    rows.first_row,
                    count,
                )
            })
            .ok_or(ExportError::Layout)?;
        let undescribable = |_| ExportError::Undescribable;
        let shape: Box<[isize]> = rows
            .shape
            .iter()
            .map(|&d| isize::try_from(d).map_err(undescribable))
            .collect::<Result<_, _>>()?;
        // Row-major strides, in bytes.
        let mut strides = vec![0isize; shape.len()];
        let mut acc = isize::try_from(element.itemsize()).map_err(undescribable)?;
        for (stride, &dim) in strides.iter_mut().zip(shape.iter()).rev() {
            *stride = acc;
            acc = acc.checked_mul(dim).ok_or(ExportError::Undescribable)?;
        }
        // The buffer protocol describes sizes and dimensions as
        // `Py_ssize_t` and `int`: refuse what `getbuffer` could not.
        isize::try_from(bytes).map_err(undescribable)?;
        c_int::try_from(shape.len()).map_err(undescribable)?;
        // Every element is aligned to its size (rows are whole elements, so
        // the first one decides): consumers may read them as typed values.
        if start.wrapping_add(offset).addr() % element.itemsize() != 0 {
            return Err(ExportError::Layout);
        }
        Ok(Self(Arc::new(Inner {
            _owner: owner,
            // `row_window` checked `offset + bytes <= region_bytes`, so this
            // stays inside the region; `wrapping_add` keeps it safe code.
            data: start.wrapping_add(offset),
            bytes,
            element,
            shape,
            strides: strides.into_boxed_slice(),
            read_only,
        })))
    }

    /// The exported bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.0.bytes
    }

    /// The element type.
    #[must_use]
    pub fn element(&self) -> Element {
        self.0.element
    }

    /// The exported shape.
    #[must_use]
    pub fn shape(&self) -> &[isize] {
        &self.0.shape
    }

    /// A DLPack capsule of the export: versioned (DLPack 1.x, with the
    /// export's read-only flag) or legacy (no flag: its consumer gets a
    /// writable alias).
    ///
    /// # Errors
    ///
    /// `BufferError` if the export cannot be described; an allocation
    /// failure.
    pub fn dlpack<'py>(&self, g: Gil<'py>, versioned: bool) -> CResult<Mine<'py>> {
        let owner: dlpack::Owner = self.0.clone();
        let src = dlpack::Source {
            owner: &owner,
            data: self.0.data,
            dl_code: self.0.element.dl_code(),
            itemsize: self.0.element.itemsize(),
            shape: &self.0.shape,
        };
        dlpack::capsule(g, &src, versioned, self.0.read_only)
    }

    /// `__dlpack__(*, stream=None, max_version=None, dl_device=None,
    /// copy=None)` of a borrowed CPU view (SPEC 5.3 rule 6): a versioned,
    /// read-only capsule when the consumer accepts DLPack 1.x, legacy
    /// otherwise. A stream other than -1, a device other than
    /// `(kDLCPU, 0)` or `copy=True` raises `BufferError`.
    ///
    /// # Errors
    ///
    /// `TypeError` for positional arguments, an unknown keyword or an
    /// argument of the wrong type;
    /// `BufferError` as above or from [`Export::dlpack`]. `copy_hint` ends
    /// the `copy=True` message (the first-party views point to
    /// `outputs="owned"`).
    pub fn dlpack_call<'py>(
        &self,
        g: Gil<'py>,
        args: Ref<'py>,
        kwargs: Option<Ref<'py>>,
        copy_hint: &str,
    ) -> CResult<Mine<'py>> {
        if !capi::tuple_items(g, args, "the arguments")?.is_empty() {
            return Err(capi::raise(
                g,
                capi::type_error(),
                "__dlpack__ takes keyword arguments only",
            ));
        }
        let kw = match kwargs {
            Some(d) => capi::dict_items(g, d, "keyword arguments")?,
            None => Vec::new(),
        };
        if let Some((key, _)) = kw
            .iter()
            .find(|(k, _)| !["stream", "max_version", "dl_device", "copy"].contains(&k.as_str()))
        {
            return Err(capi::raise(
                g,
                capi::type_error(),
                &format!("__dlpack__ got an unexpected keyword argument {key:?}"),
            ));
        }
        let get = |key: &str| {
            kw.iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_ref())
                .filter(|v| !v.is_none())
        };
        if let Some(stream) = get("stream")
            && capi::to_i64(g, stream, "stream")? != -1
        {
            return Err(capi::raise(
                g,
                capi::buffer_error(),
                "CPU outputs take no stream",
            ));
        }
        if let Some(device) = get("dl_device")
            && int_pair(g, device, "dl_device")? != (1, 0)
        {
            return Err(capi::raise(
                g,
                capi::buffer_error(),
                "outputs live on the CPU (kDLCPU, 0)",
            ));
        }
        if let Some(copy) = get("copy")
            && capi::to_bool(g, copy, "copy")?
        {
            return Err(capi::raise(
                g,
                capi::buffer_error(),
                &format!("borrowed views do not copy{copy_hint}"),
            ));
        }
        let versioned = match get("max_version") {
            Some(max) => int_pair(g, max, "max_version")?.0 >= 1,
            None => false,
        };
        self.dlpack(g, versioned)
    }
}

/// An exact `(int, int)` tuple.
fn int_pair(g: Gil<'_>, v: Ref<'_>, what: &str) -> CResult<(i64, i64)> {
    match capi::tuple_items(g, v, what)?.as_slice() {
        [a, b] => Ok((capi::to_i64(g, *a, what)?, capi::to_i64(g, *b, what)?)),
        _ => Err(capi::raise(
            g,
            capi::type_error(),
            &format!("{what} must be a pair of ints"),
        )),
    }
}

/// Whether `element` may describe a region of `dtype`: a typed region only
/// as its own type (`Bool8` also as `U8`), raw bytes (observation leaves,
/// typed by the pool layout) as any element.
const fn element_fits(dtype: OutputDtype, element: Element) -> bool {
    match dtype {
        OutputDtype::U8 => true,
        OutputDtype::Bool8 => matches!(element, Element::Bool | Element::U8),
        OutputDtype::F32 => matches!(element, Element::F32),
        OutputDtype::U32 => matches!(element, Element::U32),
        _ => false,
    }
}

/// Bytes per element of a region type; `None` for a type this crate does
/// not know (the export is then refused).
const fn dtype_itemsize(dtype: OutputDtype) -> Option<usize> {
    match dtype {
        OutputDtype::U8 | OutputDtype::Bool8 => Some(1),
        OutputDtype::F32 | OutputDtype::U32 => Some(4),
        _ => None,
    }
}

/// The byte offset and length of `rows` rows of `row_bytes` bytes from
/// `first_row`, in a region of exactly `total_rows` such rows; `None` if
/// they do not fit or the sizes overflow.
fn row_window(
    region_bytes: usize,
    row_bytes: usize,
    total_rows: usize,
    first_row: usize,
    rows: usize,
) -> Option<(usize, usize)> {
    if row_bytes.checked_mul(total_rows)? != region_bytes
        || first_row.checked_add(rows)? > total_rows
    {
        return None;
    }
    Some((
        row_bytes.checked_mul(first_row)?,
        row_bytes.checked_mul(rows)?,
    ))
}

/// The `bf_getbuffer` thunk of `B`.
pub(super) const fn getbuffer_thunk<B: Buffer>() -> ThunkFn<GetBuffer> {
    ThunkFn::<GetBuffer>::of::<GetBufferShim<B>>()
}

/// The `bf_releasebuffer` thunk of `T`'s type.
pub(super) const fn releasebuffer_thunk<T: Body>() -> ThunkFn<ReleaseBuffer> {
    ThunkFn::<ReleaseBuffer>::of::<ReleaseShim<T>>()
}

/// The `bf_getbuffer` shim: a read-only, C-contiguous export. A request for
/// a writable buffer, or for Fortran order the export does not have,
/// raises `BufferError` (with `obj` null).
struct GetBufferShim<B>(PhantomData<B>);

impl<B: Buffer> GetBufferEntry for GetBufferShim<B> {
    unsafe extern "C" fn call(slf: Obj, out: *mut ffi::Py_buffer, flags: c_int) -> c_int {
        let done = entry_status(|g| {
            // SAFETY: CPython passes an instance of the type whose buffer
            // slot this is, set only by `TypeSpec::<B::Body>::with_buffer`.
            let body = unsafe { instance_body::<B::Body>(g, slf) }?;
            let export = B::export(g, body)?;
            fill(g, slf, out, flags, &export)
        });
        if done != 0 {
            if !out.is_null() {
                // SAFETY: on failure the protocol requires `obj` to be null.
                unsafe { (*out).obj = ptr::null_mut() };
            }
            -1
        } else {
            0
        }
    }
}

/// Fills `out` from `export`, keeping a clone of the export in `internal`
/// and a reference to the exporter in `obj`.
fn fill(
    g: Gil<'_>,
    slf: Obj,
    out: *mut ffi::Py_buffer,
    flags: c_int,
    export: &Export,
) -> CResult<()> {
    let e = &export.0;
    if flags & ffi::PyBUF_WRITABLE != 0 {
        return Err(capi::raise(
            g,
            capi::buffer_error(),
            "borrowed outputs are read-only",
        ));
    }
    if out.is_null() {
        return Err(capi::raise(g, capi::buffer_error(), "no buffer to fill"));
    }
    let undescribable = || capi::raise(g, capi::buffer_error(), "view too large to export");
    let len = isize::try_from(e.bytes).map_err(|_| undescribable())?;
    let itemsize = isize::try_from(e.element.itemsize()).map_err(|_| undescribable())?;
    let ndim = c_int::try_from(e.shape.len()).map_err(|_| undescribable())?;
    // Exports are C-contiguous, which satisfies C and ANY requests. They are
    // also F-contiguous only when at most one dimension exceeds 1 (or the
    // export is empty): refuse an F request otherwise.
    let f_contiguous = e.shape.contains(&0) || e.shape.iter().filter(|&&d| d > 1).count() <= 1;
    if flags & ffi::PyBUF_F_CONTIGUOUS == ffi::PyBUF_F_CONTIGUOUS && !f_contiguous {
        return Err(capi::raise(
            g,
            capi::buffer_error(),
            "borrowed outputs are C-contiguous, not Fortran-contiguous",
        ));
    }
    let nd = flags & ffi::PyBUF_ND == ffi::PyBUF_ND;
    let strided = flags & ffi::PyBUF_STRIDES == ffi::PyBUF_STRIDES;
    // SAFETY: `slf` is live for the call (CPython's argument).
    let exporter = capi::incref(
        unsafe { Ref::from_ptr(slf) }
            .ok_or_else(|| capi::raise(g, capi::buffer_error(), "no exporter"))?,
    );
    // The clone keeps the memory, shape and strides alive until
    // `bf_releasebuffer`.
    let internal = Arc::into_raw(Arc::clone(&export.0));
    let buffer = ffi::Py_buffer {
        buf: e.data.cast_mut().cast(),
        obj: exporter,
        len,
        itemsize,
        readonly: 1,
        ndim: if nd { ndim } else { 1 },
        format: if flags & ffi::PyBUF_FORMAT != 0 {
            e.element.format().as_ptr().cast_mut()
        } else {
            ptr::null_mut()
        },
        shape: if nd {
            e.shape.as_ptr().cast_mut()
        } else {
            ptr::null_mut()
        },
        strides: if strided {
            e.strides.as_ptr().cast_mut()
        } else {
            ptr::null_mut()
        },
        suboffsets: ptr::null_mut(),
        internal: internal.cast_mut().cast::<c_void>(),
    };
    // SAFETY: `out` is the caller's non-null Py_buffer to fill.
    unsafe { out.write(buffer) };
    Ok(())
}

/// The `bf_releasebuffer` shim: a cleanup entry. It drops the export clone
/// `getbuffer` stored; CPython then releases `obj`.
struct ReleaseShim<T>(PhantomData<T>);

impl<T: Body> ReleaseBufferEntry for ReleaseShim<T> {
    unsafe extern "C" fn call(_slf: Obj, view: *mut ffi::Py_buffer) {
        if view.is_null() {
            return;
        }
        // SAFETY: the consumer's view, filled by `getbuffer`, released once.
        let internal = unsafe { (*view).internal };
        if internal.is_null() {
            return;
        }
        // SAFETY: as above; cleared before the clone is dropped.
        unsafe { (*view).internal = ptr::null_mut() };
        // SAFETY: the clone `fill` leaked with `Arc::into_raw`, dropped once.
        let export = unsafe { Arc::from_raw(internal.cast_const().cast::<Inner>()) };
        if let Err(payload) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(export)))
        {
            super::dispose(payload);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Moved from the bindings' view module with the export code.
    #[test]
    fn test_row_window_stays_inside_the_region() {
        let region = [0u8; 24];
        for first in 0..=4 {
            for rows in 0..=4 - first {
                let (offset, len) = row_window(24, 6, 4, first, rows).unwrap_or((usize::MAX, 0));
                assert_eq!(
                    region.get(offset..offset + len).map(<[u8]>::len),
                    Some(rows * 6)
                );
            }
        }
        assert_eq!(row_window(24, 6, 4, 3, 2), None);
        assert_eq!(row_window(24, 6, 4, usize::MAX, 1), None);
        assert_eq!(row_window(25, 6, 4, 0, 1), None);
        assert_eq!(row_window(usize::MAX, usize::MAX, 2, 0, 1), None);
        assert_eq!(row_window(0, 0, 4, 1, 2), Some((0, 0)));
    }

    #[test]
    fn test_row_window_rows_inside_the_region_their_offset_and_length() {
        assert_eq!(row_window(48, 12, 4, 1, 2), Some((12, 24)));
        assert_eq!(row_window(48, 12, 4, 3, 1), Some((36, 12)));
        assert_eq!(row_window(48, 12, 4, 0, 0), Some((0, 0)));
    }

    #[test]
    fn test_row_window_rows_past_the_end_or_a_wrong_layout_none() {
        assert_eq!(row_window(48, 12, 4, 3, 2), None);
        assert_eq!(row_window(48, 12, 3, 0, 1), None);
        assert_eq!(row_window(48, 16, 4, 0, 1), None);
        assert_eq!(row_window(usize::MAX, usize::MAX, 2, 0, 1), None);
        assert_eq!(row_window(0, 0, usize::MAX, usize::MAX, 1), None);
    }

    #[test]
    fn test_export_gather_checks_rows_and_shape_against_the_region() {
        let writer = super::super::gather::GatherWriter::new(&[(OutputField::Rewards, 32)]);
        let storage = writer.storage();
        let rows = |first_row, shape| Rows {
            total_rows: 4,
            first_row,
            shape,
        };
        let e = Export::gather(
            &storage,
            OutputField::Rewards,
            rows(1, &[2, 2]),
            Element::F32,
            true,
        )
        .map(|e| (e.bytes(), e.shape().to_vec(), e.element()));
        assert_eq!(e.ok(), Some((16, vec![2, 2], Element::F32)));
        // Past the end, a row of the wrong width, an absent field.
        let err = |first, shape, field| {
            Export::gather(&storage, field, rows(first, shape), Element::F32, true).err()
        };
        assert_eq!(
            err(3, &[2, 2], OutputField::Rewards),
            Some(ExportError::Layout)
        );
        assert_eq!(
            err(0, &[1, 3], OutputField::Rewards),
            Some(ExportError::Layout)
        );
        assert_eq!(
            err(0, &[1, 2], OutputField::Alive),
            Some(ExportError::NoField)
        );
    }

    #[test]
    fn test_export_refuses_an_empty_shape() {
        // Rows are whole elements, so only a region's base could misalign an
        // export; the gather storage's base is 8-byte aligned (u64 words).
        let writer = super::super::gather::GatherWriter::new(&[(OutputField::Rewards, 32)]);
        let storage = writer.storage();
        let empty = Rows {
            total_rows: 4,
            first_row: 4,
            shape: &[],
        };
        assert!(Export::gather(&storage, OutputField::Rewards, empty, Element::F32, true).is_err());
        let empty = Rows {
            total_rows: 4,
            first_row: 0,
            shape: &[],
        };
        assert!(Export::gather(&storage, OutputField::Rewards, empty, Element::F32, true).is_err());
        let wide = Rows {
            total_rows: 4,
            first_row: 1,
            shape: &[1, 1],
        };
        assert!(Export::gather(&storage, OutputField::Rewards, wide, Element::F64, true).is_ok());
    }

    #[test]
    fn test_element_fits_only_its_region_type_and_raw_bytes() {
        assert!(element_fits(OutputDtype::F32, Element::F32));
        assert!(!element_fits(OutputDtype::F32, Element::F64));
        assert!(!element_fits(OutputDtype::U32, Element::I32));
        assert!(element_fits(OutputDtype::Bool8, Element::Bool));
        assert!(!element_fits(OutputDtype::U32, Element::Bool));
        assert!(element_fits(OutputDtype::U8, Element::F64));
    }

    #[test]
    fn test_element_itemsize_matches_its_format_and_code() {
        let all = [
            Element::Bool,
            Element::I8,
            Element::I16,
            Element::I32,
            Element::I64,
            Element::U8,
            Element::U16,
            Element::U32,
            Element::F32,
            Element::F64,
        ];
        for e in all {
            let size = match e.format().to_bytes().last() {
                Some(b'?' | b'b' | b'B') => 1,
                Some(b'h' | b'H') => 2,
                Some(b'i' | b'I' | b'f') => 4,
                Some(b'q' | b'd') => 8,
                _ => 0,
            };
            assert_eq!(e.itemsize(), size, "{e:?}");
        }
    }
}
