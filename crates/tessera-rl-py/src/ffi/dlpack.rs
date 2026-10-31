// SPDX-License-Identifier: MIT OR Apache-2.0

//! DLPack exports of borrowed output views, written by hand against the
//! DLPack 1.x C ABI (decision D1: no `numpy` or `dlpack` crate).
//!
//! An export owns an `Arc` clone of the output storage (ADR-0003 rule 1),
//! so the memory outlives the environment until the consumer calls the
//! deleter, or the capsule is destroyed unconsumed. The deleters and the
//! capsule destructor are cleanup entries (ADR-0041): reachable only
//! through thunks, they never fail and never touch Python state, except
//! the capsule destructor, which preserves the pending exception.

use core::ffi::{CStr, c_void};
use core::ptr;
use std::sync::Arc;

use pyo3::ffi;

use super::capi::{self, CResult, Gil, Mine};
use super::{dispose, thunks};

/// `kDLCPU`.
const DL_CPU: i32 = 1;
/// `DLPACK_FLAG_BITMASK_READ_ONLY`.
const FLAG_READ_ONLY: u64 = 1;

/// The legacy capsule name, and its consumed form.
pub(crate) const LEGACY: &CStr = c"dltensor";
/// The versioned capsule name (DLPack 1.0).
pub(crate) const VERSIONED: &CStr = c"dltensor_versioned";

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
#[derive(Debug)]
pub(crate) struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

/// `DLManagedTensor` (pre-1.0 consumers).
#[repr(C)]
#[derive(Debug)]
pub(crate) struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<thunks::DlDeleter>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DLPackVersion {
    major: u32,
    minor: u32,
}

/// `DLManagedTensorVersioned` (DLPack 1.0).
#[repr(C)]
#[derive(Debug)]
pub(crate) struct DLManagedTensorVersioned {
    version: DLPackVersion,
    manager_ctx: *mut c_void,
    deleter: Option<thunks::DlDeleterVersioned>,
    flags: u64,
    dl_tensor: DLTensor,
}

/// What an export keeps alive: the memory owner, and the shape and strides
/// the tensor points to. They are raw boxes, so moving the export does not
/// retag them and the tensor's pointers into them stay valid.
#[derive(Debug)]
struct Keep {
    _owner: Owner,
    shape: *mut [i64],
    strides: *mut [i64],
}

impl Drop for Keep {
    fn drop(&mut self) {
        // SAFETY: both came from `Box::into_raw` in `keep` and are freed
        // only here, once.
        drop(unsafe { Box::from_raw(self.shape) });
        // SAFETY: as above.
        drop(unsafe { Box::from_raw(self.strides) });
    }
}

/// A legacy export; the managed tensor first, so a pointer to it is a
/// pointer to the whole export.
#[repr(C)]
#[derive(Debug)]
struct Legacy {
    managed: DLManagedTensor,
    keep: Keep,
}

/// A versioned export, laid out like [`Legacy`].
#[repr(C)]
#[derive(Debug)]
struct Versioned {
    managed: DLManagedTensorVersioned,
    keep: Keep,
}

/// What keeps an export's memory alive: the output storage in practice,
/// any owner in the interpreter-free tests.
pub(crate) type Owner = Arc<dyn core::any::Any + Send + Sync>;

/// The tensor of one view.
#[derive(Debug)]
pub(crate) struct Source<'a> {
    pub(crate) owner: &'a Owner,
    pub(crate) data: *const u8,
    pub(crate) dl_code: u8,
    pub(crate) itemsize: usize,
    pub(crate) shape: &'a [isize],
}

fn tensor(src: &Source<'_>, keep: &Keep) -> Option<DLTensor> {
    Some(DLTensor {
        data: src.data.cast_mut().cast(),
        device: DLDevice {
            device_type: DL_CPU,
            device_id: 0,
        },
        ndim: i32::try_from(keep.shape.len()).ok()?,
        dtype: DLDataType {
            code: src.dl_code,
            bits: u8::try_from(src.itemsize.checked_mul(8)?).ok()?,
            lanes: 1,
        },
        shape: keep.shape.cast::<i64>(),
        strides: keep.strides.cast::<i64>(),
        byte_offset: 0,
    })
}

fn keep(src: &Source<'_>) -> Option<Keep> {
    let shape: Box<[i64]> = src
        .shape
        .iter()
        .map(|&d| i64::try_from(d).ok())
        .collect::<Option<_>>()?;
    // Row-major strides, in elements.
    let mut strides = vec![0i64; shape.len()];
    let mut acc = 1i64;
    for (stride, &dim) in strides.iter_mut().zip(shape.iter()).rev() {
        *stride = acc;
        acc = acc.checked_mul(dim)?;
    }
    Some(Keep {
        _owner: Arc::clone(src.owner),
        shape: Box::into_raw(shape),
        strides: Box::into_raw(strides.into_boxed_slice()),
    })
}

/// A new export of `src` and its capsule name; `None` for a shape DLPack
/// cannot describe. The caller owns it until a deleter runs.
fn export(
    src: &Source<'_>,
    versioned: bool,
    read_only: bool,
) -> Option<(*mut c_void, &'static CStr)> {
    Some(if versioned {
        let keep = keep(src)?;
        let dl_tensor = tensor(src, &keep)?;
        let export = Box::into_raw(Box::new(Versioned {
            managed: DLManagedTensorVersioned {
                version: DLPackVersion { major: 1, minor: 0 },
                manager_ctx: ptr::null_mut(),
                deleter: Some(thunks::DL_DELETER_VERSIONED.get()),
                flags: if read_only { FLAG_READ_ONLY } else { 0 },
                dl_tensor,
            },
            keep,
        }));
        // SAFETY: `export` is the live Box just made.
        unsafe { (*export).managed.manager_ctx = export.cast() };
        (export.cast(), VERSIONED)
    } else {
        let keep = keep(src)?;
        let dl_tensor = tensor(src, &keep)?;
        let export = Box::into_raw(Box::new(Legacy {
            managed: DLManagedTensor {
                dl_tensor,
                manager_ctx: ptr::null_mut(),
                deleter: Some(thunks::DL_DELETER.get()),
            },
            keep,
        }));
        // SAFETY: `export` is the live Box just made.
        unsafe { (*export).managed.manager_ctx = export.cast() };
        (export.cast(), LEGACY)
    })
}

/// A capsule exporting `src`: versioned (with the read-only flag when
/// `read_only`) or legacy. A legacy capsule cannot carry the flag, so its
/// consumer gets a writable alias: writing through it corrupts the outputs
/// until the next call overwrites them.
pub(crate) fn capsule(g: Gil, src: &Source<'_>, versioned: bool, read_only: bool) -> CResult<Mine> {
    let Some((raw, name)) = export(src, versioned, read_only) else {
        return Err(capi::raise(
            g,
            capi::buffer_error(),
            "the view's shape does not fit DLPack",
        ));
    };
    // SAFETY: `raw` is a live export valid for the destructor, `name` a
    // static C string, and the destructor a cleanup thunk.
    let cap = unsafe { capi::capsule(g, raw, name, thunks::CAPSULE_DESTRUCTOR.get()) };
    if cap.is_err() {
        // The capsule does not own the export: free it here.
        if versioned {
            // SAFETY: `raw` is the versioned export made above, unshared.
            drop(unsafe { Box::from_raw(raw.cast::<Versioned>()) });
        } else {
            // SAFETY: `raw` is the legacy export made above, unshared.
            drop(unsafe { Box::from_raw(raw.cast::<Legacy>()) });
        }
    }
    cap
}

/// `DLManagedTensor::deleter`: a cleanup entry. Consumers may call it from
/// any thread, attached or not; it only drops Rust values.
pub(super) unsafe extern "C" fn delete_legacy(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    // SAFETY: `managed` is the first field of a `Legacy` made by
    // `capsule`, and the DLPack contract calls the deleter once.
    let export = unsafe { Box::from_raw(managed.cast::<Legacy>()) };
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(export))) {
        dispose(payload);
    }
}

/// `DLManagedTensorVersioned::deleter`: as [`delete_legacy`].
pub(super) unsafe extern "C" fn delete_versioned(managed: *mut DLManagedTensorVersioned) {
    if managed.is_null() {
        return;
    }
    // SAFETY: as in `delete_legacy`, for a `Versioned`.
    let export = unsafe { Box::from_raw(managed.cast::<Versioned>()) };
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(export))) {
        dispose(payload);
    }
}

/// The capsule destructor: a cleanup entry. A capsule still carrying its
/// original name was never consumed, so its export is deleted here; a
/// consumer renames it (`used_dltensor`) and owns the deleter call.
pub(super) unsafe extern "C" fn destroy_capsule(cap: *mut ffi::PyObject) {
    // CPython calls capsule destructors attached; the pending exception,
    // if any, is saved and restored around the checks.
    let pending = capi::fetch_pending();
    // SAFETY: `cap` is a live capsule.
    if let Some(raw) = unsafe { capi::capsule_pointer(cap, VERSIONED) } {
        // SAFETY: an unconsumed versioned export, deleted once.
        unsafe { delete_versioned(raw.cast()) };
    // SAFETY: as above.
    } else if let Some(raw) = unsafe { capi::capsule_pointer(cap, LEGACY) } {
        // SAFETY: an unconsumed legacy export, deleted once.
        unsafe { delete_legacy(raw.cast()) };
    }
    capi::restore_pending(pending);
}

#[cfg(test)]
mod tests {
    //! The ownership code without an interpreter or a pool, so Miri can run
    //! it.

    use super::*;

    fn owner() -> (Owner, *const u8) {
        let data: Arc<[f32; 3]> = Arc::new([0.5, 1.5, 2.5]);
        let ptr = data.as_ptr().cast::<u8>();
        (data, ptr)
    }

    #[test]
    fn test_dlpack_exports_describe_the_view_and_deleters_free_them() {
        let (owner, data) = owner();
        let shape = [3isize, 1];
        let src = Source {
            owner: &owner,
            data,
            dl_code: 2,
            itemsize: 4,
            shape: &shape,
        };
        let before = Arc::strong_count(&owner);
        let (raw, name) = export(&src, true, true).unwrap();
        assert_eq!(name, VERSIONED);
        assert_eq!(Arc::strong_count(&owner), before + 1);
        let managed = raw.cast::<DLManagedTensorVersioned>();
        // SAFETY: a live versioned export just made.
        let m = unsafe { &*managed };
        assert_eq!((m.version.major, m.flags), (1, FLAG_READ_ONLY));
        assert_eq!(m.dl_tensor.ndim, 2);
        assert_eq!((m.dl_tensor.dtype.code, m.dl_tensor.dtype.bits), (2, 32));
        // SAFETY: `shape` points to `ndim` live elements.
        let dims = unsafe { core::slice::from_raw_parts(m.dl_tensor.shape, 2) };
        // SAFETY: as `strides`.
        let strides = unsafe { core::slice::from_raw_parts(m.dl_tensor.strides, 2) };
        assert_eq!((dims, strides), (&[3i64, 1][..], &[1i64, 1][..]));
        assert_eq!(m.dl_tensor.data.cast_const().cast::<u8>(), data);
        // SAFETY: the export is deleted once, as a consumer would.
        unsafe { delete_versioned(managed) };
        assert_eq!(Arc::strong_count(&owner), before);

        let (raw, name) = export(&src, false, true).unwrap();
        assert_eq!(name, LEGACY);
        assert_eq!(Arc::strong_count(&owner), before + 1);
        // SAFETY: a live legacy export, deleted once.
        unsafe { delete_legacy(raw.cast()) };
        assert_eq!(Arc::strong_count(&owner), before);
        // SAFETY: null is ignored.
        unsafe { delete_legacy(ptr::null_mut()) };
    }

    #[test]
    fn test_dlpack_export_keeps_its_memory_alive() {
        let (owner, data) = owner();
        let shape = [3isize];
        let src = Source {
            owner: &owner,
            data,
            dl_code: 2,
            itemsize: 4,
            shape: &shape,
        };
        let (raw, _) = export(&src, true, false).unwrap();
        drop(owner);
        let managed = raw.cast::<DLManagedTensorVersioned>();
        // SAFETY: the export is live.
        let ptr = unsafe { (*managed).dl_tensor.data };
        // SAFETY: the export's owner keeps the three `f32`s alive.
        let values = unsafe { core::slice::from_raw_parts(ptr.cast::<f32>(), 3) };
        assert_eq!(values, &[0.5, 1.5, 2.5]);
        // SAFETY: deleted once; this frees the last owner.
        unsafe { delete_versioned(managed) };
    }

    #[test]
    fn test_dlpack_export_refuses_a_shape_that_overflows() {
        let (owner, data) = owner();
        let shape = [isize::MAX, 4];
        let src = Source {
            owner: &owner,
            data,
            dl_code: 2,
            itemsize: 4,
            shape: &shape,
        };
        assert!(export(&src, true, true).is_none());
        assert_eq!(Arc::strong_count(&owner), 1);
    }
}
