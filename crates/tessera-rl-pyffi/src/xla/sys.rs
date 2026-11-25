// SPDX-License-Identifier: MIT OR Apache-2.0

//! `#[repr(C)]` bindings for the subset of the XLA FFI C API 0.3 that the
//! handler uses (ADR-0095 point 4), written from `xla/ffi/api/c_api.h` of
//! the pinned `jaxlib==0.10.2` (`reviews/369/xla-ffi-0.10.2/c_api.h`,
//! sha256 `85fc385c`). Nothing is linked: XLA passes the API table in each
//! call frame.
//!
//! C enums are `int` on every supported target. They are modelled as `i32`
//! with constants, never as Rust enums: a value XLA adds later must not be
//! undefined behaviour here. The layout tests pin every size and offset
//! that the handler relies on, for 64-bit targets.

#![expect(
    missing_docs,
    reason = "every field mirrors the C header one to one; the header documents them"
)]

use core::ffi::{c_char, c_void};

/// The API major version these bindings were written against.
pub const API_MAJOR: i32 = 0;
/// The API minor version these bindings were written against.
pub const API_MINOR: i32 = 3;

/// `XLA_FFI_Extension_Type`.
pub type ExtensionType = i32;
/// `XLA_FFI_Extension_Metadata`.
pub const EXTENSION_METADATA: ExtensionType = 1;

/// `XLA_FFI_Error_Code`.
pub type ErrorCode = i32;
/// `XLA_FFI_Error_Code_INVALID_ARGUMENT`.
pub const ERROR_INVALID_ARGUMENT: ErrorCode = 3;
/// `XLA_FFI_Error_Code_FAILED_PRECONDITION`.
pub const ERROR_FAILED_PRECONDITION: ErrorCode = 9;
/// `XLA_FFI_Error_Code_UNIMPLEMENTED`.
pub const ERROR_UNIMPLEMENTED: ErrorCode = 12;
/// `XLA_FFI_Error_Code_INTERNAL`.
pub const ERROR_INTERNAL: ErrorCode = 13;

/// `XLA_FFI_DataType`.
pub type DataType = i32;
/// `XLA_FFI_DataType_PRED`.
pub const DTYPE_PRED: DataType = 1;
/// `XLA_FFI_DataType_S8`.
pub const DTYPE_S8: DataType = 2;
/// `XLA_FFI_DataType_S32`.
pub const DTYPE_S32: DataType = 4;
/// `XLA_FFI_DataType_S64`.
pub const DTYPE_S64: DataType = 5;
/// `XLA_FFI_DataType_U8`.
pub const DTYPE_U8: DataType = 6;
/// `XLA_FFI_DataType_U32`.
pub const DTYPE_U32: DataType = 8;
/// `XLA_FFI_DataType_U64`.
pub const DTYPE_U64: DataType = 9;
/// `XLA_FFI_DataType_F32`.
pub const DTYPE_F32: DataType = 11;
/// `XLA_FFI_DataType_F64`.
pub const DTYPE_F64: DataType = 12;

/// `XLA_FFI_ArgType`.
pub type ArgType = i32;
/// `XLA_FFI_ArgType_BUFFER`.
pub const ARG_BUFFER: ArgType = 1;
/// `XLA_FFI_RetType`.
pub type RetType = i32;
/// `XLA_FFI_RetType_BUFFER`.
pub const RET_BUFFER: RetType = 1;
/// `XLA_FFI_AttrType`.
pub type AttrType = i32;
/// `XLA_FFI_AttrType_SCALAR`.
pub const ATTR_SCALAR: AttrType = 3;

/// `XLA_FFI_ExecutionStage`.
pub type ExecutionStage = i32;
/// `XLA_FFI_ExecutionStage_INSTANTIATE`.
pub const STAGE_INSTANTIATE: ExecutionStage = 0;
/// `XLA_FFI_ExecutionStage_EXECUTE`.
pub const STAGE_EXECUTE: ExecutionStage = 3;

/// `XLA_FFI_Handler_Traits`.
pub type HandlerTraits = u32;

/// `XLA_FFI_Error`, opaque.
#[repr(C)]
pub struct Error {
    _opaque: [u8; 0],
}

/// `XLA_FFI_Extension_Base`.
#[repr(C)]
pub struct ExtensionBase {
    pub struct_size: usize,
    pub kind: ExtensionType,
    pub next: *mut ExtensionBase,
}

/// `XLA_FFI_Api_Version`.
#[repr(C)]
pub struct ApiVersion {
    pub struct_size: usize,
    pub extension_start: *mut ExtensionBase,
    pub major_version: i32,
    pub minor_version: i32,
}

/// `XLA_FFI_Error_Create_Args`.
#[repr(C)]
pub struct ErrorCreateArgs {
    pub struct_size: usize,
    pub extension_start: *mut ExtensionBase,
    pub message: *const c_char,
    pub errc: ErrorCode,
}

/// `XLA_FFI_Error_Create`.
pub type ErrorCreate = unsafe extern "C" fn(args: *mut ErrorCreateArgs) -> *mut Error;

/// `XLA_FFI_Api`, up to the first function slot the handler calls. The
/// rest of the table is never read here.
#[repr(C)]
pub struct Api {
    pub struct_size: usize,
    pub extension_start: *mut ExtensionBase,
    pub api_version: ApiVersion,
    pub internal_api: *const c_void,
    pub error_create: Option<ErrorCreate>,
}

/// `XLA_FFI_Buffer`.
#[repr(C)]
pub struct Buffer {
    pub struct_size: usize,
    pub extension_start: *mut ExtensionBase,
    pub dtype: DataType,
    pub data: *mut c_void,
    pub rank: i64,
    pub dims: *mut i64,
}

/// `XLA_FFI_Args`.
#[repr(C)]
pub struct Args {
    pub struct_size: usize,
    pub extension_start: *mut ExtensionBase,
    pub size: i64,
    pub types: *mut ArgType,
    pub args: *mut *mut c_void,
}

/// `XLA_FFI_Rets`.
#[repr(C)]
pub struct Rets {
    pub struct_size: usize,
    pub extension_start: *mut ExtensionBase,
    pub size: i64,
    pub types: *mut RetType,
    pub rets: *mut *mut c_void,
}

/// `XLA_FFI_ByteSpan`.
#[repr(C)]
pub struct ByteSpan {
    pub ptr: *const c_char,
    pub len: usize,
}

/// `XLA_FFI_Scalar`.
#[repr(C)]
pub struct Scalar {
    pub dtype: DataType,
    pub value: *mut c_void,
}

/// `XLA_FFI_Attrs`.
#[repr(C)]
pub struct Attrs {
    pub struct_size: usize,
    pub extension_start: *mut ExtensionBase,
    pub size: i64,
    pub types: *mut AttrType,
    pub names: *mut *mut ByteSpan,
    pub attrs: *mut *mut c_void,
}

/// `XLA_FFI_CallFrame`.
#[repr(C)]
pub struct CallFrame {
    pub struct_size: usize,
    pub extension_start: *mut ExtensionBase,
    pub api: *const Api,
    pub ctx: *mut c_void,
    pub stage: ExecutionStage,
    pub args: Args,
    pub rets: Rets,
    pub attrs: Attrs,
    pub future: *mut c_void,
}

/// `XLA_FFI_TypeId`.
#[repr(C)]
pub struct TypeId {
    pub type_id: i64,
}

/// `XLA_FFI_Metadata`.
#[repr(C)]
pub struct Metadata {
    pub struct_size: usize,
    pub api_version: ApiVersion,
    pub traits: HandlerTraits,
    pub state_type_id: TypeId,
}

/// `XLA_FFI_Metadata_Extension`.
#[repr(C)]
pub struct MetadataExtension {
    pub extension_base: ExtensionBase,
    pub metadata: *mut Metadata,
}

/// `XLA_FFI_Handler`.
pub type Handler = unsafe extern "C" fn(call_frame: *mut CallFrame) -> *mut Error;

#[cfg(all(test, target_pointer_width = "64"))]
mod tests {
    use core::mem::{offset_of, size_of};

    use super::{
        Api, ApiVersion, Args, Attrs, Buffer, ByteSpan, CallFrame, ErrorCreateArgs, ExtensionBase,
        Metadata, MetadataExtension, Rets, Scalar,
    };

    /// Sizes and offsets of the pinned header on 64-bit targets: `size_t`
    /// and pointers are 8 bytes, enums and `int` 4, `int64_t` 8, and every
    /// field aligned to its size (the C rule both supported ABIs follow).
    #[test]
    fn test_sys_layouts_match_the_pinned_header() {
        assert_eq!(size_of::<ExtensionBase>(), 24);
        assert_eq!(offset_of!(ExtensionBase, next), 16);
        assert_eq!(size_of::<ApiVersion>(), 24);
        assert_eq!(offset_of!(ApiVersion, minor_version), 20);
        assert_eq!(size_of::<ErrorCreateArgs>(), 32);
        assert_eq!(offset_of!(ErrorCreateArgs, errc), 24);
        assert_eq!(offset_of!(Api, api_version), 16);
        assert_eq!(offset_of!(Api, internal_api), 40);
        assert_eq!(offset_of!(Api, error_create), 48);
        assert_eq!(size_of::<Buffer>(), 48);
        assert_eq!(offset_of!(Buffer, data), 24);
        assert_eq!(offset_of!(Buffer, dims), 40);
        assert_eq!(size_of::<Args>(), 40);
        assert_eq!(size_of::<Rets>(), 40);
        assert_eq!(size_of::<Attrs>(), 48);
        assert_eq!(size_of::<ByteSpan>(), 16);
        assert_eq!(size_of::<Scalar>(), 16);
        assert_eq!(offset_of!(Scalar, value), 8);
        assert_eq!(offset_of!(CallFrame, stage), 32);
        assert_eq!(offset_of!(CallFrame, args), 40);
        assert_eq!(offset_of!(CallFrame, rets), 80);
        assert_eq!(offset_of!(CallFrame, attrs), 120);
        assert_eq!(offset_of!(CallFrame, future), 168);
        assert_eq!(size_of::<CallFrame>(), 176);
        assert_eq!(offset_of!(Metadata, api_version), 8);
        assert_eq!(offset_of!(Metadata, traits), 32);
        assert_eq!(offset_of!(Metadata, state_type_id), 40);
        assert_eq!(size_of::<Metadata>(), 48);
        assert_eq!(offset_of!(MetadataExtension, metadata), 24);
        assert_eq!(size_of::<MetadataExtension>(), 32);
    }
}
