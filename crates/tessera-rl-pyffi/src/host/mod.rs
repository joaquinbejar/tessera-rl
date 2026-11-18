// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host protocol 1 (SPEC 14.4 draft, ADR-0062, ADR-0065): what a host
//! environment's wheel needs on top of the typed layer. Safe code only;
//! every C-API call goes through [`crate::ffi::capi`].
//!
//! - [`config`]: Python keyword arguments as a `ConfigSource`, decoded by
//!   the environment's `ConfigDecode` (one decoder for both front ends);
//! - [`factory`]: `make` and the module of `export_host_env!`.

pub mod config;
pub mod factory;
