// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host protocol 1 (SPEC 14.4 draft, ADR-0062, ADR-0065): what a host
//! environment's wheel needs on top of the typed layer. Safe code only;
//! every C-API call goes through [`crate::ffi::capi`].
//!
//! - [`config`](crate::host::config): Python keyword arguments as a `ConfigSource`, decoded by
//!   the environment's `ConfigDecode` (one decoder for both front ends);
//! - [`factory`](crate::host::factory): `make` and the module of `export_host_env!`;
//! - [`object`](crate::host::object): the host object (SPEC 14.4.2), and its
//!   output views (SPEC 14.4.4).

pub mod config;
mod describe;
pub mod factory;
pub mod object;
mod views;
