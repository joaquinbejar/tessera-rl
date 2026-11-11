// SPDX-License-Identifier: MIT OR Apache-2.0

//! Support for the `debug-checks` feature (SPEC 5.5, ADR-0055).
//!
//! A test binary installs [`TrackingAllocator`] as its global allocator to
//! have the engine fail any call whose chunk step allocated, with
//! [`crate::EngineError::AllocationInChunkStep`].

pub use crate::sys::alloc_track::TrackingAllocator;
