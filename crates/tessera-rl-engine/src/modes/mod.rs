// SPDX-License-Identifier: MIT OR Apache-2.0

//! The split-phase execution modes (ADR-0006, ADR-0047). Safe code: the
//! `unsafe` parts live in `buffer::pipeline`, `buffer::batch` and `sys`.

#[cfg(test)]
mod async_jitter_tests;
pub(crate) mod async_mode;
#[cfg(test)]
mod async_wake_tests;
pub(crate) mod pipelined;
