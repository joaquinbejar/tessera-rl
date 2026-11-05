// SPDX-License-Identifier: MIT OR Apache-2.0

//! The split-phase execution modes (ADR-0006, ADR-0047). Safe code: the
//! `unsafe` parts live in `buffer::pipeline` and `sys`.

pub(crate) mod pipelined;
