// SPDX-License-Identifier: MIT OR Apache-2.0

//! Storage layout spike for ADR-0036 (phase 1a, M8): the current parallel
//! header and state arrays against combined records, SoA columns and AoSoA
//! blocks, on the real CartPole and Tag kernels. Outside the default
//! members; not part of the engine's API.

#![expect(
    clippy::doc_markdown,
    reason = "layout names (AoS, SoA, AoSoA) and CartPole are prose"
)]

pub mod columns;
pub mod kernel;
pub mod lanes;
pub mod layouts;
pub mod workload;
