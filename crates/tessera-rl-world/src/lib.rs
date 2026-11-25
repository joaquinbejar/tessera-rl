// SPDX-License-Identifier: MIT OR Apache-2.0

//! The massive multi-agent survival world (SPEC 16, phase 4):
//! `tessera/Survival-v1`. Environment ids and versions follow ADR-0023.

pub mod survival;

pub use survival::{Survival, SurvivalConfig, SurvivalObs, SurvivalState};
