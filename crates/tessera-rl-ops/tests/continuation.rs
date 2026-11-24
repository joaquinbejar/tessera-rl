// SPDX-License-Identifier: MIT OR Apache-2.0

//! Continuation tests (SPEC 9.1, normative; ADR-0010) for JobShop-v1, with
//! the harness of the classic crate. The environment draws only at reset
//! (the instance): branches diverge once an auto-reset generates a new
//! instance with the branched salt.

#[path = "../../tessera-rl-classic/tests/support/mod.rs"]
mod support;

use support::continuation::Case;
use tessera_rl_engine::{AutoReset, EnvPool, Masked};
use tessera_rl_ops::{JobShop, JobShopConfig};

#[expect(clippy::unwrap_used, reason = "test setup: a failure is the assertion")]
fn jobshop(auto_reset: AutoReset) -> Case<JobShop, Masked> {
    Case {
        cfg: JobShopConfig {
            jobs: 4,
            machines: 3,
            max_duration: 9,
        },
        build: |cfg, options| EnvPool::with_masks(cfg, options).unwrap(),
        auto_reset,
        limit: 32,
        deaths: false,
        branches: true,
    }
}

#[test]
fn test_continuation_jobshop_next_step_restores_identically_everywhere() {
    jobshop(AutoReset::NextStep).check_all(50);
}

#[test]
fn test_continuation_jobshop_same_step_restores_identically_everywhere() {
    jobshop(AutoReset::SameStep).check_all(50);
}
