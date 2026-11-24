// SPDX-License-Identifier: MIT OR Apache-2.0

//! Golden suites (ADR-0009, ADR-0024 layer 2) and cross-thread runs (SPEC
//! 8.5) for JobShop-v1: every scenario matches its committed fixture in
//! Inline mode and with 1, 2, 7 and 64 workers. The harness is the one of
//! the classic crate.

#![expect(clippy::panic, reason = "test setup: a failure is the assertion")]

#[path = "../../tessera-rl-classic/tests/support/mod.rs"]
mod support;

use support::{Scenario, check, run};
use tessera_rl_engine::{AutoReset, EnvPool, Masked};
use tessera_rl_ops::{JobShop, JobShopConfig};

impl support::GoldenEnv for JobShop {
    const ENV_ID: &'static str = tessera_rl_ops::jobshop::ENV_ID;
    const ENV_VERSION: u32 = tessera_rl_ops::jobshop::ENV_VERSION;
}

const WORKERS: [Option<usize>; 5] = [None, Some(1), Some(2), Some(7), Some(64)];

fn jobshop(name: &'static str, auto_reset: AutoReset) -> Scenario<JobShopConfig> {
    Scenario {
        name,
        cfg: JobShopConfig {
            jobs: 5,
            machines: 4,
            max_duration: 20,
        },
        seed: 3_830_001,
        num_envs: 16,
        chunk: 4,
        auto_reset,
        // Above the 20 operations of an episode: the limit never truncates.
        max_episode_steps: 32,
        steps: 300,
        reseed_at: 150,
        reseed_ids: &[1, 12],
        reseed_seed: 4_444,
    }
}

fn check_jobshop(s: &Scenario<JobShopConfig>) {
    let build = |cfg, options| {
        EnvPool::<JobShop, Masked>::with_masks(cfg, options).unwrap_or_else(|e| panic!("{e}"))
    };
    for workers in WORKERS {
        check(s.name, &run::<JobShop, _>(s, build, workers));
    }
}

#[test]
fn test_golden_jobshop_next_step_matches_fixture_on_every_mode() {
    check_jobshop(&jobshop("jobshop-v1-next-step", AutoReset::NextStep));
}

#[test]
fn test_golden_jobshop_same_step_matches_fixture_on_every_mode() {
    check_jobshop(&jobshop("jobshop-v1-same-step", AutoReset::SameStep));
}
