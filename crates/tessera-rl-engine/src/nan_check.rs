// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `debug-checks` NaN trap (ADR-0055): rewards by agent slot, then the
//! state fields not annotated `#[state(allow_nan)]`.

use tessera_rl_core::encoding::Canonical;

use crate::error::{EngineError, NanSite};

/// The error for the first NaN of instance `instance`, given its reward
/// row (one reward per agent slot) and its state; `None` if there is none.
pub(crate) fn instance_nan<S: Canonical>(
    instance: u32,
    rewards: &[f32],
    state: &S,
) -> Option<EngineError> {
    if let Some(agent) = rewards.iter().position(|r| r.is_nan()) {
        return Some(EngineError::NanDetected {
            instance,
            site: NanSite::Reward {
                agent: u32::try_from(agent).unwrap_or(u32::MAX),
            },
        });
    }
    state.first_nan().map(|field| EngineError::NanDetected {
        instance,
        site: NanSite::State { field },
    })
}

/// The error for an allocation counted inside a chunk step, if any.
pub(crate) fn allocation_error(count: u64) -> Option<EngineError> {
    (count > 0).then_some(EngineError::AllocationInChunkStep { count })
}
