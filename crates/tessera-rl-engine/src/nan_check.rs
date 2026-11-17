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
    if let Some(e) = reward_nan(instance, rewards) {
        return Some(e);
    }
    state.first_nan().map(|field| EngineError::NanDetected {
        instance,
        site: NanSite::State { field },
    })
}

/// The error for the first NaN reward of instance `instance`, by agent
/// slot; `None` if there is none. Host pools check rewards only: they have
/// no state the engine can read (ADR-0062 item 4).
pub(crate) fn reward_nan(instance: u32, rewards: &[f32]) -> Option<EngineError> {
    rewards
        .iter()
        .position(|r| r.is_nan())
        .map(|agent| EngineError::NanDetected {
            instance,
            site: NanSite::Reward {
                agent: u32::try_from(agent).unwrap_or(u32::MAX),
            },
        })
}

/// The error for an allocation counted inside a chunk step, if any.
pub(crate) fn allocation_error(count: u64) -> Option<EngineError> {
    (count > 0).then_some(EngineError::AllocationInChunkStep { count })
}
