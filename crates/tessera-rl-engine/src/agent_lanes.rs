// SPDX-License-Identifier: MIT OR Apache-2.0

//! Whether a pool takes an environment's agent-lane step (ADR-0100
//! decision 1.6).

/// Which step a pool runs for an environment that sets
/// `Environment::AGENT_LANE` (ADR-0100 decision 1.6). It changes speed
/// only, never results: the agent-lane step is bit-identical to the
/// reference step. Ignored by environments without the hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum AgentLanes {
    /// The agent-lane step when the environment is eligible (it sets the
    /// hook, does not read adaptive values and keeps the Array-of-Structs
    /// store), at the pool's resolved SIMD level; the reference step
    /// otherwise.
    #[default]
    Auto,
    /// Always the reference step, `Environment::step_with`: the oracle of
    /// the agent-lane differentials and an opt-out.
    Reference,
}
