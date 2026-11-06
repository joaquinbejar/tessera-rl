// SPDX-License-Identifier: MIT OR Apache-2.0

//! A test environment: `Counter`, whose `observe` panics on a thread that asked
//! for it. A caller-side writer such as `restore` observes on the calling
//! thread, and workers never see the switch. It is its own type so that `Counter`,
//! which the benches run, pays for no thread-local read per observation.

use core::cell::Cell;

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, HeaderView, InvariantViolation, NoAdaptive, RecordHeader, SpaceError,
    StepOut, Tier, Validate,
};
use tessera_rl_num::rng::{ObsRng, StepRng};

use crate::split::common::{CountConfig, CountObs, Counter, Counts};

/// The message of the panic [`PANIC_IN_OBSERVE`] injects.
pub(crate) const OBSERVE_PANIC: &str = "injected observe panic";

thread_local! {
    /// Panic in `observe` on this thread.
    pub(crate) static PANIC_IN_OBSERVE: Cell<bool> = const { Cell::new(false) };
}

pub(crate) struct ObservePanicCounter;

impl Environment for ObservePanicCounter {
    type State = Counts;
    type Config = CountConfig;
    type Obs = CountObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = Counter::MAX_AGENTS;
    const TIER: Tier = Counter::TIER;

    fn observation_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        Counter::observation_space(cfg)
    }

    fn action_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        Counter::action_space(cfg)
    }

    fn initial_agents(cfg: &CountConfig) -> usize {
        Counter::initial_agents(cfg)
    }

    fn reset(cfg: &CountConfig, rng: &mut StepRng, state: &mut Counts) {
        Counter::reset(cfg, rng, state);
    }

    fn step(
        cfg: &CountConfig,
        state: &mut Counts,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        Counter::step(cfg, state, actions, rng, out);
    }

    fn observe(
        cfg: &CountConfig,
        state: &Counts,
        agent: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, CountObs>,
    ) {
        assert!(!PANIC_IN_OBSERVE.with(Cell::get), "{OBSERVE_PANIC}");
        Counter::observe(cfg, state, agent, rng, out);
    }
}

impl tessera_rl_core::Persist for ObservePanicCounter {
    const ENV_ID: &'static str = "tessera/ObservePanic-v1";
    const ENV_VERSION: u32 = 1;
}

impl Validate for ObservePanicCounter {
    fn validate(cfg: &CountConfig, state: &Counts) -> Result<(), InvariantViolation> {
        Counter::validate(cfg, state)
    }

    fn validate_record(
        cfg: &CountConfig,
        header: &HeaderView<'_>,
        state: &Counts,
    ) -> Result<(), InvariantViolation> {
        Counter::validate_record(cfg, header, state)
    }
}
