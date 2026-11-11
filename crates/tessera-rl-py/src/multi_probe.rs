// SPDX-License-Identifier: MIT OR Apache-2.0

//! A test environment with two discrete heads, `MultiDiscrete([3, 5])`
//! (`make py-test-hooks`): the reward is `10 * head0 + head1` as received, so
//! a test reads back both heads. Choice 4 of head 1 is masked illegal (the
//! no-op `[0, 0]` replaces it), so the mask row and the fallback are visible
//! from Python. It exists to test the generic multi-head action path without
//! a real multi-head environment.

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Environment, EpisodeEnd, HeaderView, InvariantViolation, NoAdaptive,
    Observation, RecordHeader, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_num::rng::{ObsRng, StepRng};

use crate::split::common::{CountConfig, CountObs, Counts};

pub(crate) struct MultiProbe;

/// Choices per head.
const NVEC: [u32; 2] = [3, 5];

impl Environment for MultiProbe {
    type State = Counts;
    type Config = CountConfig;
    type Obs = CountObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        CountObs::space(cfg)
    }

    fn action_space(_: &CountConfig) -> Result<Space, SpaceError> {
        Ok(Space::MultiDiscrete {
            nvec: NVEC.to_vec(),
        })
    }

    fn reset(_: &CountConfig, _: &mut StepRng, state: &mut Counts) {
        *state = Counts::default();
    }

    #[expect(clippy::cast_precision_loss, reason = "small exact integers")]
    fn step(
        cfg: &CountConfig,
        state: &mut Counts,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let head = |k| actions.discrete(AgentSlot::FIRST, k).unwrap_or(-1);
        out.set_reward(AgentSlot::FIRST, (10 * head(0) + head(1)) as f32);
        state.steps += 1;
        if state.steps >= cfg.length {
            out.end_episode(EpisodeEnd::Truncated);
        }
    }

    fn observe(
        _: &CountConfig,
        _: &Counts,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, CountObs>,
    ) {
    }
}

impl ActionMask for MultiProbe {
    fn action_mask(_: &CountConfig, _: &Counts, _: AgentSlot, out: &mut MaskWriter<'_>) {
        // Head 0: 3 legal; head 1: 0 to 3 legal, 4 illegal.
        for k in 0..8 {
            out.set(k, k != 7);
        }
    }

    fn noop(_: &CountConfig, _: &Counts, _: AgentSlot, _: &mut ActionWriter<'_>) {}
}

impl tessera_rl_core::Persist for MultiProbe {
    const ENV_ID: &'static str = "tessera/MultiProbe-v1";
    const ENV_VERSION: u32 = 1;
}

impl Validate for MultiProbe {
    fn validate(_: &CountConfig, _: &Counts) -> Result<(), InvariantViolation> {
        Ok(())
    }

    fn validate_record(
        _: &CountConfig,
        _: &HeaderView<'_>,
        _: &Counts,
    ) -> Result<(), InvariantViolation> {
        Ok(())
    }
}
