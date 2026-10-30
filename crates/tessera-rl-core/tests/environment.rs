// SPDX-License-Identifier: MIT OR Apache-2.0

//! A toy two-agent environment exercising the authoring traits.

use tessera_rl_core::encoding::{Canonical, Decoder, Encoder, Sink};
use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Bool8, DecodeError, EnvConfig, EnvState, Environment, EpisodeEnd,
    Heuristic, InvariantViolation, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{ObsRng, RngAuthority, StepRng};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
struct Count {
    value: [u16; 2],
    steps: u16,
    done: Bool8,
    pad: u8,
}

impl Canonical for Count {
    const ENCODED_LEN: usize = 8;

    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        self.value.encode(out);
        self.steps.encode(out);
        self.done.encode(out);
        self.pad.encode(out);
    }

    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        Ok(Self {
            value: Canonical::decode(input)?,
            steps: Canonical::decode(input)?,
            done: Canonical::decode(input)?,
            pad: Canonical::decode(input)?,
        })
    }
}

impl EnvState for Count {}

struct CountConfig {
    limit: u16,
}

impl EnvConfig for CountConfig {}

struct Counter;

const AGENTS: [AgentSlot; 2] = [
    AgentSlot::FIRST,
    match AgentSlot::new(1, 2) {
        Some(slot) => slot,
        None => AgentSlot::FIRST,
    },
];

impl Environment for Counter {
    type State = Count;
    type Config = CountConfig;
    type Obs = ();
    type Action = ();

    const MAX_AGENTS: usize = 2;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(_: &CountConfig) -> Result<Space, SpaceError> {
        let space = Space::Discrete { n: 256 };
        space.validate()?;
        Ok(space)
    }

    fn action_space(_: &CountConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(_: &CountConfig, rng: &mut StepRng, state: &mut Count) {
        *state = Count::default();
        if let Some(first) = state.value.first_mut() {
            *first = u16::from(rng.next_u32().is_multiple_of(2));
        }
    }

    fn step(
        cfg: &CountConfig,
        state: &mut Count,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        for agent in AGENTS {
            let inc = actions
                .discrete(agent, 0)
                .and_then(|a| u16::try_from(a).ok())
                .unwrap_or(0);
            if let Some(value) = state.value.get_mut(agent.index()) {
                *value = value.saturating_add(inc);
            }
            out.set_reward(agent, f32::from(inc));
        }
        state.steps += 1;
        if state.steps >= cfg.limit {
            state.done = Bool8::TRUE;
            out.end_episode(EpisodeEnd::Truncated);
        }
    }

    fn observe(
        _: &CountConfig,
        state: &Count,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, ()>,
    ) {
        let value = state.value.get(agent.index()).copied().unwrap_or(0);
        if let Some(byte) = out.leaf(0).and_then(|leaf| leaf.first_mut()) {
            *byte = u8::try_from(value).unwrap_or(u8::MAX);
        }
    }
}

impl ActionMask for Counter {
    fn action_mask(_: &CountConfig, state: &Count, _: AgentSlot, out: &mut MaskWriter<'_>) {
        out.fill(!state.done.get());
    }
}

impl Heuristic for Counter {
    fn act(_: &CountConfig, _: &Count, _: AgentSlot, _: &mut StepRng, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, 1);
    }
}

impl Validate for Counter {
    fn validate(cfg: &CountConfig, state: &Count) -> Result<(), InvariantViolation> {
        if state.steps > cfg.limit {
            return Err(InvariantViolation {
                code: "steps beyond limit",
            });
        }
        Ok(())
    }
}

#[test]
fn test_counter_environment_runs_one_episode() {
    let cfg = CountConfig { limit: 2 };
    let auth = RngAuthority::engine_only();
    let identity = RngIdentity::new(Seed::new(1), LogicalId::DEFAULT);
    let mut state = Count::default();
    Counter::reset(
        &cfg,
        &mut StepRng::new(&auth, &identity, StreamId::DYNAMICS),
        &mut state,
    );
    let start = state.value[0];

    let mut episode_end = None;
    for _ in 0..2 {
        let mut heads = [0i32; 2];
        for (agent, head) in AGENTS.iter().zip(heads.chunks_mut(1)) {
            let mut w = ActionWriter::new(head, &mut []);
            Counter::act(
                &cfg,
                &state,
                *agent,
                &mut StepRng::new(&auth, &identity, StreamId::OPPONENT),
                &mut w,
            );
        }
        let actions = Actions::new(&heads, 1, &[], 0);
        let mut rewards = [0.0; 2];
        let mut flags = [[Bool8::FALSE; 2]; 4];
        let [terminated, truncated, died, spawned] = &mut flags;
        let mut out = StepOut::new(&mut rewards, terminated, truncated, died, spawned);
        Counter::step(
            &cfg,
            &mut state,
            &actions,
            &mut StepRng::new(&auth, &identity, StreamId::DYNAMICS),
            &mut out,
        );
        episode_end = out.episode_end();
        assert_eq!(rewards, [1.0, 1.0]);
    }
    assert_eq!(episode_end, Some(EpisodeEnd::Truncated));
    assert_eq!(state.value, [start + 2, 2]);
    assert_eq!(Counter::validate(&cfg, &state), Ok(()));

    let mut leaf = [0u8; 1];
    let mut leaves = [&mut leaf[..]];
    let mut obs = ObsWriter::new(&mut leaves);
    Counter::observe(
        &cfg,
        &state,
        AGENTS[1],
        &mut ObsRng::new(&auth, &identity),
        &mut obs,
    );
    assert_eq!(leaf, [2]);

    let mut mask = [Bool8::TRUE; 2];
    Counter::action_mask(&cfg, &state, AGENTS[0], &mut MaskWriter::new(&mut mask));
    assert_eq!(mask, [Bool8::FALSE; 2]);
    assert!(Counter::observation_space(&cfg).is_ok());
    assert!(Counter::action_space(&cfg).is_ok());
}

#[test]
fn test_validate_reports_invariant_violation() {
    let cfg = CountConfig { limit: 1 };
    let state = Count {
        steps: 5,
        ..Count::default()
    };
    assert_eq!(
        Counter::validate(&cfg, &state),
        Err(InvariantViolation {
            code: "steps beyond limit"
        })
    );
}
