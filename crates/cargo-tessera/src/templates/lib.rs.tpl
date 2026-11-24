//! `{{env_id}}`: a Tessera host environment (ADR-0062), scaffolded by
//! `cargo tessera new`. Replace the counter game with your environment;
//! `cargo tessera check` keeps it deterministic and within its spaces.
//!
//! The game: {{agents}} agent(s) add 0, 1 or 2 to a shared counter. An
//! add is rewarded by its size and is legal only while the counter stays
//! at or below the target; reaching the target terminates the episode, and
//! the step limit truncates it. Integer-only and seeded by the engine's
//! `HostSeed`, so equal seeds and actions give equal outputs.

use tessera_rl_core::io::{Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, EpisodeEnd, HostEnvironment, HostError, HostOut, HostSeed, SpaceError,
};
{{^simultaneous}}
use tessera_rl_core::TurnStructure;
{{/simultaneous}}
{{#snapshots}}
use tessera_rl_core::{HostSnapshot, SnapshotReject};
{{/snapshots}}
use tessera_rl_macros::EnvConfig;

/// The environment id: `HostEnvironment::ENV_ID`, the registry key and the
/// Python entry point name.
pub const ENV_ID: &str = "{{env_id}}";

/// Agent slots.
pub const AGENTS: usize = {{agents}};

/// Choices per agent: add 0, 1 or 2.
const CHOICES: u32 = 3;

/// The bound of every observation value.
const BOUND: i64 = 1_000_000;

/// The configuration, set from Python as keyword arguments of
/// `tessera_rl.make`.
#[derive(EnvConfig, Debug, Clone, PartialEq)]
pub struct CounterConfig {
    /// Steps per episode; the episode is truncated at this step.
    #[config(default = 64, range = 1..=100_000)]
    pub episode: u32,
    /// The counter's target; reaching it terminates the episode.
    #[config(default = 40, range = 1..=100_000)]
    pub target: u32,
}

/// One instance. The moves live on the heap, as a host environment's state
/// may.
#[derive(Debug)]
pub struct Counter {
    cfg: CounterConfig,
    counter: u32,
    step: u32,
    /// Sequential: the slot that acts next. Mixed: the step's parity.
    turn: u32,
    /// Every applied add of the episode, oldest first.
    moves: Vec<u32>,
}

/// Agent slot `i`.
fn slot(i: usize) -> Option<AgentSlot> {
    AgentSlot::new(u16::try_from(i).ok()?, AGENTS)
}

impl HostEnvironment for Counter {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = {{env_version}};
    const MAX_AGENTS: usize = AGENTS;
    const MASKS: bool = true;
{{#sequential}}
    const TURNS: TurnStructure = TurnStructure::Sequential;
{{/sequential}}
{{#mixed}}
    const TURNS: TurnStructure = TurnStructure::Mixed;
{{/mixed}}
    type Config = CounterConfig;

    fn new(cfg: &CounterConfig) -> Result<Self, HostError> {
        Ok(Self {
            cfg: cfg.clone(),
            counter: 0,
            step: 0,
            turn: 0,
            moves: Vec::new(),
        })
    }

    /// Per agent: the counter, the step, the turn and the agent's slot.
    fn observation_space(_: &CounterConfig) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![4],
            dtype: DType::I32,
            bounds: BoxBounds::Int {
                low: vec![0],
                high: vec![BOUND],
            },
        })
    }

    /// One discrete head: the add.
    fn action_space(_: &CounterConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: CHOICES })
    }

    fn reset(&mut self, seed: HostSeed, _: &mut HostOut<'_>) {
        // The opening counter comes from the seed and the episode index,
        // never above the target (so holding is always legal).
        let opening = (seed.seed() ^ u64::from(seed.episode_index())) % 3;
        self.counter = u32::try_from(opening).unwrap_or(0).min(self.cfg.target);
        self.step = 0;
        self.turn = 0;
        self.moves.clear();
    }

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut HostOut<'_>) {
        self.step = self.step.saturating_add(1);
        for i in 0..AGENTS {
            let Some(agent) = slot(i) else {
                continue;
            };
            if !self.acting(agent) {
                continue;
            }
            let add = actions
                .discrete(agent, 0)
                .and_then(|a| u32::try_from(a).ok())
                .unwrap_or(0);
            // The engine checks ranges, not masks: an illegal add holds.
            let add = if self.counter.saturating_add(add) > self.cfg.target {
                0
            } else {
                add
            };
            self.counter = self.counter.saturating_add(add);
            self.moves.push(add);
            out.set_reward(agent, f32::from(u8::try_from(add).unwrap_or(0)));
        }
{{#sequential}}
        self.turn = self.turn.saturating_add(1) % u32::try_from(AGENTS).unwrap_or(1);
{{/sequential}}
{{#mixed}}
        self.turn = self.turn.saturating_add(1) % 2;
{{/mixed}}
        if self.counter >= self.cfg.target {
            out.end_episode(EpisodeEnd::Terminated);
        } else if self.step >= self.cfg.episode {
            out.end_episode(EpisodeEnd::Truncated);
        }
    }

    fn observe(&self, agent: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>) {
        let index = u32::try_from(agent.index()).unwrap_or(0);
        let values = [self.counter, self.step, self.turn, index];
        if let Some(leaf) = obs.leaf(0) {
            for (bytes, v) in leaf.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                let v = i32::try_from(v).unwrap_or(i32::MAX);
                bytes.copy_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn mask(&self, _: AgentSlot, mask: &mut MaskWriter<'_>) {
        for add in 0..CHOICES {
            let legal = self.counter.saturating_add(add) <= self.cfg.target;
            mask.set(usize::try_from(add).unwrap_or(0), legal);
        }
    }
{{#sequential}}

    /// One agent per step, in slot order.
    fn acting(&self, agent: AgentSlot) -> bool {
        u32::try_from(agent.index()).is_ok_and(|a| a == self.turn)
    }
{{/sequential}}
{{#mixed}}

    /// The agents whose slot has the step's parity: with one agent, every
    /// other step acts no one (an environment step).
    fn acting(&self, agent: AgentSlot) -> bool {
        u32::try_from(agent.index()).is_ok_and(|a| a % 2 == self.turn)
    }
{{/mixed}}
}

tessera_rl_core::register_host!(Counter, id = "{{env_id}}");
{{#snapshots}}

/// Bytes before the moves: counter, step, turn and move count (`u32` each).
const FIXED: usize = 16;

/// Snapshots (ADR-0073): the counter, the step, the turn and the moves,
/// little-endian. The configuration is not state.
impl HostSnapshot for Counter {
    const SNAPSHOT_VERSION: u32 = 1;

    fn max_snapshot_len(cfg: &CounterConfig) -> usize {
        let moves = usize::try_from(cfg.episode)
            .unwrap_or(0)
            .saturating_mul(AGENTS);
        FIXED.saturating_add(moves.saturating_mul(4))
    }

    fn save(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.counter.to_le_bytes());
        out.extend_from_slice(&self.step.to_le_bytes());
        out.extend_from_slice(&self.turn.to_le_bytes());
        let count = u32::try_from(self.moves.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&count.to_le_bytes());
        for m in &self.moves {
            out.extend_from_slice(&m.to_le_bytes());
        }
    }

    fn check_snapshot(cfg: &CounterConfig, bytes: &[u8]) -> Result<(), SnapshotReject> {
        let reject = |code, reason| Err(SnapshotReject { code, reason });
        let (Some(counter), Some(step), Some(turn), Some(count)) =
            (word(bytes, 0), word(bytes, 4), word(bytes, 8), word(bytes, 12))
        else {
            return reject(1, "shorter than the fixed fields");
        };
        if counter > cfg.target {
            return reject(2, "a counter past the target");
        }
        if step > cfg.episode {
            return reject(3, "a step past the episode length");
        }
        if turn >= u32::try_from(AGENTS.max(2)).unwrap_or(u32::MAX) {
            return reject(4, "a turn past the agents");
        }
        let count = usize::try_from(count).unwrap_or(usize::MAX);
        if count > usize::try_from(cfg.episode).unwrap_or(0).saturating_mul(AGENTS) {
            return reject(5, "more moves than the episode allows");
        }
        if bytes.len() != FIXED.saturating_add(count.saturating_mul(4)) {
            return reject(6, "the move count does not match the bytes");
        }
        if (0..count).any(|k| word(bytes, FIXED + 4 * k).is_none_or(|m| m >= CHOICES)) {
            return reject(7, "a move outside the choices");
        }
        Ok(())
    }

    fn load(&mut self, bytes: &[u8]) {
        self.counter = word(bytes, 0).unwrap_or(0);
        self.step = word(bytes, 4).unwrap_or(0);
        self.turn = word(bytes, 8).unwrap_or(0);
        let count = word(bytes, 12).map_or(0, |c| usize::try_from(c).unwrap_or(0));
        self.moves.clear();
        self.moves
            .extend((0..count).map(|k| word(bytes, FIXED + 4 * k).unwrap_or(0)));
    }
}

/// The `u32` at `at`.
fn word(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at.checked_add(4)?)?
        .try_into()
        .ok()
        .map(u32::from_le_bytes)
}
{{/snapshots}}

#[cfg(feature = "python")]
{{#snapshots}}
tessera_rl_pyffi::export_host_env!(
    Counter,
    module = c"{{py_package}}._tessera_host",
    snapshots
);
{{/snapshots}}
{{^snapshots}}
tessera_rl_pyffi::export_host_env!(Counter, module = c"{{py_package}}._tessera_host");
{{/snapshots}}
