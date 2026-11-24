// SPDX-License-Identifier: MIT OR Apache-2.0

//! A scripted host environment for the abort and validation tests
//! (ADR-0065): each agent's action picks what its `step` does, a reset can
//! be told to abort, and every call into the environment is counted per
//! instance, so the tests can assert what the engine called.

#![allow(dead_code, reason = "each test binary uses part of it")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use tessera_rl_core::config::{CanonicalConfig, write_entry};
use tessera_rl_core::encoding::{Encoder, Sink};
use tessera_rl_core::io::{Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AbortCode, ActionReject, AgentSlot, EnvConfig, EpisodeEnd, HostEnvironment, HostError, HostOut,
    HostSeed, HostSnapshot, SnapshotReject, SpaceError, TurnStructure,
};

/// Agent slots.
pub const SLOTS: usize = 3;
/// Most instances a test builds.
pub const MAX_INSTANCES: usize = 16;

/// What an agent's action makes its `step` do.
pub const NOTHING: i32 = 0;
/// The agent dies.
pub const DIE: i32 = 1;
/// The agent spawns slot 2.
pub const SPAWN: i32 = 2;
/// The step aborts with code `STEP_CODE + slot`.
pub const ABORT: i32 = 3;
/// The episode terminates.
pub const TERMINATE: i32 = 4;
/// The episode is truncated.
pub const TRUNCATE: i32 = 5;
/// The agent's reward is NaN.
pub const NAN: i32 = 6;
/// The episode terminates, then the step aborts (abort wins).
pub const TERMINATE_THEN_ABORT: i32 = 7;
/// The step aborts, then the episode terminates (abort wins).
pub const ABORT_THEN_TERMINATE: i32 = 8;
/// The agent dies twice (an invalid event), then the step aborts.
pub const BAD_EVENT_THEN_ABORT: i32 = 9;
/// The agent dies twice: an invalid agent event.
pub const BAD_EVENT: i32 = 10;
/// Choices of the action head.
pub const CHOICES: u32 = 11;

/// The abort code of a step aborted by agent `slot`.
pub const STEP_CODE: u32 = 10;
/// The abort code of a reset of instance `id`.
pub const RESET_CODE: u32 = 100;
/// The code `validate` rejects instance `id` with: `REJECT_CODE + id`.
pub const REJECT_CODE: u32 = 40;
/// The reward slot 0 gets from every reset.
pub const RESET_REWARD: f32 = 0.5;

/// Calls into the environment, per instance, shared by the tests.
#[derive(Debug, Default)]
pub struct Shared {
    next_id: AtomicU32,
    /// Resets that will abort, per instance (each abort uses one).
    pub reset_aborts: [AtomicU32; MAX_INSTANCES],
    pub steps: [AtomicU64; MAX_INSTANCES],
    pub resets: [AtomicU64; MAX_INSTANCES],
    pub observes: [AtomicU64; MAX_INSTANCES],
    pub masks: [AtomicU64; MAX_INSTANCES],
    pub actings: [AtomicU64; MAX_INSTANCES],
    pub validates: [AtomicU64; MAX_INSTANCES],
    /// The episode index of each instance's last reset.
    pub episode_index: [AtomicU32; MAX_INSTANCES],
    /// Mixed only: whether any agent acts.
    pub mixed_acting: AtomicBool,
    /// Instances whose `validate` rejects (bit `i`).
    pub reject: AtomicU32,
    /// Instances whose next `validate` rejects, once (bit `i`, cleared by
    /// that call): an impure `validate`, for the engine's re-run check.
    pub reject_once: AtomicU32,
    /// Resets write events before they decide whether to abort: slot 0
    /// terminated, slot 1 died, slot 2 spawned.
    pub reset_events: AtomicBool,
    /// Instances whose `validate` panics (bit `i`).
    pub panic_validate: AtomicU32,
    /// `validate` leaves flush-to-zero set.
    pub hostile_validate: AtomicBool,
    /// `validate` allocates.
    pub alloc_validate: AtomicBool,
    /// The thread of each `new` call, in call order.
    pub built_on: std::sync::Mutex<Vec<std::thread::ThreadId>>,
    /// `new` fails on its `k`-th call (1-based); 0: never.
    pub fail_new: AtomicU32,
    /// Instances dropped.
    pub drops: AtomicU32,
    /// Snapshot hooks (ADR-0073): calls of `save`, `check_snapshot` and
    /// `load`, all instances together.
    pub saves: AtomicU64,
    pub checks: AtomicU64,
    pub loads: AtomicU64,
    /// `save` panics.
    pub panic_save: AtomicBool,
    /// `save` writes one byte past `max_snapshot_len`.
    pub save_extra: AtomicBool,
    /// `check_snapshot` panics.
    pub panic_check: AtomicBool,
    /// `load` panics.
    pub panic_load: AtomicBool,
}

impl Shared {
    pub fn new() -> Arc<Self> {
        let s = Arc::new(Self::default());
        s.mixed_acting.store(true, Ordering::Relaxed);
        s
    }

    /// How many times `new` was called.
    pub fn built(&self) -> u32 {
        self.next_id.load(Ordering::Relaxed)
    }

    pub fn get(counts: &[AtomicU64; MAX_INSTANCES], i: usize) -> u64 {
        counts.get(i).map_or(0, |c| c.load(Ordering::Relaxed))
    }

    /// Instance `i`'s next `n` resets abort.
    pub fn abort_resets(&self, i: usize, n: u32) {
        if let Some(c) = self.reset_aborts.get(i) {
            c.store(n, Ordering::Relaxed);
        }
    }
}

/// The configuration: the shared counters and the agent count.
#[derive(Debug, Clone)]
pub struct ScriptConfig {
    pub shared: Arc<Shared>,
    pub initial: usize,
}

impl ScriptConfig {
    pub fn new(shared: &Arc<Shared>) -> Self {
        Self {
            shared: Arc::clone(shared),
            initial: SLOTS,
        }
    }
}

impl EnvConfig for ScriptConfig {}

/// The canonical form holds `initial` only: the shared counters are test
/// plumbing, not configuration.
impl CanonicalConfig for ScriptConfig {
    const ADAPTIVE_FIELDS: &'static [&'static str] = &[];
    const STATIC_ENTRIES: u16 = 1;

    fn write_entries<S: Sink>(&self, out: &mut Encoder<S>) {
        write_entry(
            out,
            "initial",
            &u64::try_from(self.initial).unwrap_or(u64::MAX),
        );
    }
}

/// One instance; `T` picks the turn structure (0 simultaneous, 1
/// sequential, 2 mixed) and `V` whether it validates joint actions.
#[derive(Debug)]
pub struct Script<const T: u8, const V: bool> {
    shared: Arc<Shared>,
    id: usize,
    step: u32,
    episode: u32,
}

fn slot(s: usize) -> AgentSlot {
    AgentSlot::new(u16::try_from(s).unwrap_or(0), SLOTS).unwrap()
}

fn code(n: u32) -> AbortCode {
    AbortCode::new(n).unwrap()
}

impl<const T: u8, const V: bool> Script<T, V> {
    fn count(&self, counts: &[AtomicU64; MAX_INSTANCES]) {
        if let Some(c) = counts.get(self.id) {
            c.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl<const T: u8, const V: bool> Drop for Script<T, V> {
    fn drop(&mut self) {
        self.shared.drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl<const T: u8, const V: bool> HostEnvironment for Script<T, V> {
    const ENV_ID: &'static str = "test/Script-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = SLOTS;
    const TURNS: TurnStructure = match T {
        0 => TurnStructure::Simultaneous,
        1 => TurnStructure::Sequential,
        _ => TurnStructure::Mixed,
    };
    const MASKS: bool = true;
    const SPAWNS: bool = true;
    const VALIDATES: bool = V;
    type Config = ScriptConfig;

    fn new(cfg: &ScriptConfig) -> Result<Self, HostError> {
        if let Ok(mut threads) = cfg.shared.built_on.lock() {
            threads.push(std::thread::current().id());
        }
        let call = cfg.shared.next_id.fetch_add(1, Ordering::Relaxed);
        if call.checked_add(1) == Some(cfg.shared.fail_new.load(Ordering::Relaxed)) {
            return Err(HostError::Construction {
                reason: "scripted failure".to_owned(),
            });
        }
        let id = usize::try_from(call).unwrap_or(0);
        Ok(Self {
            shared: Arc::clone(&cfg.shared),
            id,
            step: 0,
            episode: 0,
        })
    }

    fn initial_agents(cfg: &ScriptConfig) -> usize {
        cfg.initial
    }

    fn observation_space(_: &ScriptConfig) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![3],
            dtype: DType::I32,
            bounds: BoxBounds::Int {
                low: vec![1],
                high: vec![1_000_000],
            },
        })
    }

    fn action_space(_: &ScriptConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: CHOICES })
    }

    fn reset(&mut self, seed: HostSeed, out: &mut HostOut<'_>) {
        self.count(&self.shared.resets);
        if let Some(e) = self.shared.episode_index.get(self.id) {
            e.store(seed.episode_index(), Ordering::Relaxed);
        }
        self.step = 0;
        self.episode = seed.episode_index();
        out.set_reward(slot(0), RESET_REWARD);
        if self.shared.reset_events.load(Ordering::Relaxed) {
            out.terminate_agent(slot(0));
            out.agent_died(slot(1));
            out.agent_spawned(slot(2));
        }
        let left = self
            .shared
            .reset_aborts
            .get(self.id)
            .map_or(0, |c| c.load(Ordering::Relaxed));
        if left > 0 {
            if let Some(c) = self.shared.reset_aborts.get(self.id) {
                c.store(left - 1, Ordering::Relaxed);
            }
            out.abort(code(RESET_CODE + u32::try_from(self.id).unwrap_or(0)));
        }
    }

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut HostOut<'_>) {
        self.count(&self.shared.steps);
        self.step += 1;
        for s in 0..SLOTS {
            let agent = slot(s);
            #[expect(clippy::cast_precision_loss, reason = "small slot numbers")]
            let reward = 1.0 + s as f32;
            out.set_reward(agent, reward);
            let a = actions.discrete(agent, 0).unwrap_or(0);
            let abort = code(STEP_CODE + u32::try_from(s).unwrap_or(0));
            match a {
                DIE => out.agent_died(agent),
                SPAWN => out.agent_spawned(slot(2)),
                ABORT => out.abort(abort),
                TERMINATE => out.end_episode(EpisodeEnd::Terminated),
                TRUNCATE => out.end_episode(EpisodeEnd::Truncated),
                NAN => out.set_reward(agent, f32::NAN),
                TERMINATE_THEN_ABORT => {
                    out.end_episode(EpisodeEnd::Terminated);
                    out.abort(abort);
                }
                ABORT_THEN_TERMINATE => {
                    out.abort(abort);
                    out.end_episode(EpisodeEnd::Terminated);
                }
                BAD_EVENT_THEN_ABORT => {
                    out.agent_died(agent);
                    out.agent_spawned(agent);
                    out.abort(abort);
                }
                BAD_EVENT => {
                    out.agent_died(agent);
                    out.agent_spawned(agent);
                }
                _ => {}
            }
        }
    }

    fn validate(&self, actions: &Actions<'_, ()>) -> Result<(), ActionReject> {
        self.count(&self.shared.validates);
        let _ = actions;
        let bit = 1u32 << self.id;
        assert!(
            self.shared.panic_validate.load(Ordering::Relaxed) & bit == 0,
            "validate panicked"
        );
        if self.shared.hostile_validate.load(Ordering::Relaxed) {
            tessera_rl_num::fp::hostile::set(tessera_rl_num::fp::hostile::flush_to_zero);
        }
        if self.shared.alloc_validate.load(Ordering::Relaxed) {
            core::hint::black_box(vec![0u8; 64]);
        }
        let once = self.shared.reject_once.fetch_and(!bit, Ordering::Relaxed) & bit != 0;
        if once || self.shared.reject.load(Ordering::Relaxed) & bit != 0 {
            return Err(ActionReject {
                code: REJECT_CODE + u32::try_from(self.id).unwrap_or(0),
                reason: "the script refused",
            });
        }
        Ok(())
    }

    fn observe(&self, agent: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>) {
        self.count(&self.shared.observes);
        let values = [
            i32::try_from(self.id).unwrap_or(0) + 1,
            i32::try_from(self.step).unwrap_or(0) + 1,
            i32::try_from(self.episode).unwrap_or(0) + 1,
        ];
        let _ = agent;
        if let Some(leaf) = obs.leaf(0) {
            for (bytes, v) in leaf.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                bytes.copy_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn mask(&self, _: AgentSlot, mask: &mut MaskWriter<'_>) {
        self.count(&self.shared.masks);
        mask.fill(true);
    }

    fn acting(&self, agent: AgentSlot) -> bool {
        self.count(&self.shared.actings);
        match T {
            1 => agent.index() == 0,
            2 => self.shared.mixed_acting.load(Ordering::Relaxed),
            _ => true,
        }
    }
}

/// The script's bytes are its step and episode counters (the instance id
/// is per construction, not state).
impl<const T: u8, const V: bool> HostSnapshot for Script<T, V> {
    const SNAPSHOT_VERSION: u32 = 1;

    fn max_snapshot_len(_: &ScriptConfig) -> usize {
        8
    }

    fn save(&self, out: &mut Vec<u8>) {
        self.shared.saves.fetch_add(1, Ordering::Relaxed);
        assert!(
            !self.shared.panic_save.load(Ordering::Relaxed),
            "save panicked"
        );
        out.extend_from_slice(&self.step.to_le_bytes());
        out.extend_from_slice(&self.episode.to_le_bytes());
        if self.shared.save_extra.load(Ordering::Relaxed) {
            out.push(0);
        }
    }

    fn check_snapshot(cfg: &ScriptConfig, bytes: &[u8]) -> Result<(), SnapshotReject> {
        cfg.shared.checks.fetch_add(1, Ordering::Relaxed);
        assert!(
            !cfg.shared.panic_check.load(Ordering::Relaxed),
            "check_snapshot panicked"
        );
        let _ = bytes;
        Ok(())
    }

    fn load(&mut self, bytes: &[u8]) {
        self.shared.loads.fetch_add(1, Ordering::Relaxed);
        assert!(
            !self.shared.panic_load.load(Ordering::Relaxed),
            "load panicked"
        );
        let word = |at: usize| {
            u32::from_le_bytes(
                bytes
                    .get(at..at + 4)
                    .and_then(|b| b.try_into().ok())
                    .unwrap_or([0; 4]),
            )
        };
        self.step = word(0);
        self.episode = word(4);
    }
}
