// SPDX-License-Identifier: MIT OR Apache-2.0

//! The per-instance step every layout runs: the engine's sequence (RNG
//! identity, `E::step`, rewards, deaths, episode end with an immediate
//! reset, observations and masks), simplified to what the layouts touch.

use tessera_rl_core::io::{Actions, MaskWriter, ObsWriter};
use tessera_rl_core::{ActionMask, AgentSlot, Bool8, Environment, RecordLayout, StateInit};
use tessera_rl_num::identity::{RngIdentity, StreamId};
use tessera_rl_num::rng::{ObsRng, RngAuthority, StepRng};

/// What the spike needs from an environment beyond `Environment`.
pub trait SpikeEnv: Environment {
    /// Writes the masks of `agent`, if the environment has any.
    fn mask(cfg: &Self::Config, state: &Self::State, agent: AgentSlot, out: &mut MaskWriter<'_>);
    /// Mask length per agent (0: none).
    fn mask_len(cfg: &Self::Config) -> usize;
    /// Observation bytes per agent (one leaf).
    fn obs_bytes(cfg: &Self::Config) -> usize;
}

impl SpikeEnv for tessera_rl_classic::CartPole {
    fn mask(_: &Self::Config, _: &Self::State, _: AgentSlot, _: &mut MaskWriter<'_>) {}

    fn mask_len(_: &Self::Config) -> usize {
        0
    }

    fn obs_bytes(_: &Self::Config) -> usize {
        16
    }
}

impl SpikeEnv for tessera_rl_classic::Tag {
    fn mask(cfg: &Self::Config, state: &Self::State, agent: AgentSlot, out: &mut MaskWriter<'_>) {
        <Self as ActionMask>::action_mask(cfg, state, agent, out);
    }

    fn mask_len(_: &Self::Config) -> usize {
        5
    }

    fn obs_bytes(cfg: &Self::Config) -> usize {
        let side = usize::try_from(2 * cfg.radius + 1).unwrap_or(0);
        3 * side * side
    }
}

/// Per-call output buffers for a batch, shared by every layout.
#[derive(Debug)]
pub struct Work {
    /// `[N, A]`.
    pub rewards: Vec<f32>,
    /// `[N, A]` scratch flags: terminated, truncated, died, spawned.
    pub flags: [Vec<Bool8>; 4],
    /// `[N, A, obs bytes]`.
    pub obs: Vec<u8>,
    /// `[N, A, mask len]`.
    pub masks: Vec<Bool8>,
    agents: usize,
    obs_bytes: usize,
    mask_len: usize,
}

impl Work {
    /// Buffers for `n` instances.
    #[must_use]
    pub fn new<E: SpikeEnv>(cfg: &E::Config, n: usize) -> Self {
        let rows = n * E::MAX_AGENTS;
        let (obs_bytes, mask_len) = (E::obs_bytes(cfg), E::mask_len(cfg));
        Self {
            rewards: vec![0.0; rows],
            flags: core::array::from_fn(|_| vec![Bool8::FALSE; rows]),
            obs: vec![0; rows * obs_bytes],
            masks: vec![Bool8::FALSE; rows * mask_len],
            agents: E::MAX_AGENTS,
            obs_bytes,
            mask_len,
        }
    }
}

/// Starts instance `i`'s episode `episode` (the engine's reset).
pub fn reset<E: Environment>(
    cfg: &E::Config,
    authority: &RngAuthority,
    header: &mut E::Header,
    state: &mut E::State,
    episode: u32,
) {
    let id = header.engine().identity;
    let identity =
        RngIdentity::from_parts(id.seed(), id.branch_salt(), id.logical_id(), episode, 0);
    let engine = header.engine_mut();
    engine.identity = identity;
    engine.episode_steps = 0;
    engine.pending_reset = Bool8::FALSE;
    engine.fresh = Bool8::FALSE;
    engine.end_kind = 0;
    let initial = E::initial_agents(cfg);
    for (slot, row) in header.agents_mut().iter_mut().enumerate() {
        let alive = Bool8::from(slot < initial);
        row.alive = alive;
        row.acting = alive;
        row.episode_return = 0.0;
    }
    *state = E::State::state_default();
    let mut rng = StepRng::new(authority, &identity, StreamId::DYNAMICS);
    E::reset(cfg, &mut rng, state);
}

/// One transition of instance `i`, then its observations and masks.
#[expect(clippy::too_many_arguments, reason = "one instance's step")]
pub fn step<E: SpikeEnv>(
    cfg: &E::Config,
    authority: &RngAuthority,
    header: &mut E::Header,
    state: &mut E::State,
    actions: &[i32],
    work: &mut Work,
    i: usize,
    limit: u32,
) {
    let a = work.agents;
    let rows = i * a..(i + 1) * a;
    let engine = header.engine();
    let t = engine.episode_steps + 1;
    let id = engine.identity;
    let identity = RngIdentity::from_parts(
        id.seed(),
        id.branch_salt(),
        id.logical_id(),
        id.episode_index(),
        t,
    );
    let [term, trunc, died, spawned] = &mut work.flags;
    let (Some(rewards), Some(term), Some(trunc), Some(died), Some(spawned)) = (
        work.rewards.get_mut(rows.clone()),
        term.get_mut(rows.clone()),
        trunc.get_mut(rows.clone()),
        died.get_mut(rows.clone()),
        spawned.get_mut(rows),
    ) else {
        return;
    };
    for buf in [&mut *term, &mut *trunc, &mut *died, &mut *spawned] {
        buf.fill(Bool8::FALSE);
    }
    rewards.fill(0.0);
    let mut out = tessera_rl_core::StepOut::new(rewards, term, trunc, died, spawned);
    let mut rng = StepRng::new(authority, &identity, StreamId::DYNAMICS);
    E::step(
        cfg,
        state,
        &Actions::new(actions, 1, &[], 0),
        &mut rng,
        &mut out,
    );
    let end = out.episode_end().is_some() || t >= limit;
    header.engine_mut().identity = identity;
    header.engine_mut().episode_steps = t;
    for (k, row) in header.agents_mut().iter_mut().enumerate() {
        row.episode_return += work.rewards.get(i * a + k).copied().unwrap_or(0.0);
        if work.flags[2].get(i * a + k).is_some_and(|f| f.get()) {
            row.alive = Bool8::FALSE;
            row.acting = Bool8::FALSE;
        }
    }
    if end {
        reset::<E>(cfg, authority, header, state, id.episode_index() + 1);
    }
    observe::<E>(cfg, authority, header, state, work, i);
}

/// Observations and masks of instance `i`.
pub fn observe<E: SpikeEnv>(
    cfg: &E::Config,
    authority: &RngAuthority,
    header: &E::Header,
    state: &E::State,
    work: &mut Work,
    i: usize,
) {
    let (a, ob, ml) = (work.agents, work.obs_bytes, work.mask_len);
    let identity = header.engine().identity;
    for (slot, row) in header.agents().iter().enumerate() {
        let Some(agent) = u16::try_from(slot).ok().and_then(|s| AgentSlot::new(s, a)) else {
            continue;
        };
        let k = i * a + slot;
        if !row.alive.get() {
            continue;
        }
        if let Some(bytes) = work.obs.get_mut(k * ob..(k + 1) * ob) {
            let mut leaves = [bytes];
            let mut writer = ObsWriter::new(&mut leaves);
            let mut rng = ObsRng::new(authority, &identity);
            E::observe(cfg, state, agent, &mut rng, &mut writer);
        }
        if ml > 0
            && let Some(mask) = work.masks.get_mut(k * ml..(k + 1) * ml)
        {
            E::mask(cfg, state, agent, &mut MaskWriter::new(mask));
        }
    }
}
