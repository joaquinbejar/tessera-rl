// SPDX-License-Identifier: MIT OR Apache-2.0

//! `EnvPool`: instance records, construction and the Inline mode
//! (SPEC 5, 6; ADR-0006, ADR-0028, ADR-0036).
//!
//! Records are stored as two parallel arrays, headers and states; chunks
//! are ranges of `chunk_size` records. Inline mode runs every chunk on the
//! calling thread inside `fp::with_canonical` (ADR-0037). After
//! construction, `reset` and `step` never allocate.

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::{
    AdaptiveValues, AgentSlot, Bool8, EnvConfig, Environment, EpisodeEnd, RecordLayout, StateInit,
    StepOut,
};
use tessera_rl_num::fp;
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{ObsRng, RngAuthority, StepRng};

use crate::error::EngineError;
use crate::layout::{ActionLayout, MAX_LEAVES, obs_leaves};
use crate::options::PoolOptions;
use crate::outputs::Outputs;
use crate::validate::{ActionInput, check_and_copy_actions, check_ids};

/// How `reset` reseeds the instances it resets (SPEC 8.1).
#[derive(Debug, Clone, Copy)]
pub enum Reseed<'a> {
    /// Keep the seeds; the episode index advances.
    Keep,
    /// The k-th reset instance, in the order given (ADR-0007), gets
    /// `seed + k`, modulo 2^64; for a full reset, instance i gets
    /// `seed + i`.
    Base(Seed),
    /// The k-th reset instance gets `seeds[k]`.
    List(&'a [Seed]),
}

/// A pool of `num_envs` instances of `E`.
#[derive(Debug)]
pub struct EnvPool<E: Environment> {
    cfg: E::Config,
    options: PoolOptions,
    headers: Vec<E::Header>,
    states: Vec<E::State>,
    actions: ActionLayout,
    discrete: Vec<i32>,
    continuous: Vec<f32>,
    outputs: Outputs,
    seen: Vec<Bool8>,
    authority: RngAuthority,
}

impl<E> EnvPool<E>
where
    E: Environment,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    /// Builds a pool: validates the config, the options and the spaces, and
    /// initializes every record. Instances need a `reset` before stepping.
    ///
    /// # Errors
    ///
    /// Returns an [`EngineError`] for an invalid config, options or space.
    pub fn new(cfg: E::Config, options: PoolOptions) -> Result<Self, EngineError> {
        const {
            assert!(
                <E::Header as RecordLayout>::AGENTS == E::MAX_AGENTS,
                "Environment::Header must have MAX_AGENTS agent rows"
            );
        }
        cfg.validate()?;
        options.validate()?;
        let obs_space = E::observation_space(&cfg)?;
        obs_space.validate()?;
        let action_space = E::action_space(&cfg)?;
        action_space.validate()?;
        let actions = ActionLayout::from_space(&action_space)?;
        let leaves = obs_leaves(&obs_space)?;

        let num_envs = usize::try_from(options.num_envs.get()).map_err(|_| too_large())?;
        let rows = num_envs.checked_mul(E::MAX_AGENTS).ok_or_else(too_large)?;
        let outputs = Outputs::new(num_envs, E::MAX_AGENTS, &leaves).ok_or_else(too_large)?;
        let adaptive = <E::Header as RecordLayout>::Adaptive::from_config(&cfg);
        let headers = (0..num_envs)
            .map(|i| {
                let mut header = E::Header::state_default();
                // `i < num_envs <= u32::MAX`, so the conversion never fails.
                let seed = options.seed.offset(u64::try_from(i).unwrap_or(u64::MAX));
                let engine = header.engine_mut();
                engine.identity = RngIdentity::new(seed, LogicalId::DEFAULT);
                engine.fresh = Bool8::TRUE;
                engine.pending_reset = Bool8::TRUE;
                *header.adaptive_mut() = adaptive;
                header
            })
            .collect();
        Ok(Self {
            cfg,
            options,
            headers,
            states: (0..num_envs).map(|_| E::State::state_default()).collect(),
            discrete: vec![
                0;
                rows.checked_mul(actions.discrete.len())
                    .ok_or_else(too_large)?
            ],
            continuous: vec![
                0.0;
                rows.checked_mul(actions.continuous.len())
                    .ok_or_else(too_large)?
            ],
            actions,
            outputs,
            seen: vec![Bool8::FALSE; num_envs],
            authority: RngAuthority::engine_only(),
        })
    }

    /// The pool options.
    #[must_use]
    pub const fn options(&self) -> &PoolOptions {
        &self.options
    }

    /// The outputs of the last call.
    #[must_use]
    pub const fn outputs(&self) -> &Outputs {
        &self.outputs
    }

    /// Number of discrete and continuous action heads per agent.
    #[must_use]
    pub fn action_heads(&self) -> (usize, usize) {
        (self.actions.discrete.len(), self.actions.continuous.len())
    }

    /// The record header of instance `id`.
    #[must_use]
    pub fn header(&self, id: u32) -> Option<&E::Header> {
        self.headers.get(usize::try_from(id).ok()?)
    }

    /// The author state of instance `id`.
    #[must_use]
    pub fn state(&self, id: u32) -> Option<&E::State> {
        self.states.get(usize::try_from(id).ok()?)
    }

    /// Sets the lineage of `id`, so instances sharing a seed draw
    /// independent streams (ADR-0007). Takes effect from the next reset.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InstanceOutOfRange`] for an unknown id.
    pub fn set_lineage(&mut self, id: u32, lineage: LogicalId) -> Result<(), EngineError> {
        let num_envs = self.options.num_envs.get();
        let header = usize::try_from(id)
            .ok()
            .and_then(|i| self.headers.get_mut(i))
            .ok_or(EngineError::InstanceOutOfRange { id, num_envs })?;
        let identity = header.engine().identity;
        header.engine_mut().identity = RngIdentity::from_parts(
            identity.seed(),
            identity.branch_salt(),
            lineage,
            identity.episode_index(),
            identity.step_index(),
        );
        Ok(())
    }

    /// Resets every instance.
    ///
    /// # Errors
    ///
    /// Returns an [`EngineError`] if a seed list has the wrong length or an
    /// episode counter is exhausted; nothing is mutated then.
    pub fn reset_all(&mut self, reseed: Reseed<'_>) -> Result<&Outputs, EngineError> {
        let n = self.headers.len();
        self.check_reset(0..n, n, reseed)?;
        self.apply_reset(0..n, reseed);
        Ok(&self.outputs)
    }

    /// Resets the instances `ids`, in the order given; other instances keep
    /// their state and outputs.
    ///
    /// # Errors
    ///
    /// Returns an [`EngineError`] for unknown or duplicate ids, a seed list
    /// of the wrong length or an exhausted episode counter; nothing is
    /// mutated then.
    pub fn reset(&mut self, ids: &[u32], reseed: Reseed<'_>) -> Result<&Outputs, EngineError> {
        check_ids(ids, &mut self.seen)?;
        let indices = ids.iter().filter_map(|&id| usize::try_from(id).ok());
        self.check_reset(indices.clone(), ids.len(), reseed)?;
        self.apply_reset(indices, reseed);
        Ok(&self.outputs)
    }

    fn check_reset(
        &self,
        indices: impl Iterator<Item = usize>,
        count: usize,
        reseed: Reseed<'_>,
    ) -> Result<(), EngineError> {
        if let Reseed::List(seeds) = reseed
            && seeds.len() != count
        {
            return Err(EngineError::SeedCount {
                seeds: seeds.len(),
                instances: count,
            });
        }
        if matches!(reseed, Reseed::Keep) {
            for i in indices {
                let engine = self.headers.get(i).map(RecordLayout::engine);
                if let Some(engine) = engine
                    && !engine.fresh.get()
                    && engine.identity.episode_index() == u32::MAX
                {
                    return Err(EngineError::CounterExhausted {
                        id: u32::try_from(i).unwrap_or(u32::MAX),
                        counter: "episode_index",
                    });
                }
            }
        }
        Ok(())
    }

    fn apply_reset(&mut self, indices: impl Iterator<Item = usize>, reseed: Reseed<'_>) {
        let Self {
            cfg,
            headers,
            states,
            outputs,
            authority,
            ..
        } = self;
        fp::with_canonical(|| {
            for (k, i) in indices.enumerate() {
                let (Some(header), Some(state)) = (headers.get_mut(i), states.get_mut(i)) else {
                    continue;
                };
                let seed = match reseed {
                    Reseed::Keep => None,
                    // `k` counts reset instances, at most `u32::MAX`.
                    Reseed::Base(base) => Some(base.offset(u64::try_from(k).unwrap_or(u64::MAX))),
                    Reseed::List(seeds) => seeds.get(k).copied(),
                };
                reset_one::<E>(cfg, authority, header, state, seed);
                clear_instance(outputs, i);
                write_agent_masks(outputs, i, header);
                observe_instance::<E>(cfg, authority, header, state, outputs, i);
            }
        });
    }

    /// Steps every instance with `actions` (Inline mode).
    ///
    /// # Errors
    ///
    /// Returns an [`EngineError`] if the actions have the wrong shape, a
    /// required action is invalid, an instance is pending reset or a step
    /// counter is exhausted; nothing is mutated then.
    pub fn step(&mut self, actions: ActionInput<'_>) -> Result<&Outputs, EngineError> {
        check_and_copy_actions(
            &self.actions,
            &self.headers,
            actions,
            &mut self.discrete,
            &mut self.continuous,
        )?;
        let Self {
            cfg,
            options,
            headers,
            states,
            actions: layout,
            discrete,
            continuous,
            outputs,
            authority,
            ..
        } = self;
        let (dh, ch) = (layout.discrete.len(), layout.continuous.len());
        let limit = options.max_episode_steps.map(core::num::NonZeroU32::get);
        fp::with_canonical(|| {
            for (i, (header, state)) in headers.iter_mut().zip(states.iter_mut()).enumerate() {
                let rows = i * E::MAX_AGENTS;
                let view = Actions::new(
                    discrete
                        .get(rows * dh..(rows + E::MAX_AGENTS) * dh)
                        .unwrap_or(&[]),
                    dh,
                    continuous
                        .get(rows * ch..(rows + E::MAX_AGENTS) * ch)
                        .unwrap_or(&[]),
                    ch,
                );
                step_one::<E>(cfg, authority, header, state, &view, outputs, i, limit);
                observe_instance::<E>(cfg, authority, header, state, outputs, i);
            }
        });
        Ok(&self.outputs)
    }
}

fn too_large() -> EngineError {
    EngineError::Options {
        option: "num_envs",
        reason: "buffers too large",
    }
}

fn with_step<H: RecordLayout>(header: &mut H, episode: u32, step: u32) -> RngIdentity {
    let id = header.engine().identity;
    let identity =
        RngIdentity::from_parts(id.seed(), id.branch_salt(), id.logical_id(), episode, step);
    header.engine_mut().identity = identity;
    identity
}

fn reset_one<E: Environment>(
    cfg: &E::Config,
    authority: &RngAuthority,
    header: &mut E::Header,
    state: &mut E::State,
    seed: Option<Seed>,
) {
    let engine = header.engine();
    let mut episode = engine.identity.episode_index();
    if let Some(seed) = seed {
        let id = engine.identity;
        header.engine_mut().identity =
            RngIdentity::from_parts(seed, id.branch_salt(), id.logical_id(), 0, 0);
        episode = 0;
    } else if !engine.fresh.get() {
        // Checked at the boundary.
        episode = episode.saturating_add(1);
    }
    let identity = with_step(header, episode, 0);
    let engine = header.engine_mut();
    engine.fresh = Bool8::FALSE;
    engine.pending_reset = Bool8::FALSE;
    engine.end_kind = 0;
    engine.episode_steps = 0;
    for row in header.agents_mut() {
        row.alive = Bool8::TRUE;
        row.acting = Bool8::TRUE;
        row.episode_return = 0.0;
    }
    *state = E::State::state_default();
    let mut rng = StepRng::new(authority, &identity, StreamId::DYNAMICS);
    E::reset(cfg, &mut rng, state);
}

#[expect(
    clippy::too_many_arguments,
    reason = "one instance's step, split out of the loop"
)]
fn step_one<E: Environment>(
    cfg: &E::Config,
    authority: &RngAuthority,
    header: &mut E::Header,
    state: &mut E::State,
    actions: &Actions<'_, E::Action>,
    outputs: &mut Outputs,
    i: usize,
    limit: Option<u32>,
) {
    let a = E::MAX_AGENTS;
    let range = i * a..(i + 1) * a;
    let t = header.engine().episode_steps.saturating_add(1);
    let episode = header.engine().identity.episode_index();
    let identity = with_step(header, episode, t);
    clear_instance(outputs, i);
    let end = {
        let Outputs {
            rewards,
            agent_terminated,
            agent_truncated,
            died,
            spawned,
            ..
        } = outputs;
        let (Some(reward_row), Some(term_row), Some(trunc_row), Some(died_row), Some(spawned_row)) = (
            rewards.get_mut(range.clone()),
            agent_terminated.get_mut(range.clone()),
            agent_truncated.get_mut(range.clone()),
            died.get_mut(range.clone()),
            spawned.get_mut(range.clone()),
        ) else {
            return;
        };
        let mut out = StepOut::new(reward_row, term_row, trunc_row, died_row, spawned_row);
        let mut rng = StepRng::new(authority, &identity, StreamId::DYNAMICS);
        E::step(cfg, state, actions, &mut rng, &mut out);
        out.episode_end()
    };
    let end = end.or_else(|| (limit == Some(t)).then_some(EpisodeEnd::Truncated));

    header.engine_mut().episode_steps = t;
    let mut total = 0.0f32;
    for (slot, row) in header.agents_mut().iter_mut().enumerate() {
        let k = i * a + slot;
        let reward = outputs.rewards.get(k).copied().unwrap_or(0.0);
        debug_assert!(
            reward.is_finite(),
            "non-finite reward for agent slot {slot}"
        );
        row.episode_return += reward;
        total += row.episode_return;
        let died = outputs.died.get(k).is_some_and(|f| f.get());
        let spawned = outputs.spawned.get(k).is_some_and(|f| f.get());
        if died {
            row.alive = Bool8::FALSE;
            row.acting = Bool8::FALSE;
        }
        // A spawn needs an empty slot that did not die this step; otherwise
        // it is ignored (debug builds assert).
        if spawned {
            let valid = !died && !row.alive.get();
            debug_assert!(valid, "spawn on an occupied or dying slot {slot}");
            if valid {
                row.alive = Bool8::TRUE;
                row.acting = Bool8::TRUE;
                // Generations count occupants modulo 2^32: an explicit,
                // documented wrap (unreachable in practice).
                row.generation = row.generation.wrapping_add(1);
            }
        }
    }
    // Terminal flush (SPEC 12): when the episode ends, every agent still
    // alive gets its final terminated or truncated flag in this step.
    if let Some(end) = end {
        for (slot, row) in header.agents().iter().enumerate() {
            if row.alive.get() {
                let k = i * a + slot;
                let flags = match end {
                    EpisodeEnd::Terminated => &mut outputs.agent_terminated,
                    EpisodeEnd::Truncated => &mut outputs.agent_truncated,
                };
                if let Some(f) = flags.get_mut(k) {
                    *f = Bool8::TRUE;
                }
            }
        }
    }
    if let Some(end) = end {
        let engine = header.engine_mut();
        engine.pending_reset = Bool8::TRUE;
        let (terminated, kind) = match end {
            EpisodeEnd::Terminated => (true, 1),
            EpisodeEnd::Truncated => (false, 2),
        };
        engine.end_kind = kind;
        set(&mut outputs.episode_terminated, i, terminated);
        set(&mut outputs.episode_truncated, i, !terminated);
        if let Some(r) = outputs.episode_return.get_mut(i) {
            *r = total;
        }
        if let Some(l) = outputs.episode_length.get_mut(i) {
            // Episode lengths are reported as f32 statistics (SPEC 5.2).
            #[expect(clippy::cast_precision_loss, reason = "statistic, exact below 2^24")]
            let length = t as f32;
            *l = length;
        }
    }
    write_agent_masks(outputs, i, header);
}

fn set(buf: &mut [Bool8], i: usize, value: bool) {
    if let Some(f) = buf.get_mut(i) {
        *f = Bool8::new(value);
    }
}

fn clear_instance(outputs: &mut Outputs, i: usize) {
    let a = outputs.agents;
    let range = i * a..(i + 1) * a;
    for buf in [
        &mut outputs.agent_terminated,
        &mut outputs.agent_truncated,
        &mut outputs.died,
        &mut outputs.spawned,
    ] {
        if let Some(rows) = buf.get_mut(range.clone()) {
            rows.fill(Bool8::FALSE);
        }
    }
    if let Some(rows) = outputs.rewards.get_mut(range) {
        rows.fill(0.0);
    }
    set(&mut outputs.episode_terminated, i, false);
    set(&mut outputs.episode_truncated, i, false);
    for buf in [&mut outputs.episode_return, &mut outputs.episode_length] {
        if let Some(v) = buf.get_mut(i) {
            *v = 0.0;
        }
    }
}

fn write_agent_masks<H: RecordLayout>(outputs: &mut Outputs, i: usize, header: &H) {
    let a = outputs.agents;
    for (slot, row) in header.agents().iter().enumerate() {
        let k = i * a + slot;
        if let Some(f) = outputs.alive.get_mut(k) {
            *f = row.alive;
        }
        if let Some(f) = outputs.acting.get_mut(k) {
            *f = row.acting;
        }
        if let Some(g) = outputs.generations.get_mut(k) {
            *g = row.generation;
        }
    }
}

fn observe_instance<E: Environment>(
    cfg: &E::Config,
    authority: &RngAuthority,
    header: &E::Header,
    state: &E::State,
    outputs: &mut Outputs,
    i: usize,
) {
    let identity = header.engine().identity;
    let a = E::MAX_AGENTS;
    for (slot, row) in header.agents().iter().enumerate() {
        let k = i * a + slot;
        let mut views: [&mut [u8]; MAX_LEAVES] = Default::default();
        let mut used = 0;
        for ((buf, &bytes), view) in outputs
            .observations
            .iter_mut()
            .zip(&outputs.leaf_bytes)
            .zip(views.iter_mut())
        {
            if let Some(rows) = buf.get_mut(k * bytes..(k + 1) * bytes) {
                rows.fill(0);
                *view = rows;
            }
            used += 1;
        }
        if !row.alive.get() {
            continue;
        }
        let (Some(agent), Some(leaves)) = (
            AgentSlot::new(u16::try_from(slot).unwrap_or(u16::MAX), a),
            views.get_mut(..used),
        ) else {
            continue;
        };
        let mut writer = ObsWriter::new(leaves);
        let mut rng = ObsRng::new(authority, &identity);
        E::observe(cfg, state, agent, &mut rng, &mut writer);
    }
}
