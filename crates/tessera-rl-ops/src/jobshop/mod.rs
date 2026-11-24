// SPDX-License-Identifier: MIT OR Apache-2.0

//! Job shop scheduling (`tessera/JobShop-v1`): one agent dispatches
//! operations of a generated job shop instance, an original implementation
//! of the textbook problem (SPEC 16, ADR-0034). Cross-platform tier,
//! integer state.
//!
//! - **Instance:** at reset, `jobs` jobs each visit every one of `machines`
//!   machines once, in an order drawn uniformly (Fisher-Yates) from the
//!   instance's `StepRng`; each operation takes `1..=max_duration` time
//!   units, drawn uniformly. Steps draw no randomness, so the seed fixes
//!   the whole episode.
//! - **Action:** `Discrete(jobs)`: dispatch the next operation of job `j`.
//!   It starts at the later of the job's ready time and its machine's free
//!   time (a semi-active schedule) and ends `duration` later.
//! - **Mask** (ADR-0014): a job is legal while it has operations left. The
//!   no-op is the first such job.
//! - **Reward:** minus each step's increase of the current makespan (the
//!   latest end time scheduled so far), so an episode's return is exactly
//!   minus the final makespan. Values are integers below 2^24, exact in
//!   `f32`.
//! - **End:** terminated once every operation is scheduled, after exactly
//!   `jobs * machines` steps. Never truncated by the environment.
//! - **Observation** (times relative to the current makespan, so every
//!   value is at least 0):
//!   - `jobs`, `[jobs, 5]` `i32`, per job: the machine of its next
//!     operation (`machines` when done), that operation's duration (0 when
//!     done), the job's remaining work, the makespan minus the job's ready
//!     time, and 1 once the job is done;
//!   - `machines`, `[machines, 2]` `i32`, per machine: the makespan minus
//!     its free time, and its remaining load (the durations of its
//!     operations not yet scheduled).

pub mod baseline;

use core::num::NonZeroU32;

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Environment, EpisodeEnd, InvariantViolation, NoAdaptive, ObsLeaf,
    Observation, RecordHeader, Reserved, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/JobShop-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;

// Registration under the versioned id (SPEC 10, ADR-0054).
tessera_rl_core::register!(JobShop, id = "tessera/JobShop-v1");

/// The most jobs.
pub const MAX_JOBS: usize = 64;
/// The most machines.
pub const MAX_MACHINES: usize = 32;
/// Features per job in the `jobs` leaf.
pub const JOB_FEATURES: usize = 5;
/// Features per machine in the `machines` leaf.
pub const MACHINE_FEATURES: usize = 2;
/// An upper bound of every time and observation value:
/// `MAX_JOBS * MAX_MACHINES * 255` is 522,240.
pub const MAX_TIME: i32 = 1_000_000;
const _: () = assert!(MAX_JOBS * MAX_MACHINES * 255 < MAX_TIME as usize);
const _: () = assert!(MAX_TIME < 1 << 24, "rewards stay exact in f32");

/// Job shop configuration. `jobs` and `machines` set the spaces' shapes.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct JobShopConfig {
    /// Jobs, 2 to 64.
    #[config(default = 6, range = 2..=64)]
    pub jobs: u32,
    /// Machines, 2 to 32; every job visits each once.
    #[config(default = 6, range = 2..=32)]
    pub machines: u32,
    /// The longest operation, 1 to 255 time units.
    #[config(default = 99, range = 1..=255)]
    pub max_duration: u32,
}

impl JobShopConfig {
    /// Jobs, as a count.
    #[must_use]
    pub fn jobs(&self) -> usize {
        usize::try_from(self.jobs).unwrap_or(2).clamp(2, MAX_JOBS)
    }

    /// Machines, as a count.
    #[must_use]
    pub fn machines(&self) -> usize {
        usize::try_from(self.machines)
            .unwrap_or(2)
            .clamp(2, MAX_MACHINES)
    }
}

/// Job shop state: the instance and the partial schedule. The `u32`
/// fields come first and `reserved` pads the bytes to a multiple of 4, so
/// the layout has no padding.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct JobShopState {
    /// When each job's last scheduled operation ends.
    pub job_ready: [u32; MAX_JOBS],
    /// When each machine's last scheduled operation ends.
    pub machine_free: [u32; MAX_MACHINES],
    /// The latest end time scheduled so far.
    pub makespan: u32,
    /// Operations scheduled so far.
    pub scheduled: u32,
    /// The machine of each job's `k`-th operation.
    pub route: [[u8; MAX_MACHINES]; MAX_JOBS],
    /// The duration of each job's `k`-th operation.
    pub duration: [[u8; MAX_MACHINES]; MAX_JOBS],
    /// Each job's next operation (`machines` once done).
    pub next_op: [u8; MAX_JOBS],
    /// Jobs of this instance.
    pub jobs: u8,
    /// Machines of this instance.
    pub machines: u8,
    /// 1 once every operation is scheduled.
    pub done: u8,
    /// Padding to a multiple of 4 bytes.
    pub reserved: Reserved<1>,
}

/// The state's size: 392 bytes of `u32` fields and 4,164 of bytes. A
/// change here changes every snapshot.
pub const STATE_BYTES: usize = 4_556;
const _: () = assert!(core::mem::size_of::<JobShopState>() == STATE_BYTES);

impl Default for JobShopState {
    fn default() -> Self {
        Self {
            job_ready: [0; MAX_JOBS],
            machine_free: [0; MAX_MACHINES],
            makespan: 0,
            scheduled: 0,
            route: [[0; MAX_MACHINES]; MAX_JOBS],
            duration: [[0; MAX_MACHINES]; MAX_JOBS],
            next_op: [0; MAX_JOBS],
            jobs: 0,
            machines: 0,
            done: 0,
            reserved: Reserved::ZERO,
        }
    }
}

/// Job shop observation (see the module documentation).
#[derive(Observation)]
pub struct JobShopObs {
    /// Per job: next machine, next duration, remaining work, makespan minus
    /// ready time, done.
    #[space(box(low = 0, high = 1_000_000, shape = |c: &JobShopConfig| [c.jobs, 5]))]
    pub jobs: ObsLeaf<i32>,
    /// Per machine: makespan minus free time, remaining load.
    #[space(box(low = 0, high = 1_000_000, shape = |c: &JobShopConfig| [c.machines, 2]))]
    pub machines: ObsLeaf<i32>,
}

/// The job shop environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct JobShop;

fn bound(n: usize) -> NonZeroU32 {
    u32::try_from(n)
        .ok()
        .and_then(NonZeroU32::new)
        .unwrap_or(NonZeroU32::MIN)
}

impl JobShopState {
    /// Jobs, as a count.
    #[must_use]
    pub fn job_count(&self) -> usize {
        usize::from(self.jobs).min(MAX_JOBS)
    }

    /// Machines, as a count.
    #[must_use]
    pub fn machine_count(&self) -> usize {
        usize::from(self.machines).min(MAX_MACHINES)
    }

    /// Whether job `j` has operations left.
    #[must_use]
    pub fn legal(&self, j: usize) -> bool {
        j < self.job_count()
            && self
                .next_op
                .get(j)
                .is_some_and(|&k| usize::from(k) < self.machine_count())
    }

    /// The first job with operations left (the no-op), or 0.
    #[must_use]
    pub fn noop(&self) -> usize {
        (0..self.job_count()).find(|&j| self.legal(j)).unwrap_or(0)
    }

    /// The machine and duration of job `j`'s next operation, if any.
    #[must_use]
    pub fn next(&self, j: usize) -> Option<(usize, u32)> {
        if !self.legal(j) {
            return None;
        }
        let k = usize::from(*self.next_op.get(j)?);
        let m = usize::from(*self.route.get(j)?.get(k)?);
        let d = u32::from(*self.duration.get(j)?.get(k)?);
        Some((m, d))
    }

    /// The durations of job `j`'s operations not yet scheduled.
    #[must_use]
    pub fn remaining_work(&self, j: usize) -> u32 {
        let (Some(&k), Some(row)) = (self.next_op.get(j), self.duration.get(j)) else {
            return 0;
        };
        row.iter()
            .take(self.machine_count())
            .skip(usize::from(k))
            .map(|&d| u32::from(d))
            .sum()
    }

    /// The durations of machine `m`'s operations not yet scheduled.
    #[must_use]
    pub fn remaining_load(&self, m: usize) -> u32 {
        let machines = self.machine_count();
        let mut load = 0;
        for j in 0..self.job_count() {
            let (Some(&k), Some(route), Some(dur)) =
                (self.next_op.get(j), self.route.get(j), self.duration.get(j))
            else {
                continue;
            };
            for (r, d) in route.iter().zip(dur).take(machines).skip(usize::from(k)) {
                if usize::from(*r) == m {
                    load += u32::from(*d);
                }
            }
        }
        load
    }

    /// Schedules job `j`'s next operation; returns the makespan increase,
    /// or `None` (nothing changed) when `j` has no operation left.
    fn dispatch(&mut self, j: usize) -> Option<u32> {
        let (m, d) = self.next(j)?;
        let ready = *self.job_ready.get(j)?;
        let free = *self.machine_free.get(m)?;
        let end = ready.max(free) + d;
        *self.job_ready.get_mut(j)? = end;
        *self.machine_free.get_mut(m)? = end;
        *self.next_op.get_mut(j)? += 1;
        self.scheduled += 1;
        let before = self.makespan;
        self.makespan = before.max(end);
        Some(self.makespan - before)
    }

    /// Operations in this instance.
    #[must_use]
    pub fn total(&self) -> u32 {
        u32::from(self.jobs) * u32::from(self.machines)
    }
}

impl Environment for JobShop {
    type State = JobShopState;
    type Config = JobShopConfig;
    type Obs = JobShopObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &JobShopConfig) -> Result<Space, SpaceError> {
        JobShopObs::space(cfg)
    }

    fn action_space(cfg: &JobShopConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: cfg.jobs })
    }

    fn reset(cfg: &JobShopConfig, rng: &mut StepRng, state: &mut JobShopState) {
        let (jobs, machines) = (cfg.jobs(), cfg.machines());
        *state = JobShopState {
            jobs: u8::try_from(jobs).unwrap_or(2),
            machines: u8::try_from(machines).unwrap_or(2),
            ..JobShopState::default()
        };
        let max_duration = bound(usize::try_from(cfg.max_duration).unwrap_or(1));
        for (route, duration) in state
            .route
            .iter_mut()
            .zip(state.duration.iter_mut())
            .take(jobs)
        {
            let route = route.get_mut(..machines).unwrap_or_default();
            for (m, r) in route.iter_mut().enumerate() {
                *r = u8::try_from(m).unwrap_or(0);
            }
            // Fisher-Yates, one draw per position from the back.
            for i in (1..route.len()).rev() {
                let k = usize::try_from(rng.below_u32(bound(i + 1))).unwrap_or(0);
                route.swap(i, k);
            }
            for d in duration.iter_mut().take(machines) {
                *d = u8::try_from(rng.below_u32(max_duration) + 1).unwrap_or(1);
            }
        }
    }

    fn step(
        _: &JobShopConfig,
        state: &mut JobShopState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        // The engine replaced an illegal action (ADR-0028); the kernel
        // stays total and reads anything else as the no-op.
        let j = actions
            .discrete(AgentSlot::FIRST, 0)
            .and_then(|a| usize::try_from(a).ok())
            .filter(|&j| state.legal(j))
            .unwrap_or_else(|| state.noop());
        let gain = state.dispatch(j).unwrap_or(0);
        #[expect(clippy::cast_precision_loss, reason = "below MAX_TIME, exact in f32")]
        let reward = -(gain as f32);
        out.set_reward(AgentSlot::FIRST, reward);
        if state.scheduled >= state.total() {
            state.done = 1;
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &JobShopConfig,
        state: &JobShopState,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, JobShopObs>,
    ) {
        let int = |v: u32| i32::try_from(v).unwrap_or(MAX_TIME);
        let since = |t: u32| int(state.makespan.saturating_sub(t));
        if let Some(mut leaf) = out.leaf_mut(JobShopObs::JOBS) {
            for j in 0..state.job_count() {
                let (machine, duration) = state
                    .next(j)
                    .map_or((u32::from(state.machines), 0), |(m, d)| {
                        (u32::try_from(m).unwrap_or(0), d)
                    });
                let ready = state.job_ready.get(j).copied().unwrap_or(0);
                let row = [
                    int(machine),
                    int(duration),
                    int(state.remaining_work(j)),
                    since(ready),
                    i32::from(!state.legal(j)),
                ];
                for (f, v) in row.into_iter().enumerate() {
                    leaf.set(j * JOB_FEATURES + f, v);
                }
            }
        }
        if let Some(mut leaf) = out.leaf_mut(JobShopObs::MACHINES) {
            for m in 0..state.machine_count() {
                let free = state.machine_free.get(m).copied().unwrap_or(0);
                leaf.set(m * MACHINE_FEATURES, since(free));
                leaf.set(m * MACHINE_FEATURES + 1, int(state.remaining_load(m)));
            }
        }
    }
}

impl ActionMask for JobShop {
    fn action_mask(
        _: &JobShopConfig,
        state: &JobShopState,
        _: AgentSlot,
        out: &mut MaskWriter<'_>,
    ) {
        for j in 0..state.job_count() {
            out.set(j, state.legal(j));
        }
    }

    fn noop(_: &JobShopConfig, state: &JobShopState, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, i32::try_from(state.noop()).unwrap_or(0));
    }
}

impl Validate for JobShop {
    fn validate(cfg: &JobShopConfig, s: &JobShopState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        // A record that was never reset holds the default state.
        if *s == JobShopState::default() {
            return Ok(());
        }
        let (jobs, machines) = (cfg.jobs(), cfg.machines());
        if s.job_count() != jobs || s.machine_count() != machines {
            return invalid("the instance shape differs from the config");
        }
        let max_duration = u8::try_from(cfg.max_duration).unwrap_or(u8::MAX);
        let mut scheduled = 0u32;
        let mut latest = 0u32;
        for j in 0..MAX_JOBS {
            let (Some(route), Some(dur), Some(&k), Some(&ready)) = (
                s.route.get(j),
                s.duration.get(j),
                s.next_op.get(j),
                s.job_ready.get(j),
            ) else {
                return invalid("job arrays out of range");
            };
            if j >= jobs {
                if route.iter().chain(dur).any(|&b| b != 0) || k != 0 || ready != 0 {
                    return invalid("data beyond the configured jobs");
                }
                continue;
            }
            if usize::from(k) > machines {
                return invalid("a next operation past the last");
            }
            let mut seen = 0u64;
            for (&r, &d) in route.iter().zip(dur).take(machines) {
                if usize::from(r) >= machines || seen & (1 << r) != 0 {
                    return invalid("a route that is not a permutation of the machines");
                }
                seen |= 1 << r;
                if d == 0 || d > max_duration {
                    return invalid("a duration outside 1..=max_duration");
                }
            }
            if route.iter().skip(machines).any(|&b| b != 0)
                || dur.iter().skip(machines).any(|&b| b != 0)
            {
                return invalid("data beyond the configured machines");
            }
            if (k == 0) != (ready == 0) {
                return invalid("a ready time that does not match the scheduled operations");
            }
            scheduled += u32::from(k);
            latest = latest.max(ready);
        }
        for (m, &free) in s.machine_free.iter().enumerate() {
            if m >= machines && free != 0 {
                return invalid("data beyond the configured machines");
            }
            latest = latest.max(free);
        }
        if s.scheduled != scheduled || s.makespan != latest {
            return invalid("counters that do not match the schedule");
        }
        if (s.done == 1) != (scheduled == s.total()) || s.done > 1 {
            return invalid("a done flag that does not match the schedule");
        }
        Ok(())
    }
}

impl tessera_rl_core::Persist for JobShop {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
