// SPDX-License-Identifier: MIT OR Apache-2.0

//! The league in the engine (ADR-0088 sections 2 to 4 and 7, erratum 1):
//! `Opponent` slots, the per-instance assignment drawn at each episode's
//! reset, the per-instance result rings, and the league's subsystem at the
//! epoch barrier.
//!
//! `tessera-rl-league` depends on this crate, never the reverse: the
//! engine keeps the per-step data as plain values ([`LeagueView`], the
//! assignment array, the rings) and calls the league's [`LeagueSubsystem`]
//! (a trait object allocated once at `with_league`) only at
//! `epoch_barrier`, off the step path.
//!
//! **The draw** is a pure function of the record's identity (seed, salt,
//! logical id), its episode counter and the view: a `StepRng` on the
//! `LEAGUE` stream at step 0 gives one `u64`, which the view's cumulative
//! thresholds map to a policy. The engine draws on the calling thread after
//! every call that can start an episode, for every instance whose
//! `(identity, episode)` key changed, so the result does not depend on
//! where or when the reset ran, or on the worker count.

mod onnx;
mod pool_api;
mod view;

pub(crate) use onnx::OnnxPass;
pub use onnx::{OpponentPolicies, OpponentPolicyError};
pub use pool_api::LeagueSnapshot;
pub use view::{LeagueView, LeagueViewError, OpponentKind};

use tessera_rl_core::RecordLayout;
use tessera_rl_num::identity::RngIdentity;
use tessera_rl_num::rng::{RngAuthority, StepRng};

/// The outcome of one episode for the learner (ADR-0088 item 4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatchOutcome {
    /// The learner's return was above the opponents' mean.
    Win,
    /// Equal returns.
    Draw,
    /// Below.
    Loss,
}

impl MatchOutcome {
    /// The per-instance code the worker writes at episode end; [`NO_RESULT`]
    /// means none.
    pub(crate) const fn code(self) -> i8 {
        match self {
            Self::Win => 1,
            Self::Draw => 0,
            Self::Loss => -1,
        }
    }

    pub(crate) const fn from_code(code: i8) -> Option<Self> {
        match code {
            1 => Some(Self::Win),
            0 => Some(Self::Draw),
            -1 => Some(Self::Loss),
            _ => None,
        }
    }
}

/// No episode ended in this call for the instance.
pub(crate) const NO_RESULT: i8 = i8::MIN;

/// One result a barrier applies: the instance, the policy it played and
/// the outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeagueResult {
    /// The instance.
    pub instance: u32,
    /// The policy the instance's opponent slots played.
    pub policy: u32,
    /// The learner's outcome.
    pub outcome: MatchOutcome,
}

/// What a barrier hands the league: every result since the previous
/// barrier, in instance order and oldest first within an instance (ADR-0088
/// item 4.2), and the results the full rings dropped.
#[derive(Debug, Clone, Copy)]
pub struct LeagueResults<'a> {
    /// The results, in the barrier's order.
    pub results: &'a [LeagueResult],
    /// Results dropped by full rings since the previous barrier.
    pub dropped: u64,
}

/// What the league's barrier reports (ADR-0088 item 4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct LeagueReport {
    /// Results applied.
    pub applied: u64,
    /// Results dropped by full rings.
    pub dropped: u64,
    /// Whether the new table fell back to uniform.
    pub uniform_fallback: bool,
}

impl LeagueReport {
    /// A report.
    #[must_use]
    pub const fn new(applied: u64, dropped: u64, uniform_fallback: bool) -> Self {
        Self {
            applied,
            dropped,
            uniform_fallback,
        }
    }
}

/// Why the league's barrier failed; the pool keeps its previous view and is
/// not poisoned (erratum 1, condition 3).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("league barrier: {reason}")]
pub struct LeagueBarrierError {
    /// What failed.
    pub reason: String,
}

#[doc(hidden)]
pub mod sealed {
    /// Seals [`super::LeagueSubsystem`]: only `tessera-rl-league` implements
    /// it (ADR-0088 section 7 item 2).
    pub trait Sealed {}
}

/// The league's subsystem (ADR-0081's fourth, ADR-0088 section 7 and
/// erratum 1), implemented by `tessera-rl-league`. The engine calls it only
/// at `epoch_barrier`, inside the canonical floating-point scope.
pub trait LeagueSubsystem: sealed::Sealed + Send {
    /// Applies `results` to the ratings and the per-pair counts, recomputes
    /// the matchmaking table, and returns the next view and the report.
    ///
    /// # Errors
    ///
    /// [`LeagueBarrierError`]; the pool then keeps its previous view.
    fn barrier(
        &mut self,
        results: LeagueResults<'_>,
    ) -> Result<(LeagueView, LeagueReport), LeagueBarrierError>;
}

/// The league's part of the pool context (ADR-0088 item 7.3): the view the
/// pool draws from and each instance's assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeagueContext {
    /// The view.
    pub view: LeagueView,
    /// One policy per instance.
    pub assignments: Vec<u32>,
}

/// Default capacity of an instance's result ring (ADR-0088 item 4.1).
pub const DEFAULT_RING: usize = 8;

/// The `(identity, episode)` an instance's assignment was drawn for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct DrawKey {
    seed: u64,
    salt: u64,
    logical: u32,
    episode: u32,
    drawn: bool,
}

impl DrawKey {
    fn of(id: &RngIdentity) -> Self {
        Self {
            seed: id.seed().get(),
            salt: id.branch_salt().get(),
            logical: id.logical_id().get(),
            episode: id.episode_index(),
            drawn: true,
        }
    }
}

/// The league state of one pool.
pub(crate) struct LeagueState {
    pub(crate) view: LeagueView,
    pub(crate) subsystem: Box<dyn LeagueSubsystem>,
    /// Opponent slots as a bit mask (slot `s` is bit `s`; at most 64 slots).
    pub(crate) slots: u64,
    /// Per instance: the policy its opponent slots play this episode.
    pub(crate) assignment: Vec<u32>,
    /// Per instance: what the assignment was drawn for.
    pub(crate) keys: Vec<DrawKey>,
    /// Per instance: the outcome code the worker wrote at episode end, or
    /// [`NO_RESULT`].
    pub(crate) outcomes: Vec<i8>,
    /// Result rings, `capacity` entries per instance: `(policy, outcome)`.
    pub(crate) rings: Vec<(u32, MatchOutcome)>,
    /// Per instance: the ring's first entry and its length.
    pub(crate) ring_start: Vec<u16>,
    pub(crate) ring_len: Vec<u16>,
    pub(crate) capacity: usize,
    /// Results dropped by full rings since the previous barrier.
    pub(crate) dropped: u64,
    pub(crate) frozen: bool,
    /// The barrier's flat result list, reused.
    pub(crate) scratch: Vec<LeagueResult>,
}

impl core::fmt::Debug for LeagueState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LeagueState")
            .field("view", &self.view)
            .field("slots", &self.slots)
            .field("assignment", &self.assignment)
            .field("frozen", &self.frozen)
            .finish_non_exhaustive()
    }
}

/// The policy `view` assigns the episode of `identity` (ADR-0088 item 3.1).
#[must_use]
pub(crate) fn draw(authority: &RngAuthority, identity: &RngIdentity, view: &LeagueView) -> u32 {
    let mut rng = StepRng::league(authority, identity);
    view.draw(rng.next_u64())
}

impl LeagueState {
    pub(crate) fn new(
        view: LeagueView,
        subsystem: Box<dyn LeagueSubsystem>,
        slots: u64,
        instances: usize,
        capacity: usize,
    ) -> Self {
        Self {
            view,
            subsystem,
            slots,
            assignment: vec![0; instances],
            keys: vec![DrawKey::default(); instances],
            outcomes: vec![NO_RESULT; instances],
            rings: vec![(0, MatchOutcome::Draw); instances.saturating_mul(capacity)],
            ring_start: vec![0; instances],
            ring_len: vec![0; instances],
            capacity,
            dropped: 0,
            frozen: false,
            scratch: Vec::with_capacity(instances.saturating_mul(capacity)),
        }
    }

    /// Draws the assignment of every instance whose `(identity, episode)`
    /// changed since its last draw, and writes each instance's opponent
    /// mode into `modes` (ADR-0088 items 2.4 and 3.1). No allocation.
    pub(crate) fn refresh<H: RecordLayout>(
        &mut self,
        authority: &RngAuthority,
        headers: &[H],
        modes: &mut [u8],
    ) {
        for (i, header) in headers.iter().enumerate() {
            let engine = header.engine();
            if engine.fresh.get() {
                continue;
            }
            let key = DrawKey::of(&engine.identity);
            if let Some(k) = self.keys.get_mut(i)
                && *k != key
            {
                *k = key;
                if let Some(a) = self.assignment.get_mut(i) {
                    *a = draw(authority, &engine.identity, &self.view);
                }
            }
            let mode = match self.assignment.get(i).and_then(|&p| self.view.kind(p)) {
                Some(OpponentKind::Heuristic) => crate::layout::OPPONENT_HEURISTIC,
                Some(OpponentKind::Onnx { .. }) => crate::layout::OPPONENT_ONNX,
                Some(OpponentKind::External { .. }) | None => crate::layout::OPPONENT_EXTERNAL,
            };
            if let Some(m) = modes.get_mut(i) {
                *m = mode;
            }
        }
    }

    /// The pool context entry.
    pub(crate) fn context(&self) -> LeagueContext {
        LeagueContext {
            view: self.view.clone(),
            assignments: self.assignment.clone(),
        }
    }

    /// Checks a context entry against this league: the same population
    /// and one valid assignment per instance.
    pub(crate) fn check_context(
        &self,
        context: &LeagueContext,
    ) -> Result<(), crate::error::EngineError> {
        let expected = self.view.population_digest();
        if context.view.population_digest() != expected {
            return Err(crate::error::EngineError::LeagueMismatch {
                expected,
                found: context.view.population_digest(),
            });
        }
        let population = context.view.kinds().len();
        if context.assignments.len() != self.assignment.len()
            || context
                .assignments
                .iter()
                .any(|&p| usize::try_from(p).map_or(true, |p| p >= population))
        {
            return Err(crate::error::EngineError::League {
                reason: "the context's assignments do not fit the pool",
            });
        }
        Ok(())
    }

    /// Writes a checked context entry: the view and every assignment,
    /// marked drawn for the instances' current identities.
    pub(crate) fn restore_context<H: RecordLayout>(
        &mut self,
        context: &LeagueContext,
        headers: &[H],
    ) {
        self.view = context.view.clone();
        for (i, (&policy, header)) in context.assignments.iter().zip(headers).enumerate() {
            let identity = header.engine().identity;
            self.set(i, Some(policy), &identity);
        }
        self.clear();
    }

    /// Forgets instance `i`'s draw key, so the next refresh redraws it from
    /// its record's identity (a plain restore or import, ADR-0088 item 3.2).
    pub(crate) fn forget(&mut self, i: usize) {
        if let Some(k) = self.keys.get_mut(i) {
            *k = DrawKey::default();
        }
    }

    /// Copies instance `src`'s assignment to `dst` and marks it drawn for
    /// `dst`'s own identity (a fork, whose salt may differ: ADR-0088 item
    /// 3.2 keeps the opponent).
    pub(crate) fn copy(&mut self, src: usize, dst: usize, identity: &RngIdentity) {
        self.set(dst, self.assignment.get(src).copied(), identity);
    }

    /// Sets instance `i`'s assignment to `policy`, drawn for `identity`
    /// (fork and `restore_with_league`).
    pub(crate) fn set(&mut self, i: usize, policy: Option<u32>, identity: &RngIdentity) {
        let Some(policy) = policy else {
            return;
        };
        if let Some(d) = self.assignment.get_mut(i) {
            *d = policy;
        }
        if let Some(k) = self.keys.get_mut(i) {
            *k = DrawKey::of(identity);
        }
    }

    /// Moves the outcomes the workers wrote into the rings, in instance
    /// order, dropping the oldest entry of a full ring (ADR-0088 item 4.1).
    pub(crate) fn collect(&mut self) {
        let cap = self.capacity;
        for i in 0..self.outcomes.len() {
            let Some(code) = self.outcomes.get(i).copied() else {
                continue;
            };
            if let Some(o) = self.outcomes.get_mut(i) {
                *o = NO_RESULT;
            }
            let Some(outcome) = MatchOutcome::from_code(code) else {
                continue;
            };
            if self.frozen || cap == 0 {
                continue;
            }
            let policy = self.assignment.get(i).copied().unwrap_or(0);
            let (Some(start), Some(len)) = (
                self.ring_start.get(i).copied(),
                self.ring_len.get(i).copied(),
            ) else {
                continue;
            };
            let (start, len) = (usize::from(start), usize::from(len));
            let slot = if len == cap {
                // Full: overwrite the oldest, which moves the start.
                self.dropped = self.dropped.saturating_add(1);
                if let Some(s) = self.ring_start.get_mut(i) {
                    *s = u16::try_from((start + 1) % cap).unwrap_or(0);
                }
                start
            } else {
                if let Some(l) = self.ring_len.get_mut(i) {
                    *l = u16::try_from(len + 1).unwrap_or(u16::MAX);
                }
                (start + len) % cap
            };
            if let Some(e) = self.rings.get_mut(i * cap + slot) {
                *e = (policy, outcome);
            }
        }
    }

    /// The rings as the barrier's ordered list (instance order, oldest
    /// first), into `scratch`; the rings are emptied by [`Self::clear`]
    /// only after the subsystem accepted them.
    pub(crate) fn drain(&mut self) {
        self.scratch.clear();
        let cap = self.capacity;
        for i in 0..self.ring_len.len() {
            let start = usize::from(self.ring_start.get(i).copied().unwrap_or(0));
            let len = usize::from(self.ring_len.get(i).copied().unwrap_or(0));
            for k in 0..len {
                if let Some(&(policy, outcome)) = self.rings.get(i * cap + (start + k) % cap.max(1))
                {
                    self.scratch.push(LeagueResult {
                        instance: u32::try_from(i).unwrap_or(u32::MAX),
                        policy,
                        outcome,
                    });
                }
            }
        }
    }

    /// Empties the rings and the drop count (after a barrier).
    pub(crate) fn clear(&mut self) {
        self.ring_len.fill(0);
        self.ring_start.fill(0);
        self.dropped = 0;
    }

    /// Replaces the view as a barrier would, from a replay's `League` event
    /// (ADR-0088 section 8, erratum 3): the subsystem is not called.
    ///
    /// # Errors
    ///
    /// A population that shrank, as [`Self::barrier`] refuses.
    pub(crate) fn replace_view(&mut self, view: LeagueView) -> Result<(), LeagueBarrierError> {
        if view.kinds().len() < self.view.kinds().len() {
            return Err(LeagueBarrierError {
                reason: "a population shrank: entries are never removed".to_owned(),
            });
        }
        self.view = view;
        self.clear();
        Ok(())
    }

    /// The league's barrier (ADR-0088 section 7, erratum 1 condition 3):
    /// hands the rings to the subsystem and takes the next view. On error
    /// the view and the rings stay as they were; nothing is poisoned.
    pub(crate) fn barrier(&mut self) -> Result<Option<LeagueReport>, LeagueBarrierError> {
        if self.frozen {
            return Ok(None);
        }
        self.drain();
        let results = LeagueResults {
            results: &self.scratch,
            dropped: self.dropped,
        };
        let (view, report) = self.subsystem.barrier(results)?;
        if view.kinds().len() < self.view.kinds().len() {
            return Err(LeagueBarrierError {
                reason: "a population shrank: entries are never removed".to_owned(),
            });
        }
        self.view = view;
        self.clear();
        Ok(Some(report))
    }
}

/// The learner's outcome of an episode that just ended, from the agents'
/// episode returns (ADR-0088 item 4.1, erratum 1): the sign of the mean
/// return of the learner slots (non-opponent slots alive at the end or
/// with a nonzero return) minus the opponent slots' mean. An empty learner
/// set, equal means or a NaN are a draw; `None` only without an opponent
/// slot.
#[must_use]
pub(crate) fn outcome_of<H: RecordLayout>(header: &H, opponents: u64) -> Option<MatchOutcome> {
    let (mut learner, mut ln, mut opp, mut on) = (0.0f64, 0u32, 0.0f64, 0u32);
    for (slot, row) in header.agents().iter().enumerate() {
        let opponent = slot < 64 && (opponents >> slot) & 1 == 1;
        let r = f64::from(row.episode_return);
        if opponent {
            opp += r;
            on += 1;
        } else if row.alive.get() || row.episode_return != 0.0 {
            learner += r;
            ln += 1;
        }
    }
    if on == 0 {
        return None;
    }
    if ln == 0 {
        return Some(MatchOutcome::Draw);
    }
    let diff = learner / f64::from(ln) - opp / f64::from(on);
    Some(if diff > 0.0 {
        MatchOutcome::Win
    } else if diff < 0.0 {
        MatchOutcome::Loss
    } else {
        MatchOutcome::Draw
    })
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::{Bool8, NoAdaptive, RecordHeader, StateInit};

    use super::{MatchOutcome, outcome_of};

    fn header(rows: &[(bool, f32)]) -> RecordHeader<3, NoAdaptive> {
        let mut h = RecordHeader::<3, NoAdaptive>::state_default();
        for (row, &(alive, ret)) in h.agents.iter_mut().zip(rows) {
            row.alive = Bool8::from(alive);
            row.episode_return = ret;
        }
        h
    }

    /// Slot 2 is the opponent (bit 2).
    const OPP: u64 = 1 << 2;

    #[test]
    fn test_outcome_sign_of_learner_mean_minus_opponent_mean() {
        assert_eq!(
            outcome_of(&header(&[(true, 2.0), (true, 0.0), (true, 0.5)]), OPP),
            Some(MatchOutcome::Win)
        );
        assert_eq!(
            outcome_of(&header(&[(true, -1.0), (false, 0.0), (true, 0.5)]), OPP),
            Some(MatchOutcome::Loss)
        );
        assert_eq!(
            outcome_of(&header(&[(true, 0.5), (false, 0.0), (true, 0.5)]), OPP),
            Some(MatchOutcome::Draw)
        );
        assert_eq!(
            outcome_of(&header(&[(true, f32::NAN), (false, 0.0), (true, 0.5)]), OPP),
            Some(MatchOutcome::Draw)
        );
    }

    /// Rev's ruling 2: no learner slot (none alive at the end, none with a
    /// return) is a draw; without an opponent slot there is no result.
    #[test]
    fn test_outcome_empty_learner_set_is_a_draw() {
        assert_eq!(
            outcome_of(&header(&[(false, 0.0), (false, 0.0), (true, 3.0)]), OPP),
            Some(MatchOutcome::Draw)
        );
        assert_eq!(
            outcome_of(&header(&[(true, 1.0), (true, 0.0), (true, 0.0)]), 0),
            None
        );
    }
}
