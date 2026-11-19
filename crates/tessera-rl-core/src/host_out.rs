// SPDX-License-Identifier: MIT OR Apache-2.0

//! `HostOut`: what a host environment's `reset` or `step` reports for one
//! instance, `StepOut` plus an abort (ADR-0065 item 1).

use core::num::NonZeroU32;
use core::ops::{Deref, DerefMut};

use crate::step_out::StepOut;

/// Why a host environment aborted an episode: a non-zero code whose meaning
/// is the environment author's. Zero is reserved for "not aborted" in the
/// outputs, so `AbortCode::new(0)` is `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AbortCode(NonZeroU32);

impl AbortCode {
    /// A code, or `None` for 0.
    #[must_use]
    pub const fn new(code: u32) -> Option<Self> {
        match NonZeroU32::new(code) {
            Some(code) => Some(Self(code)),
            None => None,
        }
    }

    /// The code.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

/// What a host `reset` or `step` writes: everything [`StepOut`] offers,
/// through `Deref`, plus [`HostOut::abort`].
///
/// An abort ends the episode of an instance whose state became
/// inconsistent, without a panic (which poisons the pool, ADR-0028). The
/// engine truncates every live agent, keeps the call's rewards (except
/// a `SameStep` reset's, which writes scratch rows and whose rewards are
/// discarded, abort or not), discards its deaths and spawns, and calls neither `observe`, `mask` nor `acting`
/// on the aborted state: its observation rows are zeros, which may lie
/// outside the observation space, and its masks are false. The instance is
/// left pending a reset (ADR-0065 item 1). An author who can still observe
/// a consistent final state ends the episode with
/// `end_episode(EpisodeEnd::Truncated)` instead.
#[derive(Debug)]
pub struct HostOut<'a> {
    out: StepOut<'a>,
    abort: Option<AbortCode>,
}

impl<'a> HostOut<'a> {
    /// Engine-only: a view over one instance's rows.
    #[doc(hidden)]
    #[must_use]
    #[inline]
    pub const fn new(out: StepOut<'a>) -> Self {
        Self { out, abort: None }
    }

    /// Engine-only: the abort the call reported, if any.
    #[doc(hidden)]
    #[must_use]
    #[inline]
    pub const fn aborted(&self) -> Option<AbortCode> {
        self.abort
    }

    /// Ends this instance's episode as aborted, with `code`. The last call
    /// wins. It takes precedence over `end_episode`, whether `end_episode`
    /// is called before or after it.
    #[inline]
    pub const fn abort(&mut self, code: AbortCode) {
        self.abort = Some(code);
    }
}

impl<'a> Deref for HostOut<'a> {
    type Target = StepOut<'a>;

    #[inline]
    fn deref(&self) -> &StepOut<'a> {
        &self.out
    }
}

impl<'a> DerefMut for HostOut<'a> {
    #[inline]
    fn deref_mut(&mut self) -> &mut StepOut<'a> {
        &mut self.out
    }
}

#[cfg(test)]
mod tests {
    use super::{AbortCode, HostOut};
    use crate::agent::AgentSlot;
    use crate::bool8::Bool8;
    use crate::step_out::{EpisodeEnd, StepOut};

    fn code(n: u32) -> AbortCode {
        AbortCode::new(n).unwrap()
    }

    #[test]
    fn test_abort_code_zero_is_none() {
        assert_eq!(AbortCode::new(0), None);
        assert_eq!(AbortCode::new(7).map(AbortCode::get), Some(7));
        assert_eq!(AbortCode::new(u32::MAX).map(AbortCode::get), Some(u32::MAX));
    }

    #[test]
    fn test_host_out_abort_wins_over_end_episode_in_either_order_and_last_code_wins() {
        let mut rewards = [0.0f32; 2];
        let (mut term, mut trunc, mut died, mut spawned) = (
            [Bool8::FALSE; 2],
            [Bool8::FALSE; 2],
            [Bool8::FALSE; 2],
            [Bool8::FALSE; 2],
        );
        let mut out = HostOut::new(StepOut::new(
            &mut rewards,
            &mut term,
            &mut trunc,
            &mut died,
            &mut spawned,
        ));
        assert_eq!(out.aborted(), None);
        out.end_episode(EpisodeEnd::Terminated);
        out.abort(code(3));
        assert_eq!(out.aborted(), Some(code(3)));
        out.end_episode(EpisodeEnd::Terminated);
        out.abort(code(9));
        out.end_episode(EpisodeEnd::Truncated);
        // The abort stays, with the last code; the engine applies it before
        // any episode end.
        assert_eq!(out.aborted(), Some(code(9)));
    }

    #[test]
    fn test_host_out_deref_forwards_every_step_out_call() {
        let mut rewards = [0.0f32; 2];
        let (mut term, mut trunc, mut died, mut spawned) = (
            [Bool8::FALSE; 2],
            [Bool8::FALSE; 2],
            [Bool8::FALSE; 2],
            [Bool8::FALSE; 2],
        );
        {
            let mut out = HostOut::new(StepOut::new(
                &mut rewards,
                &mut term,
                &mut trunc,
                &mut died,
                &mut spawned,
            ));
            let a0 = AgentSlot::new(0, 2).unwrap();
            let a1 = AgentSlot::new(1, 2).unwrap();
            out.set_reward(a0, 1.5);
            out.add_reward(a0, 0.5);
            out.terminate_agent(a0);
            out.truncate_agent(a1);
            out.agent_died(a0);
            out.agent_spawned(a1);
            out.end_episode(EpisodeEnd::Truncated);
            assert_eq!(out.episode_end(), Some(EpisodeEnd::Truncated));
        }
        assert_eq!(rewards, [2.0, 0.0]);
        assert_eq!(term, [Bool8::TRUE, Bool8::FALSE]);
        assert_eq!(trunc, [Bool8::FALSE, Bool8::TRUE]);
        assert_eq!(died, [Bool8::TRUE, Bool8::FALSE]);
        assert_eq!(spawned, [Bool8::FALSE, Bool8::TRUE]);
    }
}
