// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Refusals`: the per-agent verdicts a host environment writes for one
//! instance before its step (ADR-0070).

use core::num::NonZeroU32;

use crate::agent::AgentSlot;
use crate::bool8::Bool8;
use crate::header::AgentRow;

/// Why a host environment refused an agent's action: a non-zero code whose
/// meaning is the environment author's. Zero is reserved for "not refused"
/// in the outputs, so `RefuseCode::new(0)` is `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RefuseCode(NonZeroU32);

impl RefuseCode {
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

/// What a host `refuse` writes for one instance: one code per agent slot,
/// 0 where the agent is not refused (ADR-0070 item 1).
///
/// Only an alive, acting agent may be refused. Refusing any other slot is
/// an author bug: the engine records it, does not step the instance, and
/// poisons the pool after the call (ADR-0070 item 2).
#[derive(Debug)]
pub struct Refusals<'a> {
    codes: &'a mut [u32],
    refused: &'a mut [Bool8],
    agents: &'a [AgentRow],
    misuse: Option<AgentSlot>,
}

impl<'a> Refusals<'a> {
    /// Engine-only: views over one instance's `refuse_code` and `refused`
    /// rows, which the engine zeroed, and its agent rows.
    #[doc(hidden)]
    #[must_use]
    #[inline]
    pub const fn new(
        codes: &'a mut [u32],
        refused: &'a mut [Bool8],
        agents: &'a [AgentRow],
    ) -> Self {
        Self {
            codes,
            refused,
            agents,
            misuse: None,
        }
    }

    /// Engine-only: the first slot refused while it was not acting, if any.
    #[doc(hidden)]
    #[must_use]
    #[inline]
    pub const fn misuse(&self) -> Option<AgentSlot> {
        self.misuse
    }

    /// Refuses `agent`'s action in this call with `code`. The last call for
    /// an agent wins. `step` still runs, with the full joint action, and
    /// must not act on a refused agent's action
    /// ([`Actions::is_refused`](crate::io::Actions::is_refused)).
    #[inline]
    pub fn refuse(&mut self, agent: AgentSlot, code: RefuseCode) {
        let i = agent.index();
        let acting = self
            .agents
            .get(i)
            .is_some_and(|r| r.alive.get() && r.acting.get());
        if !acting {
            if self.misuse.is_none() {
                self.misuse = Some(agent);
            }
            return;
        }
        if let (Some(c), Some(r)) = (self.codes.get_mut(i), self.refused.get_mut(i)) {
            *c = code.get();
            *r = Bool8::TRUE;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Refusals, RefuseCode};
    use crate::agent::AgentSlot;
    use crate::bool8::Bool8;
    use crate::header::AgentRow;

    fn rows() -> [AgentRow; 3] {
        let mut r = [AgentRow::default(); 3];
        r[0].alive = Bool8::TRUE;
        r[0].acting = Bool8::TRUE;
        r[1].alive = Bool8::TRUE;
        r[2].acting = Bool8::TRUE;
        r
    }

    fn slot(i: u16) -> Option<AgentSlot> {
        AgentSlot::new(i, 3)
    }

    #[test]
    fn test_refuse_code_new_zero_is_none() {
        assert!(RefuseCode::new(0).is_none());
        assert_eq!(RefuseCode::new(9).map(RefuseCode::get), Some(9));
    }

    #[test]
    fn test_refusals_refuse_acting_agent_writes_rows() {
        let (mut codes, mut refused, agents) = ([0u32; 3], [Bool8::FALSE; 3], rows());
        let mut r = Refusals::new(&mut codes, &mut refused, &agents);
        if let (Some(a), Some(c)) = (slot(0), RefuseCode::new(4)) {
            r.refuse(a, c);
        }
        assert!(r.misuse().is_none());
        assert_eq!(codes, [4, 0, 0]);
        assert_eq!(refused, [Bool8::TRUE, Bool8::FALSE, Bool8::FALSE]);
    }

    #[test]
    fn test_refusals_refuse_non_acting_agent_records_misuse() {
        let (mut codes, mut refused, agents) = ([0u32; 3], [Bool8::FALSE; 3], rows());
        let mut r = Refusals::new(&mut codes, &mut refused, &agents);
        // Slot 1 is alive but not acting; slot 2 is acting but dead.
        if let (Some(a), Some(b), Some(c)) = (slot(1), slot(2), RefuseCode::new(4)) {
            r.refuse(a, c);
            r.refuse(b, c);
        }
        assert_eq!(r.misuse(), slot(1));
        assert_eq!(codes, [0, 0, 0]);
        assert_eq!(refused, [Bool8::FALSE; 3]);
    }
}
