// SPDX-License-Identifier: MIT OR Apache-2.0

//! Import validation of acrobot (design note 4.2).

use tessera_rl_core::{HeaderView, InvariantViolation, Phase, Validate};

use core::f64::consts::PI;

use super::physics::{MAX_VEL_1, MAX_VEL_2, is_terminal};
use super::{Acrobot, AcrobotConfig, AcrobotState};

/// Imported states: finite, angles in `[-pi, pi]` (so the step's wrap
/// loops stay short), `dtheta1` within `4 pi`, `dtheta2` within `9 pi`.
impl Validate for Acrobot {
    fn validate(_: &AcrobotConfig, state: &AcrobotState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        if !state.s.iter().all(|v| v.is_finite()) {
            return invalid("a non-finite component");
        }
        let [t1, t2, d1, d2] = state.s;
        if !(-PI..=PI).contains(&t1) || !(-PI..=PI).contains(&t2) {
            return invalid("an angle out of range");
        }
        if !(-MAX_VEL_1..=MAX_VEL_1).contains(&d1) || !(-MAX_VEL_2..=MAX_VEL_2).contains(&d2) {
            return invalid("a velocity out of range");
        }
        Ok(())
    }

    /// ADR-0040: before the first reset, exactly the initial value, bit
    /// for bit; after it, slot 0 alive and acting, and the terminal
    /// predicate as the header's end flags say: false while running or
    /// when truncated only, true when terminated.
    fn validate_record(
        cfg: &AcrobotConfig,
        header: &HeaderView<'_>,
        state: &AcrobotState,
    ) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let phase = header.phase();
        if phase == Phase::NeverReset {
            if state.s.iter().any(|v| v.to_bits() != 0) {
                return invalid("a state other than the initial value before the first reset");
            }
            return Ok(());
        }
        Self::validate(cfg, state)?;
        if let Some(view) = header.agent(0)
            && !(view.alive && view.acting)
        {
            return invalid("the agent not alive and acting after a reset");
        }
        let terminal = is_terminal(state.s);
        let terminated = phase == Phase::PendingReset && header.terminated();
        if terminal != terminated {
            return invalid(if terminal {
                "a terminal state not reported as terminated"
            } else {
                "a terminated record in a non-terminal state"
            });
        }
        Ok(())
    }
}
