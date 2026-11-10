// SPDX-License-Identifier: MIT OR Apache-2.0

//! Import validation of pendulum (design note 4.2).

use tessera_rl_core::{HeaderView, InvariantViolation, Phase, Validate};

use super::physics::MAX_SPEED;
use super::{Pendulum, PendulumConfig, PendulumState};

/// Imported states: `theta` any finite value (the transition is closed on
/// them), `theta_dot` in `[-8, 8]`.
impl Validate for Pendulum {
    fn validate(_: &PendulumConfig, state: &PendulumState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let [theta, theta_dot] = state.s;
        if !theta.is_finite() || !theta_dot.is_finite() {
            return invalid("a non-finite component");
        }
        if !(-MAX_SPEED..=MAX_SPEED).contains(&theta_dot) {
            return invalid("an angular speed out of range");
        }
        Ok(())
    }

    /// ADR-0040: before the first reset, exactly the initial value, bit for
    /// bit; after it, slot 0 alive and acting. Pendulum never terminates, so
    /// a terminated record is refused.
    fn validate_record(
        cfg: &PendulumConfig,
        header: &HeaderView<'_>,
        state: &PendulumState,
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
        if phase == Phase::PendingReset && header.terminated() {
            return invalid("a terminated record: pendulum never terminates");
        }
        Ok(())
    }
}
