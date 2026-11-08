// SPDX-License-Identifier: MIT OR Apache-2.0

//! Import validation of mountain car (design note 4.2).

use tessera_rl_core::{HeaderView, InvariantViolation, Phase, Validate};

use super::physics::{MAX_POSITION, MAX_SPEED, MIN_POSITION, is_terminal};
use super::{MountainCar, MountainCarConfig, MountainCarState};

/// Imported states: finite, position in `[-1.2, 0.6]`, velocity in
/// `[-0.07, 0.07]`, and never leftward at the wall (the step stops it).
impl Validate for MountainCar {
    fn validate(_: &MountainCarConfig, state: &MountainCarState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let [position, velocity] = state.s;
        if !position.is_finite() || !velocity.is_finite() {
            return invalid("a non-finite component");
        }
        if !(MIN_POSITION..=MAX_POSITION).contains(&position) {
            return invalid("a position out of range");
        }
        if !(-MAX_SPEED..=MAX_SPEED).contains(&velocity) {
            return invalid("a velocity out of range");
        }
        if position == MIN_POSITION && velocity < 0.0 {
            return invalid("a leftward velocity at the wall");
        }
        Ok(())
    }

    /// ADR-0040: before the first reset, exactly the initial value, bit
    /// for bit; after it, slot 0 alive and acting, and the terminal
    /// predicate as the header's end flags say: false while running or
    /// when truncated only, true when terminated.
    fn validate_record(
        cfg: &MountainCarConfig,
        header: &HeaderView<'_>,
        state: &MountainCarState,
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
