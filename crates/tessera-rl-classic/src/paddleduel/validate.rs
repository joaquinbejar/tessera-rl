// SPDX-License-Identifier: MIT OR Apache-2.0

//! Import validation of paddle duel.

use tessera_rl_core::{HeaderView, InvariantViolation, Phase, Validate};
use tessera_rl_num::Fx32;

use super::{
    BALL_MAX_X, BALL_MAX_Y, CENTRE, PADDLE_MAX, PaddleDuel, PaddleDuelConfig, PaddleDuelState,
    SERVE_WAIT, SPEED_X, SPEED_Y, fx,
};

/// Whether `s` is a terminal state: parked with no wait.
#[must_use]
pub fn is_terminal(s: &PaddleDuelState) -> bool {
    s.parked() && s.wait == 0
}

/// Imported states, as the design note's Validate rules: integer committed
/// values in range; parked exactly when `vx` is 0, at the centre with `vy` 0;
/// a wait only when parked; a parked state with no wait exactly when one
/// score is `max_score`; scores at most `max_score`, not both at it.
impl Validate for PaddleDuel {
    fn validate(cfg: &PaddleDuelConfig, s: &PaddleDuelState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let all = [s.ly, s.ry, s.x, s.y, s.vx, s.vy];
        if all.iter().any(|v| v.to_raw() & 0xFFFF != 0) {
            return invalid("a fractional position or speed");
        }
        let within = |v: Fx32, lo: i32, hi: i32| fx(lo) <= v && v <= fx(hi);
        if !within(s.ly, 0, PADDLE_MAX) || !within(s.ry, 0, PADDLE_MAX) {
            return invalid("a paddle out of range");
        }
        if !within(s.x, 0, BALL_MAX_X) || !within(s.y, 0, BALL_MAX_Y) {
            return invalid("a ball out of range");
        }
        if ![fx(-SPEED_X), Fx32::ZERO, fx(SPEED_X)].contains(&s.vx) {
            return invalid("a horizontal speed other than -6, 0 or 6");
        }
        if !within(s.vy, -SPEED_Y, SPEED_Y) {
            return invalid("a vertical speed out of range");
        }
        if s.wait > SERVE_WAIT {
            return invalid("a wait out of range");
        }
        if s.parked() {
            if s.vy != Fx32::ZERO || s.x != fx(CENTRE.0) || s.y != fx(CENTRE.1) {
                return invalid("a parked ball off the centre or moving");
            }
        } else if s.wait != 0 {
            return invalid("a wait while the ball moves");
        }
        let [left, right] = s.score.map(u32::from);
        if left > cfg.max_score || right > cfg.max_score {
            return invalid("a score above max_score");
        }
        let won = left == cfg.max_score || right == cfg.max_score;
        if left == right && won {
            return invalid("both scores at max_score");
        }
        if won != is_terminal(s) {
            return invalid("a winning score exactly when parked with no wait, violated");
        }
        Ok(())
    }

    /// ADR-0040: before the first reset, exactly the initial value; after it,
    /// used slots alive and acting and an unused slot neither; a running
    /// episode is not terminal.
    fn validate_record(
        cfg: &PaddleDuelConfig,
        header: &HeaderView<'_>,
        s: &PaddleDuelState,
    ) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let phase = header.phase();
        if phase == Phase::NeverReset {
            if *s != PaddleDuelState::default() {
                return invalid("a state other than the initial value before the first reset");
            }
            return Ok(());
        }
        Self::validate(cfg, s)?;
        for k in 0..header.agents() {
            let Some(view) = header.agent(k) else {
                continue;
            };
            let used = k < cfg.agents as usize;
            if used && !(view.alive && view.acting) {
                return invalid("a used slot not alive and acting after a reset");
            }
            if !used && (view.alive || view.acting) {
                return invalid("an unused slot that is alive or acting");
            }
        }
        if phase == Phase::Active && is_terminal(s) {
            return invalid("a running episode in a terminal state");
        }
        Ok(())
    }
}
