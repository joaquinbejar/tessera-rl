// SPDX-License-Identifier: MIT OR Apache-2.0

//! Import validation of brick wall.

use tessera_rl_core::{HeaderView, InvariantViolation, Phase, Validate};
use tessera_rl_num::Fx32;

use super::{
    BALL_MAX_X, BrickWall, BrickWallConfig, BrickWallState, COLS, FULL, HEIGHT, MAX_SCORE,
    PADDLE_MAX, ROWS, SERVE, SERVE_WAIT, SPEED_X, SPEED_Y, WALL_POINTS, fx, points,
};

/// Points of the cleared bricks of the current wall, at most 216.
fn cleared_points(s: &BrickWallState) -> Option<u16> {
    let mut total: u16 = 0;
    for row in 0..ROWS {
        for col in 0..COLS {
            if !s.has(row, col) {
                total = total.checked_add(points(row))?;
            }
        }
    }
    Some(total)
}

/// Whether `s` is one of the two terminal states: parked with no wait, and
/// either out of balls with bricks left, or the second wall cleared.
#[must_use]
pub fn is_terminal(s: &BrickWallState) -> bool {
    s.parked() && s.wait == 0
}

/// Imported states, as the design note's Validate rules: committed ranges;
/// parked exactly when `vy` is 0, at the serve position with `vx` 0; a wait
/// only when parked; a parked state with no wait only as a terminal state;
/// bricks present except in the final terminal state; no balls only in a
/// terminal state; the score relation; unused bits zero.
impl Validate for BrickWall {
    fn validate(cfg: &BrickWallConfig, s: &BrickWallState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let within = |v: Fx32, lo: i32, hi: i32| fx(lo) <= v && v <= fx(hi);
        if !within(s.px, 0, PADDLE_MAX) {
            return invalid("a paddle out of range");
        }
        if !within(s.x, 0, BALL_MAX_X) || !within(s.y, 0, HEIGHT) {
            return invalid("a ball out of range");
        }
        if !within(s.vx, -SPEED_X, SPEED_X) {
            return invalid("a horizontal speed out of range");
        }
        if ![fx(-SPEED_Y), Fx32::ZERO, fx(SPEED_Y)].contains(&s.vy) {
            return invalid("a vertical speed other than -4, 0 or 4");
        }
        if s.wait > SERVE_WAIT {
            return invalid("a wait out of range");
        }
        if u32::from(s.lives) > cfg.lives || s.refills > 1 {
            return invalid("lives or refills out of range");
        }
        if s.bricks[1] & !FULL[1] != 0 {
            return invalid("a brick bit beyond the wall");
        }
        if s.parked() {
            if s.vx != Fx32::ZERO || s.x != fx(SERVE.0) || s.y != fx(SERVE.1) {
                return invalid("a parked ball off the serve position or moving");
            }
        } else if s.wait != 0 {
            return invalid("a wait while the ball moves");
        }
        let lost = s.lives == 0;
        let won = s.empty();
        if lost && won {
            return invalid("no balls and an empty wall at once");
        }
        if won && !(s.refills == 1 && is_terminal(s)) {
            return invalid("an empty wall outside the final terminal state");
        }
        if lost && !is_terminal(s) {
            return invalid("no balls outside a terminal state");
        }
        if is_terminal(s) && !lost && !won {
            return invalid("a parked ball with no wait in play");
        }
        let expected = WALL_POINTS
            .checked_mul(u16::from(s.refills))
            .zip(cleared_points(s))
            .and_then(|(walls, bricks)| walls.checked_add(bricks));
        if expected != Some(s.score) || s.score > MAX_SCORE {
            return invalid("a score that disagrees with the cleared bricks");
        }
        Ok(())
    }

    /// ADR-0040: before the first reset, exactly the initial value; after
    /// it, slot 0 alive and acting; a running episode is not terminal.
    fn validate_record(
        cfg: &BrickWallConfig,
        header: &HeaderView<'_>,
        s: &BrickWallState,
    ) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let phase = header.phase();
        if phase == Phase::NeverReset {
            if *s != BrickWallState::default() {
                return invalid("a state other than the initial value before the first reset");
            }
            return Ok(());
        }
        Self::validate(cfg, s)?;
        if let Some(view) = header.agent(0)
            && !(view.alive && view.acting)
        {
            return invalid("the agent not alive and acting after a reset");
        }
        if phase == Phase::Active && is_terminal(s) {
            return invalid("a running episode in a terminal state");
        }
        Ok(())
    }
}
