// SPDX-License-Identifier: MIT OR Apache-2.0

//! Mountain car dynamics: Gymnasium `MountainCar-v0`.
//!
//! The expressions follow Gymnasium's operation order exactly, so each
//! next-state component is reproducible against it (SPEC 18). The cosine
//! comes from the frozen math library (ADR-0008).

use tessera_rl_num::math;

/// Leftmost position (the wall).
pub const MIN_POSITION: f64 = -1.2;
/// Rightmost position.
pub const MAX_POSITION: f64 = 0.6;
/// Speed bound, both directions.
pub const MAX_SPEED: f64 = 0.07;
/// The goal: the episode terminates at or past this position.
pub const GOAL_POSITION: f64 = 0.5;
/// The goal velocity: terminating also needs at least this velocity.
pub const GOAL_VELOCITY: f64 = 0.0;
/// Engine force per unit of action.
pub const FORCE: f64 = 0.001;
/// Gravity term.
pub const GRAVITY: f64 = 0.0025;

/// One step of `[position, velocity]` under `action` in `0..3`.
#[must_use]
#[inline]
pub fn step(s: [f64; 2], action: i32) -> [f64; 2] {
    let [mut position, mut velocity] = s;
    velocity += f64::from(action - 1) * FORCE + math::cos(3.0 * position) * (-GRAVITY);
    velocity = velocity.clamp(-MAX_SPEED, MAX_SPEED);
    position += velocity;
    position = position.clamp(MIN_POSITION, MAX_POSITION);
    if position == MIN_POSITION && velocity < 0.0 {
        velocity = 0.0;
    }
    [position, velocity]
}

/// Whether `s` is at the goal: far enough right and not moving left.
#[must_use]
#[inline]
pub fn is_terminal(s: [f64; 2]) -> bool {
    let [position, velocity] = s;
    position >= GOAL_POSITION && velocity >= GOAL_VELOCITY
}

#[cfg(test)]
mod tests {
    use super::{MAX_SPEED, MIN_POSITION, is_terminal, step};

    #[test]
    fn test_step_at_the_left_wall_stops_a_leftward_car() {
        let s = step([MIN_POSITION, -MAX_SPEED], 0);
        assert_eq!(s, [MIN_POSITION, 0.0]);
        assert!(s[1].is_sign_positive());
    }

    #[test]
    fn test_step_clips_the_speed() {
        assert_eq!(step([-0.5, MAX_SPEED], 2)[1], MAX_SPEED);
        assert_eq!(step([-0.5, -MAX_SPEED], 0)[1], -MAX_SPEED);
    }

    #[test]
    fn test_is_terminal_at_the_goal_is_inclusive() {
        assert!(is_terminal([0.5, 0.0]));
        assert!(is_terminal([0.5, -0.0]));
        assert!(!is_terminal([0.5, -1e-12]));
        assert!(!is_terminal([0.499_999_999_999_999_9, 0.01]));
    }
}
