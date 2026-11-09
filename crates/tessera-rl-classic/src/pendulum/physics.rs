// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pendulum dynamics: Gymnasium `Pendulum-v1`, in the reference's mixed
//! precision. The action `u` is `f32`; `u * u`, `0.001 * u * u` and `3 * u`
//! are computed in `f32` (`NumPy` 2 scalar promotion, NEP 50) and widened only
//! when added to `f64` terms. Everything else is `f64`, in the reference's
//! operation order. Squares are products (ADR-0008); the comparison with the
//! reference's platform `pow` is ADR-0050's.

use tessera_rl_num::math;

/// Largest torque.
pub const MAX_TORQUE: f32 = 2.0;
/// Largest angular speed (rad/s).
pub const MAX_SPEED: f64 = 8.0;
/// Integration step (s).
pub const DT: f64 = 0.05;
/// Gravity.
pub const GRAVITY: f64 = 10.0;

const TWO_PI: f64 = 2.0 * core::f64::consts::PI;

/// `NumPy`'s floor modulo of `a` by the positive `TWO_PI`.
#[inline]
fn floor_mod(a: f64) -> f64 {
    let r = math::fmod(a, TWO_PI);
    if r == 0.0 {
        return 0.0;
    }
    if r < 0.0 { r + TWO_PI } else { r }
}

/// The angle wrapped to `[-pi, pi]`, as Gymnasium's `angle_normalize`; `+pi`
/// occurs only by rounding, just below `-pi`.
#[must_use]
#[inline]
pub fn angle_normalize(x: f64) -> f64 {
    floor_mod(x + core::f64::consts::PI) - core::f64::consts::PI
}

/// The reward of `[theta, theta_dot]` under the torque `u`: `-costs`.
#[must_use]
#[inline]
pub fn reward(s: [f64; 2], u: f32) -> f64 {
    let [theta, theta_dot] = s;
    let u = u.clamp(-MAX_TORQUE, MAX_TORQUE);
    let angle = angle_normalize(theta);
    let costs = angle * angle + 0.1 * (theta_dot * theta_dot) + f64::from(0.001_f32 * (u * u));
    -costs
}

/// One step of `[theta, theta_dot]` under the torque `u`.
#[must_use]
#[inline]
pub fn step(s: [f64; 2], u: f32) -> [f64; 2] {
    let [theta, theta_dot] = s;
    let u = u.clamp(-MAX_TORQUE, MAX_TORQUE);
    let gravity = 3.0 * GRAVITY / 2.0 * math::sin(theta);
    let new_speed = theta_dot + (gravity + f64::from(3.0_f32 * u)) * DT;
    let new_speed = new_speed.clamp(-MAX_SPEED, MAX_SPEED);
    [theta + new_speed * DT, new_speed]
}

#[cfg(test)]
mod tests {
    use super::{angle_normalize, reward, step};
    use core::f64::consts::PI;

    #[test]
    fn test_angle_normalize_wraps_into_the_interval() {
        assert_eq!(angle_normalize(0.0), 0.0);
        assert_eq!(angle_normalize(PI), -PI);
        assert_eq!(angle_normalize(-PI), -PI);
        assert!((-PI..PI).contains(&angle_normalize(f64::MAX)));
        assert!((-PI..PI).contains(&angle_normalize(-1e300)));
        assert_eq!(angle_normalize((-PI).next_down()), PI);
    }

    #[test]
    fn test_step_upright_at_rest_with_no_torque_stays() {
        assert_eq!(step([0.0, 0.0], 0.0), [0.0, 0.0]);
    }

    #[test]
    fn test_step_clips_the_speed_both_ways() {
        assert_eq!(step([1.0, 8.0], 2.0)[1], 8.0);
        assert_eq!(step([-1.0, -8.0], -2.0)[1], -8.0);
    }

    #[test]
    fn test_step_closes_on_huge_angles() {
        for theta in [f64::MAX, -f64::MAX, 1e300, -1e300] {
            let next = step([theta, 8.0], 2.0);
            assert!(next[0].is_finite() && next[1].is_finite());
            assert!(reward([theta, 8.0], 2.0).is_finite());
        }
    }

    #[test]
    fn test_reward_is_zero_at_the_top_and_bounded() {
        assert_eq!(reward([0.0, 0.0], 0.0).to_bits(), (-0.0f64).to_bits());
        let lowest = -(PI * PI + 6.4 + 0.0041);
        assert!(reward([PI, 8.0], 2.0) > lowest);
    }
}
