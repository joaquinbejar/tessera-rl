// SPDX-License-Identifier: MIT OR Apache-2.0

//! CartPole dynamics: Gymnasium `CartPole-v1`, explicit Euler.
//!
//! The expressions follow Gymnasium's operation order exactly, so each
//! next-state component is reproducible against it (SPEC 18). Sine and
//! cosine come from the frozen math library (ADR-0008).

use tessera_rl_num::math;

/// Gravity (m/s^2).
pub const GRAVITY: f64 = 9.8;
/// Cart mass (kg).
pub const MASS_CART: f64 = 1.0;
/// Pole mass (kg).
pub const MASS_POLE: f64 = 0.1;
/// Half the pole length (m).
pub const LENGTH: f64 = 0.5;
/// Magnitude of the push (N).
pub const FORCE_MAG: f64 = 10.0;
/// Integration step (s).
pub const TAU: f64 = 0.02;
/// The episode terminates when `|x|` exceeds this (m).
pub const X_THRESHOLD: f64 = 2.4;
/// The episode terminates when `|theta|` exceeds this: 12 degrees, computed
/// as Gymnasium does (`12 * 2 * pi / 360`).
pub const THETA_THRESHOLD: f64 = 12.0 * 2.0 * core::f64::consts::PI / 360.0;

const TOTAL_MASS: f64 = MASS_POLE + MASS_CART;
const POLEMASS_LENGTH: f64 = MASS_POLE * LENGTH;

/// One Euler step of `[x, x_dot, theta, theta_dot]` under `push_right`.
#[must_use]
#[inline]
pub fn step(s: [f64; 4], push_right: bool) -> [f64; 4] {
    let [x, x_dot, theta, theta_dot] = s;
    let force = if push_right { FORCE_MAG } else { -FORCE_MAG };
    let costheta = math::cos(theta);
    let sintheta = math::sin(theta);
    let temp = (force + POLEMASS_LENGTH * (theta_dot * theta_dot) * sintheta) / TOTAL_MASS;
    let thetaacc = (GRAVITY * sintheta - costheta * temp)
        / (LENGTH * (4.0 / 3.0 - MASS_POLE * (costheta * costheta) / TOTAL_MASS));
    let xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;
    [
        x + TAU * x_dot,
        x_dot + TAU * xacc,
        theta + TAU * theta_dot,
        theta_dot + TAU * thetaacc,
    ]
}

/// Whether `s` is outside the allowed cart position or pole angle.
#[must_use]
#[inline]
pub fn is_terminal(s: [f64; 4]) -> bool {
    let [x, _, theta, _] = s;
    x < -X_THRESHOLD || x > X_THRESHOLD || theta < -THETA_THRESHOLD || theta > THETA_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::{THETA_THRESHOLD, is_terminal, step};

    #[test]
    fn test_step_upright_at_rest_moves_only_the_cart_velocity() {
        let s = step([0.0; 4], true);
        assert_eq!(s[0], 0.0);
        assert!(s[1] > 0.0);
        assert_eq!(s[2], 0.0);
        assert!(s[3] < 0.0);
    }

    #[test]
    fn test_step_is_mirror_symmetric() {
        let s = [0.01, -0.02, 0.03, -0.04];
        let m = s.map(|v| -v);
        assert_eq!(step(m, false), step(s, true).map(|v| -v));
    }

    #[test]
    fn test_is_terminal_at_thresholds_is_exclusive() {
        assert!(!is_terminal([2.4, 0.0, THETA_THRESHOLD, 0.0]));
        assert!(is_terminal([2.400_000_000_000_001, 0.0, 0.0, 0.0]));
        assert!(is_terminal([0.0, 0.0, -0.21, 0.0]));
    }
}
