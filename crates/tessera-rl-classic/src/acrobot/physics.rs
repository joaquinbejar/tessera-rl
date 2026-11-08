// SPDX-License-Identifier: MIT OR Apache-2.0

//! Acrobot dynamics: Gymnasium `Acrobot-v1`, the "book" equations and one
//! fourth-order Runge-Kutta step of 0.2 s.
//!
//! The expressions follow Gymnasium's operation order exactly, so each
//! next-state component is reproducible against it (SPEC 18). Sine and
//! cosine come from the frozen math library (ADR-0008).

use core::f64::consts::PI;

use tessera_rl_num::math;

/// Integration step (s).
pub const DT: f64 = 0.2;
/// Bound of `dtheta1`.
pub const MAX_VEL_1: f64 = 4.0 * PI;
/// Bound of `dtheta2`.
pub const MAX_VEL_2: f64 = 9.0 * PI;

const M1: f64 = 1.0;
const M2: f64 = 1.0;
const L1: f64 = 1.0;
const LC1: f64 = 0.5;
const LC2: f64 = 0.5;
const I1: f64 = 1.0;
const I2: f64 = 1.0;
const G: f64 = 9.8;

/// The time derivative of `[theta1, theta2, dtheta1, dtheta2]` under
/// `torque`.
#[expect(
    clippy::similar_names,
    reason = "the names of Gymnasium's source, kept for comparison"
)]
#[inline]
fn dsdt(s: [f64; 4], torque: f64) -> [f64; 4] {
    let [theta1, theta2, dtheta1, dtheta2] = s;
    let d1 = M1 * (LC1 * LC1)
        + M2 * (L1 * L1 + LC2 * LC2 + 2.0 * L1 * LC2 * math::cos(theta2))
        + I1
        + I2;
    let d2 = M2 * (LC2 * LC2 + L1 * LC2 * math::cos(theta2)) + I2;
    let phi2 = M2 * LC2 * G * math::cos(theta1 + theta2 - PI / 2.0);
    let phi1 = -M2 * L1 * LC2 * (dtheta2 * dtheta2) * math::sin(theta2)
        - 2.0 * M2 * L1 * LC2 * dtheta2 * dtheta1 * math::sin(theta2)
        + (M1 * LC1 + M2 * L1) * G * math::cos(theta1 - PI / 2.0)
        + phi2;
    let ddtheta2 =
        (torque + d2 / d1 * phi1 - M2 * L1 * LC2 * (dtheta1 * dtheta1) * math::sin(theta2) - phi2)
            / (M2 * (LC2 * LC2) + I2 - d2 * d2 / d1);
    let ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;
    [dtheta1, dtheta2, ddtheta1, ddtheta2]
}

/// `y + h * k`, component by component.
#[expect(clippy::many_single_char_names, reason = "one component each")]
#[inline]
fn advance(y: [f64; 4], h: f64, k: [f64; 4]) -> [f64; 4] {
    let [a, b, c, d] = y;
    let [ka, kb, kc, kd] = k;
    [a + h * ka, b + h * kb, c + h * kc, d + h * kd]
}

/// `k1 + 2 * k2 + 2 * k3 + k4`, component by component, in that order.
#[inline]
fn weighted(k1: [f64; 4], k2: [f64; 4], k3: [f64; 4], k4: [f64; 4]) -> [f64; 4] {
    let mut out = [0.0; 4];
    for (o, (((a, b), c), d)) in out.iter_mut().zip(k1.iter().zip(&k2).zip(&k3).zip(&k4)) {
        *o = a + 2.0 * b + 2.0 * c + d;
    }
    out
}

/// `x` wrapped into `[m, big_m]` by Gymnasium's add and subtract loops.
/// Imported angles are validated into the range, and one RK4 step moves
/// them by a bounded amount, so the loops run a few times at most.
#[inline]
fn wrap(mut x: f64, m: f64, big_m: f64) -> f64 {
    let diff = big_m - m;
    while x > big_m {
        x -= diff;
    }
    while x < m {
        x += diff;
    }
    x
}

/// One step of `[theta1, theta2, dtheta1, dtheta2]` under `action` in
/// `0..3` (torque -1, 0 or +1).
#[must_use]
#[inline]
pub fn step(s: [f64; 4], action: i32) -> [f64; 4] {
    let torque = f64::from(action - 1);
    let dt2 = DT / 2.0;
    let k1 = dsdt(s, torque);
    let k2 = dsdt(advance(s, dt2, k1), torque);
    let k3 = dsdt(advance(s, dt2, k2), torque);
    let k4 = dsdt(advance(s, DT, k3), torque);
    let [w1, w2, w3, w4] = weighted(k1, k2, k3, k4);
    let [theta1, theta2, dtheta1, dtheta2] = s;
    let h = DT / 6.0;
    [
        wrap(theta1 + h * w1, -PI, PI),
        wrap(theta2 + h * w2, -PI, PI),
        (dtheta1 + h * w3).clamp(-MAX_VEL_1, MAX_VEL_1),
        (dtheta2 + h * w4).clamp(-MAX_VEL_2, MAX_VEL_2),
    ]
}

/// Whether the free end is above the bar by one link length.
#[must_use]
#[inline]
pub fn is_terminal(s: [f64; 4]) -> bool {
    let [theta1, theta2, _, _] = s;
    -math::cos(theta1) - math::cos(theta2 + theta1) > 1.0
}
