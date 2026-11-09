// SPDX-License-Identifier: MIT OR Apache-2.0

//! A standalone eight-lane CartPole step (design stage 1). The pool does not
//! use it. [`super::physics::step`] stays the reference; the lanes follow it
//! expression by expression and must match it bit for bit.
//!
//! A lane is on the fast path only if no intermediate can overflow: the high
//! word of each of `|x|`, `|x_dot|`, `|theta_dot|` is at most that of 2^32,
//! and `theta` is on the sine and cosine polynomial path. Every other lane,
//! NaN and infinity included, runs the scalar transition, whole, and the
//! scalar cast. The caller holds the canonical FP mode.

#![expect(
    clippy::inline_always,
    reason = "must inline into each target_feature kernel"
)]

use tessera_rl_num::math::lane::{self, F64Lanes};
use tessera_rl_simd::{Kernel, Resolved};

use super::physics::{
    self, FORCE_MAG, GRAVITY, LENGTH, MASS_POLE, POLEMASS_LENGTH, TAU, THETA_THRESHOLD, TOTAL_MASS,
    X_THRESHOLD,
};

/// High word of 2^32.
const FAST_MAGNITUDE_WORD: u32 = 0x41f0_0000;

/// Eight states as columns `[x, x_dot, theta, theta_dot]`.
pub type Columns = [[f64; 8]; 4];

/// What one lane step reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaneOutcome {
    /// Observations as `f32` columns.
    pub observations: [[f32; 8]; 4],
    /// Bit `i` set if lane `i` is outside the bounds.
    pub terminal: u8,
}

/// Steps `state` in place at `level`.
#[must_use]
pub fn step(level: Resolved, state: &mut Columns, push_right: [bool; 8]) -> LaneOutcome {
    level.dispatch(StepKernel { state, push_right })
}

struct StepKernel<'a> {
    state: &'a mut Columns,
    push_right: [bool; 8],
}

impl Kernel for StepKernel<'_> {
    type Output = LaneOutcome;

    #[inline(always)]
    fn run<L: F64Lanes>(self) -> LaneOutcome {
        step_lanes::<L>(self.state, self.push_right)
    }
}

#[inline(always)]
fn step_lanes<L: F64Lanes>(state: &mut Columns, push_right: [bool; 8]) -> LaneOutcome {
    let k = L::splat;
    let before = *state;
    let [x, x_dot, theta, theta_dot] = before.map(L::from_array);

    let fast = L::mask_and(
        L::mask_and(
            lane::high_word_le(x, FAST_MAGNITUDE_WORD),
            lane::high_word_le(x_dot, FAST_MAGNITUDE_WORD),
        ),
        L::mask_and(
            lane::high_word_le(theta_dot, FAST_MAGNITUDE_WORD),
            lane::in_polynomial_domain(theta),
        ),
    );

    let force = L::select(L::mask_from_array(push_right), k(FORCE_MAG), k(-FORCE_MAG));
    let costheta = lane::cos(theta);
    let sintheta = lane::sin(theta);
    let temp = (force + k(POLEMASS_LENGTH) * (theta_dot * theta_dot) * sintheta) / k(TOTAL_MASS);
    let thetaacc = (k(GRAVITY) * sintheta - costheta * temp)
        / (k(LENGTH) * (k(4.0 / 3.0) - k(MASS_POLE) * (costheta * costheta) / k(TOTAL_MASS)));
    let xacc = temp - k(POLEMASS_LENGTH) * thetaacc * costheta / k(TOTAL_MASS);
    let next = [
        x + k(TAU) * x_dot,
        x_dot + k(TAU) * xacc,
        theta + k(TAU) * theta_dot,
        theta_dot + k(TAU) * thetaacc,
    ];

    let [nx, _, ntheta, _] = next;
    let terminal = L::mask_or(
        L::mask_or(nx.lt(k(-X_THRESHOLD)), nx.gt(k(X_THRESHOLD))),
        L::mask_or(
            ntheta.lt(k(-THETA_THRESHOLD)),
            ntheta.gt(k(THETA_THRESHOLD)),
        ),
    );
    let mut outcome = LaneOutcome {
        observations: next.map(L::to_f32_array),
        terminal: L::mask_bits(terminal),
    };
    *state = next.map(L::to_array);

    let slow = !L::mask_bits(fast);
    if slow != 0 {
        scalar_lanes(&before, state, push_right, slow, &mut outcome);
    }
    outcome
}

/// The scalar transition of the lanes in `slow`, over their lane results.
#[inline(never)]
fn scalar_lanes(
    before: &Columns,
    state: &mut Columns,
    push_right: [bool; 8],
    slow: u8,
    outcome: &mut LaneOutcome,
) {
    for (lane, push) in push_right.into_iter().enumerate() {
        if slow >> lane & 1 == 0 {
            continue;
        }
        let row = before.map(|c| c.get(lane).copied().unwrap_or(f64::NAN));
        let next = physics::step(row, push);
        for ((column, obs), value) in state.iter_mut().zip(&mut outcome.observations).zip(next) {
            if let (Some(c), Some(o)) = (column.get_mut(lane), obs.get_mut(lane)) {
                *c = value;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the observation cast, as the scalar environment's"
                )]
                {
                    *o = value as f32;
                }
            }
        }
        let bit = 1u8 << lane;
        outcome.terminal =
            (outcome.terminal & !bit) | if physics::is_terminal(next) { bit } else { 0 };
    }
}
