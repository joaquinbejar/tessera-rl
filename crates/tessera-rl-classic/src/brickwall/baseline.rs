// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scripted baseline of brick wall.

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic};
use tessera_rl_num::rng::StepRng;

use super::{BALL, BrickWall, BrickWallConfig, BrickWallState, LEFT, PADDLE_W, RIGHT, STAY, fx};

/// Scripted baseline: stay while the ball is parked; otherwise move the
/// paddle centre toward the ball centre, with a dead band of 4.
impl Heuristic for BrickWall {
    fn act(
        _: &BrickWallConfig,
        s: &BrickWallState,
        _: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let action = if s.parked() {
            STAY
        } else {
            let ball = s.x.saturating_add(fx(BALL / 2));
            let paddle = s.px.saturating_add(fx(PADDLE_W / 2));
            if ball > paddle.saturating_add(fx(4)) {
                RIGHT
            } else if ball < paddle.saturating_sub(fx(4)) {
                LEFT
            } else {
                STAY
            }
        };
        out.set_discrete(0, action);
    }
}
