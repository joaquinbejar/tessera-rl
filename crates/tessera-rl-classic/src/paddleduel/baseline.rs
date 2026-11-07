// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scripted baseline of paddle duel.

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic};
use tessera_rl_num::rng::StepRng;

use super::{BALL, DOWN, PADDLE_H, PaddleDuel, PaddleDuelConfig, PaddleDuelState, STAY, UP, fx};

/// Scripted baseline: stay while the ball is parked; otherwise move the own
/// paddle's centre toward the ball's centre y, with a dead band of 4.
impl Heuristic for PaddleDuel {
    fn act(
        _: &PaddleDuelConfig,
        s: &PaddleDuelState,
        agent: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let own = if agent.index() == 0 { s.ly } else { s.ry };
        let action = if s.parked() {
            STAY
        } else {
            let ball = s.y.saturating_add(fx(BALL / 2));
            let paddle = own.saturating_add(fx(PADDLE_H / 2));
            if ball > paddle.saturating_add(fx(4)) {
                DOWN
            } else if ball < paddle.saturating_sub(fx(4)) {
                UP
            } else {
                STAY
            }
        };
        out.set_discrete(0, action);
    }
}
