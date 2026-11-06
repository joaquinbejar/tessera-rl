// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scripted baseline of snake.

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic};
use tessera_rl_num::rng::StepRng;

use super::{NO_FOOD, Snake, SnakeConfig, SnakeState, neighbour, opposite};

/// Scripted baseline: of the directions that are safe against the old bodies,
/// counting the agent's own tail as leaving, the one closest to the food; ties
/// go to the lowest direction. With none safe, straight on.
impl Heuristic for Snake {
    fn act(
        cfg: &SnakeConfig,
        s: &SnakeState,
        agent: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let me = agent.index();
        let facing = s.facing.get(me).copied().unwrap_or(0);
        let Some(head) = s.head_cell(me) else {
            out.set_discrete(0, i32::from(facing));
            return;
        };
        let mut occupied = s.occupancy();
        if let Some(tail) = s.tail_cell(me) {
            occupied.remove(tail);
        }
        let n = cfg.size.cast_signed();
        let distance = |cell: u16| {
            if s.food == NO_FOOD {
                return 0;
            }
            let (a, b) = (i32::from(cell), i32::from(s.food));
            (a / n - b / n).abs() + (a % n - b % n).abs()
        };
        let best = (0..4u8)
            .filter(|&d| d != opposite(facing))
            .filter_map(|d| neighbour(cfg.size, head, d).map(|c| (d, c)))
            .filter(|&(_, c)| !occupied.has(c))
            .min_by_key(|&(d, c)| (distance(c), d));
        out.set_discrete(0, i32::from(best.map_or(facing, |(d, _)| d)));
    }
}
