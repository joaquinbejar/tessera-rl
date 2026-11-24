// SPDX-License-Identifier: MIT OR Apache-2.0

//! Snake scenes (ADR-0083): the board, the food and each live snake from
//! tail to head, the head in a brighter shade.

use tessera_rl_core::Render;
use tessera_rl_core::render::{Px16, Rgba, SceneWriter};

use super::{MAX_AGENTS, NO_FOOD, RING, Snake, SnakeConfig, SnakeState};

/// Cell side in pixels.
const CELL: i32 = 16;

const BACKGROUND: Rgba = Rgba::rgb(20, 20, 20);
const FOOD: Rgba = Rgba::rgb(230, 60, 50);
/// Body and head colours per slot.
const SNAKES: [(Rgba, Rgba); MAX_AGENTS] = [
    (Rgba::rgb(40, 160, 70), Rgba::rgb(120, 230, 120)),
    (Rgba::rgb(50, 100, 200), Rgba::rgb(120, 170, 250)),
];

/// The pixel edge of cell index `k` (at most `MAX_SIZE`, so no overflow).
fn edge(k: i32) -> Px16 {
    k.checked_mul(CELL).map_or(Px16(i32::MAX), Px16::px)
}

/// Fills board cell `cell` (`row * size + col`).
fn fill(scene: &mut SceneWriter<'_>, size: u32, cell: u16, colour: Rgba) {
    let Some(n) = core::num::NonZeroU32::new(size) else {
        return;
    };
    let at = u32::from(cell);
    let (Ok(col), Ok(row)) = (i32::try_from(at % n), i32::try_from(at / n)) else {
        return;
    };
    let next = |k: i32| k.checked_add(1).map_or(Px16(i32::MAX), edge);
    scene.rect(edge(col), edge(row), next(col), next(row), colour);
}

impl Render for Snake {
    /// Background, food, and at most `RING` cells per slot.
    const MAX_PRIMITIVES: usize = 2 + RING * MAX_AGENTS;

    fn canvas(cfg: &SnakeConfig) -> (u16, u16) {
        let side = cfg
            .size
            .checked_mul(16)
            .and_then(|v| u16::try_from(v).ok())
            .unwrap_or(512);
        (side, side)
    }

    fn draw(cfg: &SnakeConfig, state: &SnakeState, scene: &mut SceneWriter<'_>) {
        let n = edge(i32::try_from(cfg.size).unwrap_or(32));
        scene.rect(Px16(0), Px16(0), n, n, BACKGROUND);
        if state.food != NO_FOOD {
            fill(scene, cfg.size, state.food, FOOD);
        }
        let agents = usize::try_from(cfg.agents).unwrap_or(0);
        for (slot, (body, head)) in SNAKES.into_iter().enumerate().take(agents) {
            if !state.is_alive(slot) {
                continue;
            }
            // The body first so the head is painted last.
            for cell in state.cells_of(slot).skip(1) {
                fill(scene, cfg.size, cell, body);
            }
            if let Some(cell) = state.head_cell(slot) {
                fill(scene, cfg.size, cell, head);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::render::{Prim, Px16, Rgba, SceneWriter};
    use tessera_rl_core::{Environment, Render};
    use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
    use tessera_rl_num::rng::{RngAuthority, StepRng};

    use super::{Snake, SnakeConfig, SnakeState};

    #[test]
    fn test_snake_draw_reset_state_starts_with_the_background() {
        let cfg = SnakeConfig::default();
        let identity = RngIdentity::new(Seed::new(7), LogicalId::new(0));
        let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
        let mut state = SnakeState::default();
        Snake::reset(&cfg, &mut rng, &mut state);
        let blank = Prim::Circle {
            cx: Px16(0),
            cy: Px16(0),
            r: Px16(0),
            fill: Rgba::default(),
        };
        let mut buf = vec![blank; Snake::MAX_PRIMITIVES];
        let mut scene = SceneWriter::new(&mut buf);
        Snake::draw(&cfg, &state, &mut scene);
        let (w, h) = Snake::canvas(&cfg);
        assert!(!scene.overflowed());
        assert!(scene.prims().len() > 1);
        assert!(matches!(
            scene.prims().first(),
            Some(Prim::Rect { x0: Px16(0), y0: Px16(0), x1, y1, .. })
                if *x1 == Px16::px(i32::from(w)) && *y1 == Px16::px(i32::from(h))
        ));
    }
}
