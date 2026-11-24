// SPDX-License-Identifier: MIT OR Apache-2.0

//! Grid scenes (ADR-0083): one square per cell, the same three layers as
//! the observation (border and walls, goal, agent).

use tessera_rl_core::Render;
use tessera_rl_core::render::{Px16, Rgba, SceneWriter};

use super::layout::border_or_wall;
use super::{Grid, GridConfig, GridState};

/// Cell side in pixels.
const CELL: i32 = 16;
/// The largest `GridConfig::size` (its config range is `4..=64`).
const MAX_SIZE: usize = 64;

const BACKGROUND: Rgba = Rgba::rgb(20, 20, 20);
const WALL: Rgba = Rgba::rgb(128, 128, 128);
const GOAL: Rgba = Rgba::rgb(40, 200, 80);
const AGENT: Rgba = Rgba::rgb(230, 60, 50);

/// The pixel edge of cell index `k`; off-range indices clamp (documented:
/// they lie far outside any canvas).
fn edge(k: i32) -> Px16 {
    k.checked_mul(CELL).map_or(
        if k < 0 {
            Px16(i32::MIN)
        } else {
            Px16(i32::MAX)
        },
        Px16::px,
    )
}

/// Fills cell `(col, row)`.
fn cell(scene: &mut SceneWriter<'_>, col: i32, row: i32, fill: Rgba) {
    let next = |k: i32| k.checked_add(1).map_or(Px16(i32::MAX), edge);
    scene.rect(edge(col), edge(row), next(col), next(row), fill);
}

impl Render for Grid {
    /// Background, every cell as a wall at most, goal and agent.
    const MAX_PRIMITIVES: usize = MAX_SIZE * MAX_SIZE + 3;

    fn canvas(cfg: &GridConfig) -> (u16, u16) {
        let side = cfg.size.checked_mul(16).unwrap_or(1024);
        (side, side)
    }

    fn draw(cfg: &GridConfig, state: &GridState, scene: &mut SceneWriter<'_>) {
        let n = i32::from(cfg.size);
        scene.rect(Px16(0), Px16(0), edge(n), edge(n), BACKGROUND);
        for row in 0..n {
            for col in 0..n {
                if border_or_wall(cfg, col, row) {
                    cell(scene, col, row, WALL);
                }
            }
        }
        cell(
            scene,
            i32::from(state.goal[0]),
            i32::from(state.goal[1]),
            GOAL,
        );
        let (col, row) = state.cell();
        cell(scene, col, row, AGENT);
    }
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::render::{Prim, Px16, Rgba, SceneWriter};
    use tessera_rl_core::{Environment, Render};
    use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
    use tessera_rl_num::rng::{RngAuthority, StepRng};

    use super::{Grid, GridConfig, GridState};

    #[test]
    fn test_grid_draw_reset_state_starts_with_the_background() {
        let cfg = GridConfig::default();
        let identity = RngIdentity::new(Seed::new(7), LogicalId::new(0));
        let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
        let mut state = GridState::default();
        Grid::reset(&cfg, &mut rng, &mut state);
        let blank = Prim::Circle {
            cx: Px16(0),
            cy: Px16(0),
            r: Px16(0),
            fill: Rgba::default(),
        };
        let mut buf = vec![blank; Grid::MAX_PRIMITIVES];
        let mut scene = SceneWriter::new(&mut buf);
        Grid::draw(&cfg, &state, &mut scene);
        let (w, h) = Grid::canvas(&cfg);
        assert!(!scene.overflowed());
        assert_ne!(scene.prims().len(), 0);
        assert!(matches!(
            scene.prims().first(),
            Some(Prim::Rect { x0: Px16(0), y0: Px16(0), x1, y1, .. })
                if *x1 == Px16::px(i32::from(w)) && *y1 == Px16::px(i32::from(h))
        ));
    }
}
