// SPDX-License-Identifier: MIT OR Apache-2.0

//! Othello scenes (ADR-0083): the 8 by 8 board, its grid lines and one
//! disc per occupied square, row 0 at the top.

use tessera_rl_core::Render;
use tessera_rl_core::render::{Px16, Rgba, SceneWriter};

use super::board::{SIDE, square};
use super::{Othello, OthelloConfig, OthelloState};

/// Square side in pixels.
const CELL: i32 = 64;
/// Grid line width in pixels.
const LINE: i32 = 2;

const BOARD: Rgba = Rgba::rgb(30, 120, 60);
const GRID: Rgba = Rgba::rgb(10, 50, 25);
/// Disc colours of slot 0 (black, moves first) and slot 1 (white).
const DISCS: [Rgba; 2] = [Rgba::rgb(20, 20, 20), Rgba::rgb(240, 240, 240)];

/// `k` cells plus `off` pixels, clamped (documented: board coordinates
/// are far below the clamp).
fn at(k: u32, off: i32) -> Px16 {
    i32::try_from(k)
        .ok()
        .and_then(|k| k.checked_mul(CELL))
        .and_then(|v| v.checked_add(off))
        .map_or(Px16(i32::MAX), Px16::px)
}

impl Render for Othello {
    /// Background, `2 * (SIDE + 1)` grid lines and one disc per square.
    const MAX_PRIMITIVES: usize = 1 + 2 * (SIDE as usize + 1) + (SIDE * SIDE) as usize;

    fn canvas(_: &OthelloConfig) -> (u16, u16) {
        let side = i32::try_from(SIDE)
            .ok()
            .and_then(|s| s.checked_mul(CELL))
            .and_then(|s| u16::try_from(s).ok())
            .unwrap_or(512);
        (side, side)
    }

    fn draw(_: &OthelloConfig, state: &OthelloState, scene: &mut SceneWriter<'_>) {
        let end = at(SIDE, 0);
        scene.rect(Px16(0), Px16(0), end, end, BOARD);
        for k in 0..=SIDE {
            scene.rect(at(k, -LINE / 2), Px16(0), at(k, LINE / 2), end, GRID);
            scene.rect(Px16(0), at(k, -LINE / 2), end, at(k, LINE / 2), GRID);
        }
        let radius = Px16::px(CELL * 7 / 16);
        for row in 0..SIDE {
            for col in 0..SIDE {
                let bit = square(col, row);
                let disc = DISCS
                    .iter()
                    .zip(state.stones)
                    .find_map(|(c, s)| (s & bit != 0).then_some(*c));
                if let Some(fill) = disc {
                    scene.circle(at(col, CELL / 2), at(row, CELL / 2), radius, fill);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::Render;
    use tessera_rl_core::render::{Prim, Px16, Rgba, SceneWriter};

    use super::{Othello, OthelloConfig, OthelloState};

    #[test]
    fn test_othello_draw_start_board_starts_with_the_background() {
        let cfg = OthelloConfig::default();
        let state = OthelloState::start();
        let blank = Prim::Circle {
            cx: Px16(0),
            cy: Px16(0),
            r: Px16(0),
            fill: Rgba::default(),
        };
        let mut buf = vec![blank; Othello::MAX_PRIMITIVES];
        let mut scene = SceneWriter::new(&mut buf);
        Othello::draw(&cfg, &state, &mut scene);
        let (w, h) = Othello::canvas(&cfg);
        assert!(!scene.overflowed());
        assert_eq!(scene.prims().len(), 1 + 18 + 4);
        assert!(matches!(
            scene.prims().first(),
            Some(Prim::Rect { x0: Px16(0), y0: Px16(0), x1, y1, .. })
                if *x1 == Px16::px(i32::from(w)) && *y1 == Px16::px(i32::from(h))
        ));
    }
}
