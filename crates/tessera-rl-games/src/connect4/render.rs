// SPDX-License-Identifier: MIT OR Apache-2.0

//! Connect 4 scenes (ADR-0083): the 7 by 6 board with one disc or hole
//! per cell, row 0 at the bottom.

use tessera_rl_core::Render;
use tessera_rl_core::render::{Px16, Rgba, SceneWriter};

use super::board::{HEIGHT, WIDTH, cell};
use super::{Connect4, Connect4Config, Connect4State};

/// Cell side in pixels.
const CELL: u16 = 80;

const BOARD: Rgba = Rgba::rgb(30, 70, 200);
const HOLE: Rgba = Rgba::rgb(235, 235, 235);
/// Disc colours of slot 0 and slot 1.
const DISCS: [Rgba; 2] = [Rgba::rgb(220, 40, 40), Rgba::rgb(240, 200, 30)];

/// The centre of cell index `k` along an axis, in pixels.
fn centre(k: u32) -> Px16 {
    k.checked_mul(u32::from(CELL))
        .and_then(|v| v.checked_add(u32::from(CELL / 2)))
        .and_then(|v| i32::try_from(v).ok())
        .map_or(Px16(i32::MAX), Px16::px)
}

impl Render for Connect4 {
    /// Background and one primitive per cell.
    const MAX_PRIMITIVES: usize = 1 + (WIDTH * HEIGHT) as usize;

    fn canvas(_: &Connect4Config) -> (u16, u16) {
        let side = |n: u32| {
            u16::try_from(n)
                .ok()
                .and_then(|n| n.checked_mul(CELL))
                .unwrap_or(CELL)
        };
        (side(WIDTH), side(HEIGHT))
    }

    fn draw(cfg: &Connect4Config, state: &Connect4State, scene: &mut SceneWriter<'_>) {
        let (w, h) = Self::canvas(cfg);
        let (w, h) = (Px16::px(i32::from(w)), Px16::px(i32::from(h)));
        scene.rect(Px16(0), Px16(0), w, h, BOARD);
        let radius = Px16::px(i32::from(CELL) * 2 / 5);
        for row in 0..HEIGHT {
            let Some(from_top) = (HEIGHT - 1).checked_sub(row) else {
                continue;
            };
            let y = centre(from_top);
            for col in 0..WIDTH {
                let bit = cell(col, row);
                let fill = DISCS
                    .iter()
                    .zip(state.stones)
                    .find_map(|(c, s)| (s & bit != 0).then_some(*c))
                    .unwrap_or(HOLE);
                scene.circle(centre(col), y, radius, fill);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::Render;
    use tessera_rl_core::render::{Prim, Px16, Rgba, SceneWriter};

    use super::{Connect4, Connect4Config, Connect4State};

    #[test]
    fn test_connect4_draw_played_board_starts_with_the_background() {
        let cfg = Connect4Config::default();
        let mut state = Connect4State::default();
        let first = state.play(3);
        let second = state.play(3);
        assert_ne!(first, second);
        let blank = Prim::Circle {
            cx: Px16(0),
            cy: Px16(0),
            r: Px16(0),
            fill: Rgba::default(),
        };
        let mut buf = vec![blank; Connect4::MAX_PRIMITIVES];
        let mut scene = SceneWriter::new(&mut buf);
        Connect4::draw(&cfg, &state, &mut scene);
        let (w, h) = Connect4::canvas(&cfg);
        assert!(!scene.overflowed());
        assert_eq!(scene.prims().len(), Connect4::MAX_PRIMITIVES);
        assert!(matches!(
            scene.prims().first(),
            Some(Prim::Rect { x0: Px16(0), y0: Px16(0), x1, y1, .. })
                if *x1 == Px16::px(i32::from(w)) && *y1 == Px16::px(i32::from(h))
        ));
    }
}
