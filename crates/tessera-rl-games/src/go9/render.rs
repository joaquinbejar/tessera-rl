// SPDX-License-Identifier: MIT OR Apache-2.0

//! Go 9x9 scenes (ADR-0083): the board, its 9 by 9 lines and one stone
//! per occupied point, row 0 at the top.

use tessera_rl_core::Render;
use tessera_rl_core::render::{Px16, Rgba, SceneWriter};

use super::board::{SIDE, point};
use super::{Go9, Go9Config, Go9State};

/// Distance between lines, and the margin to the first line, in pixels.
const STEP: i32 = 48;
/// Line width in pixels.
const LINE: i32 = 2;

const BOARD: Rgba = Rgba::rgb(220, 179, 92);
const GRID: Rgba = Rgba::rgb(40, 30, 20);
/// Stone colours of slot 0 (black, moves first) and slot 1 (white).
const STONES: [Rgba; 2] = [Rgba::rgb(20, 20, 20), Rgba::rgb(240, 240, 240)];

/// Line `k` plus `off` pixels, clamped (documented: board coordinates
/// are far below the clamp).
fn at(k: u32, off: i32) -> Px16 {
    i32::try_from(k)
        .ok()
        .and_then(|k| k.checked_add(1))
        .and_then(|k| k.checked_mul(STEP))
        .and_then(|v| v.checked_add(off))
        .map_or(Px16(i32::MAX), Px16::px)
}

impl Render for Go9 {
    /// Background, `2 * SIDE` lines and one stone per point.
    const MAX_PRIMITIVES: usize = 1 + 2 * SIDE as usize + (SIDE * SIDE) as usize;

    fn canvas(_: &Go9Config) -> (u16, u16) {
        let side = i32::try_from(SIDE)
            .ok()
            .and_then(|s| s.checked_add(1))
            .and_then(|s| s.checked_mul(STEP))
            .and_then(|s| u16::try_from(s).ok())
            .unwrap_or(480);
        (side, side)
    }

    fn draw(_: &Go9Config, state: &Go9State, scene: &mut SceneWriter<'_>) {
        // The canvas edge is one step past the last line.
        let end = at(SIDE, 0);
        scene.rect(Px16(0), Px16(0), end, end, BOARD);
        let (first, last) = (at(0, -LINE / 2), at(SIDE - 1, LINE / 2));
        for k in 0..SIDE {
            scene.rect(at(k, -LINE / 2), first, at(k, LINE / 2), last, GRID);
            scene.rect(first, at(k, -LINE / 2), last, at(k, LINE / 2), GRID);
        }
        let (black, white) = state.sets();
        let radius = Px16::px(STEP * 7 / 16);
        for row in 0..SIDE {
            for col in 0..SIDE {
                let bit = point(col, row);
                let stone = STONES
                    .iter()
                    .zip([black, white])
                    .find_map(|(c, s)| (s & bit != 0).then_some(*c));
                if let Some(fill) = stone {
                    scene.circle(at(col, 0), at(row, 0), radius, fill);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::Render;
    use tessera_rl_core::render::{Prim, Px16, Rgba, SceneWriter};

    use super::{Go9, Go9Config, Go9State};

    #[test]
    fn test_go9_draw_played_board_starts_with_the_background() {
        let cfg = Go9Config::default();
        let mut state = Go9State::default();
        state.play(40);
        state.play(0);
        let blank = Prim::Circle {
            cx: Px16(0),
            cy: Px16(0),
            r: Px16(0),
            fill: Rgba::default(),
        };
        let mut buf = vec![blank; Go9::MAX_PRIMITIVES];
        let mut scene = SceneWriter::new(&mut buf);
        Go9::draw(&cfg, &state, &mut scene);
        let (w, h) = Go9::canvas(&cfg);
        assert!(!scene.overflowed());
        assert_eq!(scene.prims().len(), 1 + 18 + 2);
        assert!(matches!(
            scene.prims().first(),
            Some(Prim::Rect { x0: Px16(0), y0: Px16(0), x1, y1, .. })
                if *x1 == Px16::px(i32::from(w)) && *y1 == Px16::px(i32::from(h))
        ));
    }
}
