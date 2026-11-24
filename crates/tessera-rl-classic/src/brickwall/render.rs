// SPDX-License-Identifier: MIT OR Apache-2.0

//! `BrickWall` scenes (ADR-0083): the 576 by 330 playfield at one pixel per
//! game unit, the standing bricks, the paddle and the ball.

use tessera_rl_core::Render;
use tessera_rl_core::render::{Px16, Rgba, SceneWriter};
use tessera_rl_num::Fx32;

use super::{
    BALL, BRICK_H, BRICK_W, BrickWall, BrickWallConfig, BrickWallState, COLS, HEIGHT, PADDLE_H,
    PADDLE_W, PADDLE_Y, ROWS, WALL_TOP, WIDTH,
};

const BACKGROUND: Rgba = Rgba::rgb(0, 0, 0);
/// Brick colours, two rows per band, top first.
const BANDS: [Rgba; 3] = [
    Rgba::rgb(200, 72, 72),
    Rgba::rgb(198, 108, 58),
    Rgba::rgb(162, 162, 42),
];
const PADDLE: Rgba = Rgba::rgb(200, 72, 72);
const BALL_FILL: Rgba = Rgba::rgb(236, 236, 236);

/// A 16.16 game coordinate as [`Px16`]: dropping 12 of its 16 fraction
/// bits is exact integer arithmetic (an arithmetic shift).
const fn fx_px16(v: Fx32) -> Px16 {
    Px16(v.to_raw() >> 12)
}

/// `a + b` whole pixels, clamped (documented: game coordinates are far
/// below the clamp).
fn add(a: i32, b: i32) -> Px16 {
    a.checked_add(b).map_or(Px16(i32::MAX), Px16::px)
}

impl Render for BrickWall {
    /// Background, 108 bricks, paddle and ball.
    const MAX_PRIMITIVES: usize = 112;

    fn canvas(_: &BrickWallConfig) -> (u16, u16) {
        (
            u16::try_from(WIDTH).unwrap_or(576),
            u16::try_from(HEIGHT).unwrap_or(330),
        )
    }

    fn draw(_: &BrickWallConfig, state: &BrickWallState, scene: &mut SceneWriter<'_>) {
        scene.rect(
            Px16(0),
            Px16(0),
            Px16::px(WIDTH),
            Px16::px(HEIGHT),
            BACKGROUND,
        );
        for row in 0..ROWS {
            let band = usize::try_from(row / 2).unwrap_or(0);
            let colour = BANDS.get(band).copied().unwrap_or(BACKGROUND);
            let top = WALL_TOP + BRICK_H * row;
            for col in 0..COLS {
                if state.has(row, col) {
                    let left = BRICK_W * col;
                    scene.rect(
                        Px16::px(left),
                        Px16::px(top),
                        add(left, BRICK_W),
                        add(top, BRICK_H),
                        colour,
                    );
                }
            }
        }
        let px = fx_px16(state.px);
        scene.rect(
            px,
            Px16::px(PADDLE_Y),
            Px16(px.0.checked_add(PADDLE_W * 16).unwrap_or(i32::MAX)),
            add(PADDLE_Y, PADDLE_H),
            PADDLE,
        );
        let (x, y) = (fx_px16(state.x), fx_px16(state.y));
        let span = |p: Px16| Px16(p.0.checked_add(BALL * 16).unwrap_or(i32::MAX));
        scene.rect(x, y, span(x), span(y), BALL_FILL);
    }
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::render::{Prim, Px16, Rgba, SceneWriter};
    use tessera_rl_core::{Environment, Render};
    use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
    use tessera_rl_num::rng::{RngAuthority, StepRng};

    use super::{BrickWall, BrickWallConfig, BrickWallState};

    #[test]
    fn test_brickwall_draw_reset_state_starts_with_the_background() {
        let cfg = BrickWallConfig::default();
        let identity = RngIdentity::new(Seed::new(7), LogicalId::new(0));
        let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
        let mut state = BrickWallState::default();
        BrickWall::reset(&cfg, &mut rng, &mut state);
        let blank = Prim::Circle {
            cx: Px16(0),
            cy: Px16(0),
            r: Px16(0),
            fill: Rgba::default(),
        };
        let mut buf = vec![blank; BrickWall::MAX_PRIMITIVES];
        let mut scene = SceneWriter::new(&mut buf);
        BrickWall::draw(&cfg, &state, &mut scene);
        let (w, h) = BrickWall::canvas(&cfg);
        assert!(!scene.overflowed());
        assert_eq!(scene.prims().len(), 111);
        assert!(matches!(
            scene.prims().first(),
            Some(Prim::Rect { x0: Px16(0), y0: Px16(0), x1, y1, .. })
                if *x1 == Px16::px(i32::from(w)) && *y1 == Px16::px(i32::from(h))
        ));
    }
}
