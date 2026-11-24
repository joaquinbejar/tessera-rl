// SPDX-License-Identifier: MIT OR Apache-2.0

//! `PaddleDuel` scenes (ADR-0083): the 640 by 500 court at one pixel per
//! game unit, the centre line, both paddles and the ball.

use tessera_rl_core::Render;
use tessera_rl_core::render::{Px16, Rgba, SceneWriter};
use tessera_rl_num::Fx32;

use super::{
    BALL, HEIGHT, PADDLE_H, PADDLE_W, PaddleDuel, PaddleDuelConfig, PaddleDuelState, RIGHT_FACE,
    WIDTH,
};

const BACKGROUND: Rgba = Rgba::rgb(0, 0, 0);
const NET: Rgba = Rgba::rgb(60, 60, 60);
const LEFT_PADDLE: Rgba = Rgba::rgb(230, 80, 60);
const RIGHT_PADDLE: Rgba = Rgba::rgb(80, 140, 230);
const BALL_FILL: Rgba = Rgba::rgb(236, 236, 236);

/// A 16.16 game coordinate as [`Px16`]: dropping 12 of its 16 fraction
/// bits is exact integer arithmetic (an arithmetic shift).
const fn fx_px16(v: Fx32) -> Px16 {
    Px16(v.to_raw() >> 12)
}

/// `p` moved by `d` whole pixels, clamped (documented: game coordinates
/// are far below the clamp).
fn shifted(p: Px16, d: i32) -> Px16 {
    d.checked_mul(16)
        .and_then(|d| p.0.checked_add(d))
        .map_or(Px16(i32::MAX), Px16)
}

impl Render for PaddleDuel {
    /// Background, net, two paddles and the ball.
    const MAX_PRIMITIVES: usize = 8;

    fn canvas(_: &PaddleDuelConfig) -> (u16, u16) {
        (
            u16::try_from(WIDTH).unwrap_or(640),
            u16::try_from(HEIGHT).unwrap_or(500),
        )
    }

    fn draw(_: &PaddleDuelConfig, state: &PaddleDuelState, scene: &mut SceneWriter<'_>) {
        let (w, h) = (Px16::px(WIDTH), Px16::px(HEIGHT));
        scene.rect(Px16(0), Px16(0), w, h, BACKGROUND);
        let mid = Px16::px(WIDTH / 2);
        scene.rect(shifted(mid, -1), Px16(0), shifted(mid, 1), h, NET);
        let ly = fx_px16(state.ly);
        scene.rect(
            Px16(0),
            ly,
            Px16::px(PADDLE_W),
            shifted(ly, PADDLE_H),
            LEFT_PADDLE,
        );
        let ry = fx_px16(state.ry);
        scene.rect(
            Px16::px(RIGHT_FACE),
            ry,
            w,
            shifted(ry, PADDLE_H),
            RIGHT_PADDLE,
        );
        let (x, y) = (fx_px16(state.x), fx_px16(state.y));
        scene.rect(x, y, shifted(x, BALL), shifted(y, BALL), BALL_FILL);
    }
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::render::{Prim, Px16, Rgba, SceneWriter};
    use tessera_rl_core::{Environment, Render};
    use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
    use tessera_rl_num::rng::{RngAuthority, StepRng};

    use super::{PaddleDuel, PaddleDuelConfig, PaddleDuelState};

    #[test]
    fn test_paddleduel_draw_reset_state_starts_with_the_background() {
        let cfg = PaddleDuelConfig::default();
        let identity = RngIdentity::new(Seed::new(7), LogicalId::new(0));
        let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
        let mut state = PaddleDuelState::default();
        PaddleDuel::reset(&cfg, &mut rng, &mut state);
        let blank = Prim::Circle {
            cx: Px16(0),
            cy: Px16(0),
            r: Px16(0),
            fill: Rgba::default(),
        };
        let mut buf = vec![blank; PaddleDuel::MAX_PRIMITIVES];
        let mut scene = SceneWriter::new(&mut buf);
        PaddleDuel::draw(&cfg, &state, &mut scene);
        let (w, h) = PaddleDuel::canvas(&cfg);
        assert!(!scene.overflowed());
        assert_eq!(scene.prims().len(), 5);
        assert!(matches!(
            scene.prims().first(),
            Some(Prim::Rect { x0: Px16(0), y0: Px16(0), x1, y1, .. })
                if *x1 == Px16::px(i32::from(w)) && *y1 == Px16::px(i32::from(h))
        ));
    }
}
