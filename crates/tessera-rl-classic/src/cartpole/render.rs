// SPDX-License-Identifier: MIT OR Apache-2.0

//! CartPole scenes (ADR-0083): Gymnasium's 600 by 400 layout, a cart on a
//! track and the pole as a rotated segment.

use tessera_rl_core::Render;
use tessera_rl_core::render::{Px16, Rgba, SceneWriter};
use tessera_rl_num::math;

use super::physics::{LENGTH, X_THRESHOLD};
use super::{CartPole, CartPoleConfig, CartPoleState};

/// Canvas width in pixels (`SCALE` repeats it as a float).
const WIDTH: u16 = 600;
/// Canvas height in pixels.
const HEIGHT: u16 = 400;
/// Pixels per metre: the track spans `2 * X_THRESHOLD` over the width.
const SCALE: f64 = 600.0 / (2.0 * X_THRESHOLD);
/// The cart's centre line, in pixels from the top.
const CART_Y: f64 = 300.0;
/// Cart size in pixels.
const CART_W: f64 = 50.0;
/// Cart height in pixels.
const CART_H: f64 = 30.0;
/// Pole width in pixels.
const POLE_W: i32 = 10;
/// The axle's radius in pixels.
const AXLE_R: i32 = 5;
/// The largest coordinate magnitude a scene uses, in pixels: a diverged
/// or NaN state is drawn off canvas rather than overflowing.
const CLAMP_PX: f64 = 65_536.0;

const BACKGROUND: Rgba = Rgba::rgb(255, 255, 255);
const TRACK: Rgba = Rgba::rgb(0, 0, 0);
const CART: Rgba = Rgba::rgb(0, 0, 0);
const POLE: Rgba = Rgba::rgb(202, 152, 101);
const AXLE: Rgba = Rgba::rgb(129, 132, 203);

/// A pixel coordinate as [`Px16`]: clamped to `±CLAMP_PX` (NaN maps to
/// 0), then scaled by 16 and truncated towards zero. The clamp keeps the
/// cast exact and in range; IEEE multiplication by 16 is exact.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the value is clamped to ±2^20, well inside i32"
)]
fn px16(v: f64) -> Px16 {
    if v.is_nan() {
        return Px16(0);
    }
    Px16((v.clamp(-CLAMP_PX, CLAMP_PX) * 16.0) as i32)
}

impl Render for CartPole {
    const MAX_PRIMITIVES: usize = 8;

    fn canvas(_: &CartPoleConfig) -> (u16, u16) {
        (WIDTH, HEIGHT)
    }

    fn draw(_: &CartPoleConfig, state: &CartPoleState, scene: &mut SceneWriter<'_>) {
        let [x, _, theta, _] = state.s;
        let (w, h) = (i32::from(WIDTH), i32::from(HEIGHT));
        scene.rect(Px16(0), Px16(0), Px16::px(w), Px16::px(h), BACKGROUND);
        scene.rect(
            Px16(0),
            px16(CART_Y),
            Px16::px(w),
            px16(CART_Y + 1.0),
            TRACK,
        );
        let cx = f64::from(WIDTH) / 2.0 + x * SCALE;
        scene.rect(
            px16(cx - CART_W / 2.0),
            px16(CART_Y - CART_H / 2.0),
            px16(cx + CART_W / 2.0),
            px16(CART_Y + CART_H / 2.0),
            CART,
        );
        let axle_y = CART_Y - CART_H / 4.0;
        let pole = SCALE * 2.0 * LENGTH;
        let tip = (
            cx + pole * math::sin(theta),
            axle_y - pole * math::cos(theta),
        );
        scene.line(
            (px16(cx), px16(axle_y)),
            (px16(tip.0), px16(tip.1)),
            Px16::px(POLE_W),
            POLE,
        );
        scene.circle(px16(cx), px16(axle_y), Px16::px(AXLE_R), AXLE);
    }
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::Render;
    use tessera_rl_core::render::{Prim, Px16, Rgba, SceneWriter};

    use super::{CartPole, CartPoleConfig, CartPoleState};

    #[test]
    fn test_cartpole_draw_default_starts_with_the_background() {
        let cfg = CartPoleConfig::default();
        let state = CartPoleState {
            s: [0.5, 0.0, 0.1, 0.0],
        };
        let blank = Prim::Circle {
            cx: Px16(0),
            cy: Px16(0),
            r: Px16(0),
            fill: Rgba::default(),
        };
        let mut buf = vec![blank; CartPole::MAX_PRIMITIVES];
        let mut scene = SceneWriter::new(&mut buf);
        CartPole::draw(&cfg, &state, &mut scene);
        let (w, h) = CartPole::canvas(&cfg);
        assert!(!scene.overflowed());
        assert_ne!(scene.prims().len(), 0);
        assert!(matches!(
            scene.prims().first(),
            Some(Prim::Rect { x0: Px16(0), y0: Px16(0), x1, y1, .. })
                if *x1 == Px16::px(i32::from(w)) && *y1 == Px16::px(i32::from(h))
        ));
    }
}
