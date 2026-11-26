// SPDX-License-Identifier: MIT OR Apache-2.0

//! Brick wall's pixel observation: an 84 x 84 grey frame of the wall, the
//! paddle and the ball, written into a row the engine has already zeroed
//! (ADR-0058), so only the lit pixels are written.
//!
//! #755: each row of the wall is filled as runs of adjacent present bricks,
//! one `fill` per run and pixel row, instead of one per brick. The bytes are
//! those of the per-brick loop, which the `tests` module keeps as its
//! reference.

use tessera_rl_num::Fx32;

use super::{
    BALL, BRICK_H, BRICK_W, BrickWallState, COLS, HEIGHT, PADDLE_H, PADDLE_W, PADDLE_Y, PIXELS,
    ROWS, SHADE_BANDS, SHADE_OBJECT, WALL_TOP, WIDTH, fx,
};

/// Output cells `[lo, hi)` covered by the world interval `[a0, a1)` of a side
/// `world` long, clipped to the frame.
fn span(a0: Fx32, a1: Fx32, world: i32) -> (usize, usize) {
    let side = PIXELS.cast_signed();
    let lo = a0.to_raw() >> 16;
    let hi = (a1.to_raw().saturating_add(0xFFFF)) >> 16;
    let c0 = (lo * side).div_euclid(world).clamp(0, side);
    let c1 = (hi * side + world - 1).div_euclid(world).clamp(0, side);
    let index = |v: i32| usize::try_from(v).unwrap_or(0);
    (index(c0), index(c1.max(c0)))
}

/// `bytes` is zero on entry (ADR-0058): only the lit pixels are written.
pub(super) fn draw(s: &BrickWallState, bytes: &mut [u8]) {
    let side = PIXELS as usize;
    let mut paint = |x0: Fx32, x1: Fx32, y0: Fx32, y1: Fx32, shade: u8| {
        let (c0, c1) = span(x0, x1, WIDTH);
        let (r0, r1) = span(y0, y1, HEIGHT);
        for row in r0..r1 {
            if let Some(line) = bytes.get_mut(row * side + c0..row * side + c1) {
                line.fill(shade);
            }
        }
    };
    for row in 0..ROWS {
        let shade = SHADE_BANDS
            .get(usize::try_from(row / 2).unwrap_or(0))
            .copied()
            .unwrap_or(0);
        // #755: one fill per run of adjacent present bricks, not one per
        // brick. Adjacent bricks touch (x = BRICK_W * col) and share the row's
        // shade and pixel rows, and each brick's first cell is at most the
        // previous brick's end cell, so the run's span is exactly the union of
        // its bricks' spans: the same bytes with a fraction of the calls.
        let y = WALL_TOP + BRICK_H * row;
        let mut col = 0;
        while col < COLS {
            if !s.has(row, col) {
                col += 1;
                continue;
            }
            let start = col;
            while col < COLS && s.has(row, col) {
                col += 1;
            }
            paint(
                fx(BRICK_W * start),
                fx(BRICK_W * col),
                fx(y),
                fx(y + BRICK_H),
                shade,
            );
        }
    }
    paint(
        s.px,
        s.px.saturating_add(fx(PADDLE_W)),
        fx(PADDLE_Y),
        fx(PADDLE_Y + PADDLE_H),
        SHADE_OBJECT,
    );
    if !s.parked() {
        paint(
            s.x,
            s.x.saturating_add(fx(BALL)),
            s.y,
            s.y.saturating_add(fx(BALL)),
            SHADE_OBJECT,
        );
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use tessera_rl_num::Fx32;

    use super::{draw, span};
    use crate::brickwall::{
        BALL, BRICK_H, BRICK_W, BRICKS, BrickWallState, COLS, FULL, HEIGHT, PADDLE_H, PADDLE_W,
        PADDLE_Y, PIXELS, ROWS, SHADE_BANDS, SHADE_OBJECT, WALL_TOP, WIDTH, fx,
    };

    /// The per-brick painter this module replaces, verbatim from
    /// `brickwall/mod.rs` at f4b7e13a (`draw_pixels`).
    fn draw_reference(s: &BrickWallState, bytes: &mut [u8]) {
        let side = PIXELS as usize;
        let mut paint = |x0: Fx32, x1: Fx32, y0: Fx32, y1: Fx32, shade: u8| {
            let (c0, c1) = span(x0, x1, WIDTH);
            let (r0, r1) = span(y0, y1, HEIGHT);
            for row in r0..r1 {
                if let Some(line) = bytes.get_mut(row * side + c0..row * side + c1) {
                    line.fill(shade);
                }
            }
        };
        for row in 0..ROWS {
            let shade = SHADE_BANDS
                .get(usize::try_from(row / 2).unwrap_or(0))
                .copied()
                .unwrap_or(0);
            for col in 0..COLS {
                if s.has(row, col) {
                    let (x, y) = (BRICK_W * col, WALL_TOP + BRICK_H * row);
                    paint(fx(x), fx(x + BRICK_W), fx(y), fx(y + BRICK_H), shade);
                }
            }
        }
        paint(
            s.px,
            s.px.saturating_add(fx(PADDLE_W)),
            fx(PADDLE_Y),
            fx(PADDLE_Y + PADDLE_H),
            SHADE_OBJECT,
        );
        if !s.parked() {
            paint(
                s.x,
                s.x.saturating_add(fx(BALL)),
                s.y,
                s.y.saturating_add(fx(BALL)),
                SHADE_OBJECT,
            );
        }
    }

    fn frame(s: &BrickWallState, f: fn(&BrickWallState, &mut [u8])) -> Vec<u8> {
        // Zero, as the engine hands rows to `observe`.
        let mut out = vec![0u8; (PIXELS * PIXELS) as usize];
        f(s, &mut out);
        out
    }

    // `moving` sets vy, so the ball is drawn (not parked) or not.
    fn state(bricks: [u64; 2], px: i32, x: i32, y: i32, moving: bool) -> BrickWallState {
        BrickWallState {
            bricks: [bricks[0], bricks[1] & FULL[1]],
            px: Fx32::from_raw(px),
            x: Fx32::from_raw(x),
            y: Fx32::from_raw(y),
            vy: if moving { fx(-3) } else { Fx32::ZERO },
            ..BrickWallState::default()
        }
    }

    #[test]
    fn test_pixels_runs_equal_per_brick_fill_on_every_one_and_two_brick_wall() {
        // Every single brick and every pair, adjacent or not, in every row.
        for a in 0..BRICKS {
            for b in a..BRICKS {
                let mut bricks = [0u64; 2];
                for k in [a, b] {
                    bricks[(k / 64) as usize] |= 1 << (k % 64);
                }
                let s = state(
                    bricks,
                    fx(200).to_raw(),
                    fx(100).to_raw(),
                    fx(200).to_raw(),
                    true,
                );
                assert_eq!(frame(&s, draw), frame(&s, draw_reference), "bricks {a} {b}");
            }
        }
    }

    #[test]
    fn test_pixels_runs_equal_per_brick_fill_on_full_and_empty_walls() {
        for bricks in [
            [u64::MAX, u64::MAX],
            [0, 0],
            [0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA],
        ] {
            let s = state(
                bricks,
                fx(0).to_raw(),
                fx(570).to_raw(),
                fx(325).to_raw(),
                true,
            );
            assert_eq!(frame(&s, draw), frame(&s, draw_reference), "{bricks:x?}");
        }
    }

    proptest! {
        #[test]
        fn test_pixels_runs_equal_per_brick_fill_on_random_states(
            lo in any::<u64>(),
            hi in any::<u64>(),
            px in -(1i32 << 26)..(1i32 << 26),
            x in -(1i32 << 26)..(1i32 << 26),
            y in -(1i32 << 26)..(1i32 << 26),
            moving in any::<bool>(),
        ) {
            let s = state([lo, hi], px, x, y, moving);
            prop_assert_eq!(frame(&s, draw), frame(&s, draw_reference));
        }
    }
}
