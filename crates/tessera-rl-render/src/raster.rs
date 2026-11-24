// SPDX-License-Identifier: MIT OR Apache-2.0

//! The integer rasterizer (ADR-0083 item 3): one sample per pixel, at its
//! centre, on the 1/16-pixel grid; binary coverage (no anti-aliasing);
//! source-over blending in integer arithmetic. No floats, so a scene gives
//! the same frame on every platform.

use tessera_rl_core::render::{Prim, Px16, Rgba};

/// Rasterizes `prims` in order into `out`, an `[height, width, 3]` RGB8
/// frame cleared to black first. `out.len()` must be `width * height * 3`;
/// the caller checks it ([`crate::Renderer`] does).
pub fn rasterize(prims: &[Prim], width: u16, height: u16, out: &mut [u8]) {
    out.fill(0);
    for prim in prims {
        let (x0, y0, x1, y1) = bbox(prim);
        let (px0, px1) = clip(x0, x1, width);
        let (py0, py1) = clip(y0, y1, height);
        for py in py0..py1 {
            let sy = i64::from(py) * 16 + 8;
            for px in px0..px1 {
                let sx = i64::from(px) * 16 + 8;
                if covers(prim, sx, sy) {
                    let at = (usize::from(py) * usize::from(width) + usize::from(px)) * 3;
                    if let Some(rgb) = out.get_mut(at..at + 3) {
                        blend(rgb, color(prim));
                    }
                }
            }
        }
    }
}

const fn c(p: Px16) -> i64 {
    p.0 as i64
}

/// The primitive's bounding box in 1/16 pixels (inclusive, generous).
fn bbox(prim: &Prim) -> (i64, i64, i64, i64) {
    match *prim {
        Prim::Rect { x0, y0, x1, y1, .. } => (c(x0), c(y0), c(x1), c(y1)),
        Prim::Circle { cx, cy, r, .. } => {
            let r = c(r).abs();
            (c(cx) - r, c(cy) - r, c(cx) + r, c(cy) + r)
        }
        Prim::Line {
            x0,
            y0,
            x1,
            y1,
            width,
            ..
        } => {
            let h = c(width).abs() / 2 + 1;
            (
                c(x0).min(c(x1)) - h,
                c(y0).min(c(y1)) - h,
                c(x0).max(c(x1)) + h,
                c(y0).max(c(y1)) + h,
            )
        }
        Prim::Polygon { points, n, .. } => {
            let pts = points.get(..usize::from(n)).unwrap_or(&[]);
            let mut b = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
            for &(x, y) in pts {
                b = (b.0.min(c(x)), b.1.min(c(y)), b.2.max(c(x)), b.3.max(c(y)));
            }
            b
        }
    }
}

/// The pixels `[lo, hi)` whose centres may lie in `[a, b]` (1/16 px),
/// clipped to `0..size`.
fn clip(a: i64, b: i64, size: u16) -> (u16, u16) {
    if a > b {
        return (0, 0);
    }
    let lo = (a - 8).div_euclid(16).clamp(0, i64::from(size));
    let hi = ((b - 8).div_euclid(16) + 1).clamp(0, i64::from(size));
    (
        u16::try_from(lo).unwrap_or(0),
        u16::try_from(hi).unwrap_or(0),
    )
}

fn covers(prim: &Prim, sx: i64, sy: i64) -> bool {
    match *prim {
        Prim::Rect { x0, y0, x1, y1, .. } => c(x0) <= sx && sx < c(x1) && c(y0) <= sy && sy < c(y1),
        Prim::Circle { cx, cy, r, .. } => {
            let (dx, dy, r) = (
                i128::from(sx - c(cx)),
                i128::from(sy - c(cy)),
                i128::from(c(r)),
            );
            dx * dx + dy * dy <= r * r
        }
        Prim::Line {
            x0,
            y0,
            x1,
            y1,
            width,
            ..
        } => within_segment(
            (c(x0), c(y0)),
            (c(x1), c(y1)),
            (sx, sy),
            i128::from(c(width).abs()),
        ),
        Prim::Polygon { points, n, .. } => {
            even_odd(points.get(..usize::from(n)).unwrap_or(&[]), sx, sy)
        }
    }
}

/// Whether `p` lies within `width / 2` of the segment `a`-`b` (round caps),
/// exactly, in integers: `4 * dist^2 <= width^2`.
fn within_segment(a: (i64, i64), b: (i64, i64), p: (i64, i64), width: i128) -> bool {
    let (dx, dy) = (i128::from(b.0 - a.0), i128::from(b.1 - a.1));
    let (px, py) = (i128::from(p.0 - a.0), i128::from(p.1 - a.1));
    let len2 = dx * dx + dy * dy;
    let t = px * dx + py * dy;
    let w2 = width * width;
    if len2 == 0 || t <= 0 {
        return 4 * (px * px + py * py) <= w2;
    }
    if t >= len2 {
        let (qx, qy) = (i128::from(p.0 - b.0), i128::from(p.1 - b.1));
        return 4 * (qx * qx + qy * qy) <= w2;
    }
    let cross = px * dy - py * dx;
    4 * cross * cross <= w2 * len2
}

/// The even-odd rule with half-open edges, in integers.
fn even_odd(pts: &[(Px16, Px16)], sx: i64, sy: i64) -> bool {
    let mut inside = false;
    let Some(&last) = pts.last() else {
        return false;
    };
    let mut prev = last;
    for &cur in pts {
        let (x0, y0, x1, y1) = (c(prev.0), c(prev.1), c(cur.0), c(cur.1));
        if (y0 > sy) != (y1 > sy) {
            // The edge crosses the horizontal through sy; is the crossing
            // right of sx? x = x0 + (sy - y0) * (x1 - x0) / (y1 - y0).
            let lhs = i128::from(sx - x0) * i128::from(y1 - y0);
            let rhs = i128::from(sy - y0) * i128::from(x1 - x0);
            if (y1 > y0 && lhs < rhs) || (y1 < y0 && lhs > rhs) {
                inside = !inside;
            }
        }
        prev = cur;
    }
    inside
}

const fn color(prim: &Prim) -> Rgba {
    match *prim {
        Prim::Rect { fill, .. } | Prim::Circle { fill, .. } | Prim::Polygon { fill, .. } => fill,
        Prim::Line { color, .. } => color,
    }
}

/// Source-over with straight alpha, rounded: `(s * a + d * (255 - a) + 127) / 255`.
fn blend(rgb: &mut [u8], src: Rgba) {
    let a = u32::from(src.a);
    for (d, s) in rgb.iter_mut().zip([src.r, src.g, src.b]) {
        let v = (u32::from(s) * a + u32::from(*d) * (255 - a) + 127) / 255;
        *d = u8::try_from(v).unwrap_or(u8::MAX);
    }
}

#[cfg(test)]
mod tests {
    use super::rasterize;
    use tessera_rl_core::render::{Prim, Px16, Rgba};

    const W: Rgba = Rgba::rgb(255, 255, 255);

    fn px(frame: &[u8], width: usize, x: usize, y: usize) -> [u8; 3] {
        let at = (y * width + x) * 3;
        let mut rgb = [0; 3];
        rgb.copy_from_slice(frame.get(at..at + 3).unwrap_or(&[7, 7, 7]));
        rgb
    }

    #[test]
    fn test_rasterize_rect_covers_half_open_pixels() {
        let mut f = vec![9; 4 * 4 * 3];
        let r = Prim::Rect {
            x0: Px16::px(1),
            y0: Px16::px(1),
            x1: Px16::px(3),
            y1: Px16::px(2),
            fill: W,
        };
        rasterize(&[r], 4, 4, &mut f);
        assert_eq!(px(&f, 4, 1, 1), [255; 3]);
        assert_eq!(px(&f, 4, 2, 1), [255; 3]);
        assert_eq!(px(&f, 4, 3, 1), [0; 3]);
        assert_eq!(px(&f, 4, 1, 2), [0; 3]);
        assert_eq!(px(&f, 4, 0, 0), [0; 3]);
    }

    #[test]
    fn test_rasterize_circle_centre_in_corner_out() {
        let mut f = vec![0; 8 * 8 * 3];
        let d = Prim::Circle {
            cx: Px16::px(4),
            cy: Px16::px(4),
            r: Px16::px(2),
            fill: W,
        };
        rasterize(&[d], 8, 8, &mut f);
        assert_eq!(px(&f, 8, 3, 3), [255; 3]);
        assert_eq!(px(&f, 8, 4, 4), [255; 3]);
        assert_eq!(px(&f, 8, 0, 0), [0; 3]);
        assert_eq!(px(&f, 8, 6, 6), [0; 3]);
    }

    #[test]
    fn test_rasterize_line_on_path_off_path() {
        let mut f = vec![0; 8 * 8 * 3];
        let l = Prim::Line {
            x0: Px16::px(0),
            y0: Px16(8 * 16 / 2 + 8),
            x1: Px16::px(8),
            y1: Px16(8 * 16 / 2 + 8),
            width: Px16::ONE,
            color: W,
        };
        rasterize(&[l], 8, 8, &mut f);
        assert_eq!(px(&f, 8, 2, 4), [255; 3]);
        assert_eq!(px(&f, 8, 2, 2), [0; 3]);
    }

    #[test]
    fn test_rasterize_polygon_triangle_inside_outside() {
        let mut points = [(Px16::px(0), Px16::px(0)); 16];
        points[1] = (Px16::px(8), Px16::px(0));
        points[2] = (Px16::px(0), Px16::px(8));
        let t = Prim::Polygon {
            points,
            n: 3,
            fill: W,
        };
        let mut f = vec![0; 8 * 8 * 3];
        rasterize(&[t], 8, 8, &mut f);
        assert_eq!(px(&f, 8, 1, 1), [255; 3]);
        assert_eq!(px(&f, 8, 6, 6), [0; 3]);
    }

    #[test]
    fn test_rasterize_half_alpha_blends_rounded() {
        let mut f = vec![0; 3];
        let back = Prim::Rect {
            x0: Px16::px(0),
            y0: Px16::px(0),
            x1: Px16::px(1),
            y1: Px16::px(1),
            fill: Rgba::rgb(100, 0, 0),
        };
        let over = Prim::Rect {
            x0: Px16::px(0),
            y0: Px16::px(0),
            x1: Px16::px(1),
            y1: Px16::px(1),
            fill: Rgba {
                r: 200,
                g: 0,
                b: 0,
                a: 128,
            },
        };
        rasterize(&[back, over], 1, 1, &mut f);
        // (200 * 128 + 100 * 127 + 127) / 255 = 150
        assert_eq!(f, [150, 0, 0]);
    }

    #[test]
    fn test_rasterize_rect_edge_on_sample_excluded() {
        // The right and bottom edges pass exactly through pixel 1's centre
        // (1 * 16 + 8): half-open, so pixel 1 is outside.
        let mut f = vec![0; 2 * 2 * 3];
        let r = Prim::Rect {
            x0: Px16(0),
            y0: Px16(0),
            x1: Px16(24),
            y1: Px16(24),
            fill: W,
        };
        rasterize(&[r], 2, 2, &mut f);
        assert_eq!(px(&f, 2, 0, 0), [255; 3]);
        assert_eq!(px(&f, 2, 1, 0), [0; 3]);
        assert_eq!(px(&f, 2, 0, 1), [0; 3]);
    }

    #[test]
    fn test_rasterize_circle_boundary_sample_included() {
        // Pixel 1's centre lies exactly at distance r from the centre.
        let mut f = vec![0; 3 * 3];
        let d = Prim::Circle {
            cx: Px16(8),
            cy: Px16(8),
            r: Px16(16),
            fill: W,
        };
        rasterize(&[d], 3, 1, &mut f);
        assert_eq!(px(&f, 3, 1, 0), [255; 3]);
        assert_eq!(px(&f, 3, 2, 0), [0; 3]);
    }

    #[test]
    fn test_rasterize_blend_rounds_to_nearest() {
        // 1 * 128 / 255 = 0.502: rounds to 1, truncation would give 0.
        let mut f = vec![0; 3];
        let r = Prim::Rect {
            x0: Px16(0),
            y0: Px16(0),
            x1: Px16(16),
            y1: Px16(16),
            fill: Rgba {
                r: 1,
                g: 0,
                b: 0,
                a: 128,
            },
        };
        rasterize(&[r], 1, 1, &mut f);
        assert_eq!(f, [1, 0, 0]);
    }

    #[test]
    fn test_rasterize_off_canvas_prims_ignored() {
        let mut f = vec![0; 2 * 2 * 3];
        let r = Prim::Rect {
            x0: Px16::px(-10),
            y0: Px16::px(-10),
            x1: Px16::px(-5),
            y1: Px16::px(-5),
            fill: W,
        };
        let d = Prim::Circle {
            cx: Px16(i32::MAX),
            cy: Px16(i32::MAX),
            r: Px16(i32::MAX),
            fill: W,
        };
        rasterize(&[r, d], 2, 2, &mut f);
        assert_eq!(f, vec![0; 12]);
    }
}
