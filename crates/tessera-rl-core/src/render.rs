// SPDX-License-Identifier: MIT OR Apache-2.0

//! Vector scenes (ADR-0083 item 1): what [`Render::draw`](crate::Render::draw)
//! writes from a snapshot's state. Frames are rasterized from a scene
//! (`tessera-rl-render`), so frames and scenes come from one source.
//!
//! Coordinates are [`Px16`], 1/16 of a pixel, with the origin at the top
//! left and `y` growing downwards. Primitives are drawn in order (painter's
//! algorithm). Colours are straight (not premultiplied) RGBA8.

/// A scene coordinate in 1/16 of a pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Px16(pub i32);

impl Px16 {
    /// One pixel.
    pub const ONE: Self = Self(16);

    /// `p` whole pixels. Clamped to `i32` (a documented clamp: such a
    /// coordinate lies far outside any canvas, at most 4096 pixels wide).
    #[must_use]
    pub const fn px(p: i32) -> Self {
        match p.checked_mul(16) {
            Some(v) => Self(v),
            None if p < 0 => Self(i32::MIN),
            None => Self(i32::MAX),
        }
    }
}

/// A straight-alpha RGBA8 colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Rgba {
    /// Red.
    pub r: u8,
    /// Green.
    pub g: u8,
    /// Blue.
    pub b: u8,
    /// Alpha: 255 is opaque.
    pub a: u8,
}

impl Rgba {
    /// An opaque colour.
    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }
}

/// The most points of one [`Prim::Polygon`].
pub const MAX_POLYGON_POINTS: usize = 16;

/// The largest canvas side, in pixels (ADR-0083 item 1).
pub const MAX_CANVAS: u16 = 4096;

/// The largest [`Render::MAX_PRIMITIVES`](crate::Render::MAX_PRIMITIVES).
pub const MAX_PRIMITIVES_LIMIT: usize = 65_536;

/// One scene primitive. Exhaustive on purpose: the scene format is a
/// contract (ADR-0083), and a new primitive is a versioned format change
/// that every rasterizer must handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Prim {
    /// The half-open rectangle `[x0, x1) x [y0, y1)`, filled.
    Rect {
        /// Left edge.
        x0: Px16,
        /// Top edge.
        y0: Px16,
        /// Right edge (exclusive).
        x1: Px16,
        /// Bottom edge (exclusive).
        y1: Px16,
        /// Fill colour.
        fill: Rgba,
    },
    /// A filled disc of radius `r` around `(cx, cy)`.
    Circle {
        /// Centre `x`.
        cx: Px16,
        /// Centre `y`.
        cy: Px16,
        /// Radius.
        r: Px16,
        /// Fill colour.
        fill: Rgba,
    },
    /// A segment from `(x0, y0)` to `(x1, y1)` of total width `width`, with
    /// round caps.
    Line {
        /// Start `x`.
        x0: Px16,
        /// Start `y`.
        y0: Px16,
        /// End `x`.
        x1: Px16,
        /// End `y`.
        y1: Px16,
        /// Total width.
        width: Px16,
        /// Colour.
        color: Rgba,
    },
    /// A filled polygon (even-odd rule) of `n` points from `points`.
    Polygon {
        /// The points; only the first `n` are used.
        points: [(Px16, Px16); MAX_POLYGON_POINTS],
        /// The number of points, 3 to [`MAX_POLYGON_POINTS`].
        n: u8,
        /// Fill colour.
        fill: Rgba,
    },
}

/// Appends primitives to a caller-owned buffer (ADR-0083 item 1). Never
/// allocates and never panics: a primitive that does not fit, or a
/// polygon with fewer than 3 or more than [`MAX_POLYGON_POINTS`] points,
/// is dropped and sets [`SceneWriter::overflowed`].
#[derive(Debug)]
pub struct SceneWriter<'a> {
    buf: &'a mut [Prim],
    len: usize,
    overflowed: bool,
}

impl<'a> SceneWriter<'a> {
    /// A writer over `buf`, empty.
    #[must_use]
    pub fn new(buf: &'a mut [Prim]) -> Self {
        Self {
            buf,
            len: 0,
            overflowed: false,
        }
    }

    /// Appends `prim`, or sets the overflow flag when the buffer is full.
    pub fn push(&mut self, prim: Prim) {
        match self.buf.get_mut(self.len) {
            Some(slot) => {
                *slot = prim;
                self.len += 1;
            }
            None => self.overflowed = true,
        }
    }

    /// A filled rectangle `[x0, x1) x [y0, y1)`.
    pub fn rect(&mut self, x0: Px16, y0: Px16, x1: Px16, y1: Px16, fill: Rgba) {
        self.push(Prim::Rect {
            x0,
            y0,
            x1,
            y1,
            fill,
        });
    }

    /// A filled disc.
    pub fn circle(&mut self, cx: Px16, cy: Px16, r: Px16, fill: Rgba) {
        self.push(Prim::Circle { cx, cy, r, fill });
    }

    /// A segment with round caps.
    pub fn line(&mut self, from: (Px16, Px16), to: (Px16, Px16), width: Px16, color: Rgba) {
        self.push(Prim::Line {
            x0: from.0,
            y0: from.1,
            x1: to.0,
            y1: to.1,
            width,
            color,
        });
    }

    /// A filled polygon of 3 to [`MAX_POLYGON_POINTS`] points.
    pub fn polygon(&mut self, pts: &[(Px16, Px16)], fill: Rgba) {
        let mut points = [(Px16(0), Px16(0)); MAX_POLYGON_POINTS];
        let (Some(dst), Ok(n)) = (points.get_mut(..pts.len()), u8::try_from(pts.len())) else {
            self.overflowed = true;
            return;
        };
        if pts.len() < 3 {
            self.overflowed = true;
            return;
        }
        dst.copy_from_slice(pts);
        self.push(Prim::Polygon { points, n, fill });
    }

    /// The primitives written so far.
    #[must_use]
    pub fn prims(&self) -> &[Prim] {
        self.buf.get(..self.len).unwrap_or(&[])
    }

    /// Whether a primitive was dropped.
    #[must_use]
    pub const fn overflowed(&self) -> bool {
        self.overflowed
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_POLYGON_POINTS, Prim, Px16, Rgba, SceneWriter};

    const RED: Rgba = Rgba::rgb(255, 0, 0);

    fn blank() -> [Prim; 2] {
        [Prim::Circle {
            cx: Px16(0),
            cy: Px16(0),
            r: Px16(0),
            fill: Rgba::default(),
        }; 2]
    }

    #[test]
    fn test_scene_writer_full_buffer_drops_and_flags() {
        let mut buf = blank();
        let mut w = SceneWriter::new(&mut buf);
        w.circle(Px16(1), Px16(1), Px16(1), RED);
        w.circle(Px16(2), Px16(2), Px16(2), RED);
        assert!(!w.overflowed());
        w.circle(Px16(3), Px16(3), Px16(3), RED);
        assert!(w.overflowed());
        assert_eq!(w.prims().len(), 2);
    }

    #[test]
    fn test_scene_writer_polygon_bad_point_count_dropped() {
        let mut buf = blank();
        let mut w = SceneWriter::new(&mut buf);
        w.polygon(&[(Px16(0), Px16(0)), (Px16(1), Px16(1))], RED);
        assert!(w.overflowed());
        assert_eq!(w.prims(), &[] as &[Prim]);

        let mut buf = blank();
        let mut w = SceneWriter::new(&mut buf);
        let many = [(Px16(0), Px16(0)); MAX_POLYGON_POINTS + 1];
        w.polygon(&many, RED);
        assert!(w.overflowed());
        assert_eq!(w.prims(), &[] as &[Prim]);
    }

    #[test]
    fn test_scene_writer_polygon_three_points_kept() {
        let mut buf = blank();
        let mut w = SceneWriter::new(&mut buf);
        w.polygon(
            &[(Px16(0), Px16(0)), (Px16(16), Px16(0)), (Px16(0), Px16(16))],
            RED,
        );
        assert!(!w.overflowed());
        assert!(matches!(w.prims(), [Prim::Polygon { n: 3, .. }]));
    }

    #[test]
    fn test_px16_px_scales_and_clamps() {
        assert_eq!(Px16::px(3), Px16(48));
        assert_eq!(Px16::px(-2), Px16(-32));
        assert_eq!(Px16::px(i32::MAX), Px16(i32::MAX));
        assert_eq!(Px16::px(i32::MIN), Px16(i32::MIN));
    }
}
