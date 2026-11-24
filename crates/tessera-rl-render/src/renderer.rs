// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`Renderer`]: one environment's scene buffer and frame size, reused
//! across frames (ADR-0083 items 1 and 3).

use core::marker::PhantomData;

use tessera_rl_core::Render;
use tessera_rl_core::render::{MAX_CANVAS, MAX_PRIMITIVES_LIMIT, Prim, Px16, Rgba, SceneWriter};
use tessera_rl_num::fp;
use thiserror::Error;

use crate::raster::rasterize;

/// A rendering failure: a bad canvas or capacity declaration, or a frame
/// buffer of the wrong length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum RenderError {
    /// `Render::canvas` returned a side outside `1..=4096`.
    #[error("canvas {width}x{height} is outside 1..=4096 per side")]
    Canvas {
        /// The declared width.
        width: u16,
        /// The declared height.
        height: u16,
    },
    /// `Render::MAX_PRIMITIVES` is 0 or above 65536.
    #[error("MAX_PRIMITIVES {0} is outside 1..=65536")]
    Capacity(usize),
    /// A frame buffer whose length is not `width * height * 3`.
    #[error("frame buffer of {got} bytes, expected {expected}")]
    FrameLength {
        /// The expected length.
        expected: usize,
        /// The given length.
        got: usize,
    },
}

/// A scene: its primitives and whether some were dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scene<'a> {
    /// The primitives in drawing order.
    pub prims: &'a [Prim],
    /// Whether the scene overflowed `Render::MAX_PRIMITIVES` (or drew a
    /// polygon with a bad point count); the dropped primitives are missing.
    pub overflowed: bool,
}

/// Renders states of `E` for one configuration: the canvas size and the
/// preallocated scene buffer. Allocates once, at construction.
#[derive(Debug)]
pub struct Renderer<E: Render> {
    width: u16,
    height: u16,
    buf: Vec<Prim>,
    _env: PhantomData<fn() -> E>,
}

impl<E: Render> Renderer<E> {
    /// A renderer for `cfg`.
    ///
    /// # Errors
    ///
    /// [`RenderError::Canvas`] or [`RenderError::Capacity`] when the
    /// environment's declarations are out of range.
    pub fn new(cfg: &E::Config) -> Result<Self, RenderError> {
        let (width, height) = E::canvas(cfg);
        if !(1..=MAX_CANVAS).contains(&width) || !(1..=MAX_CANVAS).contains(&height) {
            return Err(RenderError::Canvas { width, height });
        }
        if !(1..=MAX_PRIMITIVES_LIMIT).contains(&E::MAX_PRIMITIVES) {
            return Err(RenderError::Capacity(E::MAX_PRIMITIVES));
        }
        let blank = Prim::Rect {
            x0: Px16(0),
            y0: Px16(0),
            x1: Px16(0),
            y1: Px16(0),
            fill: Rgba::default(),
        };
        Ok(Self {
            width,
            height,
            buf: vec![blank; E::MAX_PRIMITIVES],
            _env: PhantomData,
        })
    }

    /// The canvas size, `(width, height)` in pixels.
    #[must_use]
    pub const fn size(&self) -> (u16, u16) {
        (self.width, self.height)
    }

    /// The length of one frame, `width * height * 3` bytes.
    #[must_use]
    pub fn frame_len(&self) -> usize {
        usize::from(self.width) * usize::from(self.height) * 3
    }

    /// Draws `state` and returns its scene. `draw` runs under the canonical
    /// floating-point controls (ADR-0037), whatever the caller's, so a scene
    /// computed with floats does not depend on them.
    pub fn scene(&mut self, cfg: &E::Config, state: &E::State) -> Scene<'_> {
        let buf = &mut self.buf;
        let (len, overflowed) = fp::with_canonical(|| {
            let mut writer = SceneWriter::new(buf);
            E::draw(cfg, state, &mut writer);
            (writer.prims().len(), writer.overflowed())
        });
        Scene {
            prims: self.buf.get(..len).unwrap_or(&[]),
            overflowed,
        }
    }

    /// Draws `state` and rasterizes it into `out`, an `[height, width, 3]`
    /// RGB8 frame. Returns whether the scene overflowed.
    ///
    /// # Errors
    ///
    /// [`RenderError::FrameLength`] when `out` is not [`Self::frame_len`] long.
    pub fn frame_into(
        &mut self,
        cfg: &E::Config,
        state: &E::State,
        out: &mut [u8],
    ) -> Result<bool, RenderError> {
        let expected = self.frame_len();
        if out.len() != expected {
            return Err(RenderError::FrameLength {
                expected,
                got: out.len(),
            });
        }
        let (width, height) = (self.width, self.height);
        let scene = self.scene(cfg, state);
        fp::with_canonical(|| rasterize(scene.prims, width, height, out));
        Ok(scene.overflowed)
    }
}
