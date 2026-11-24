// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rendering for the bindings (ADR-0083 item 4): an environment that
//! implements `Render` gets [`RenderFns`], monomorphized function pointers
//! that draw and rasterize one state; a pool without them refuses `render`
//! and `scene` with `CapabilityError`. Rendering reads snapshot records,
//! never live state, and runs on the caller's thread between calls.

use tessera_rl_core::render::Prim;
use tessera_rl_core::{Environment, Render};
use tessera_rl_render::{RenderError, Renderer};

use crate::env::BindError;

/// Rasterizes a state into a frame; returns whether the scene overflowed.
type FrameFn<E> = fn(
    &<E as Environment>::Config,
    &<E as Environment>::State,
    &mut [u8],
) -> Result<bool, BindError>;

/// A state's scene and whether it overflowed.
type SceneFn<E> = fn(
    &<E as Environment>::Config,
    &<E as Environment>::State,
) -> Result<(Vec<Prim>, bool), BindError>;

/// Draws and rasterizes states of `E` for one configuration.
pub(crate) struct RenderFns<E: Environment> {
    /// The canvas, `(width, height)`.
    pub(crate) canvas: (u16, u16),
    /// Rasterizes `state` into `out` (`width * height * 3` bytes); returns
    /// whether the scene overflowed.
    pub(crate) frame: FrameFn<E>,
    /// The scene of `state` and whether it overflowed.
    pub(crate) scene: SceneFn<E>,
}

fn render_error(e: RenderError) -> BindError {
    BindError::Argument(e.to_string())
}

fn frame<E: Render>(cfg: &E::Config, state: &E::State, out: &mut [u8]) -> Result<bool, BindError> {
    Renderer::<E>::new(cfg)
        .map_err(render_error)?
        .frame_into(cfg, state, out)
        .map_err(render_error)
}

fn scene<E: Render>(cfg: &E::Config, state: &E::State) -> Result<(Vec<Prim>, bool), BindError> {
    let mut renderer = Renderer::<E>::new(cfg).map_err(render_error)?;
    let s = renderer.scene(cfg, state);
    Ok((s.prims.to_vec(), s.overflowed))
}

/// The render functions of `E` for `cfg`.
///
/// # Errors
///
/// When `E` declares a canvas or a capacity out of range.
pub(crate) fn render_fns<E: Render>(cfg: &E::Config) -> Result<RenderFns<E>, BindError> {
    let canvas = Renderer::<E>::new(cfg).map_err(render_error)?.size();
    Ok(RenderFns {
        canvas,
        frame: frame::<E>,
        scene: scene::<E>,
    })
}

/// The refusal for a pool that cannot render.
pub(crate) fn not_renderable() -> BindError {
    BindError::Capability(
        "render and scene need a sync-mode pool of an environment that implements Render (ADR-0083)"
            .to_owned(),
    )
}

/// One primitive in the binary form `scene` returns: 144 bytes, little
/// endian: `kind: u32` (0 rect, 1 circle, 2 line, 3 polygon), `rgba: [u8; 4]`,
/// `count: u32` (values used), then 33 `i32` values in 1/16 pixel: rect
/// `x0 y0 x1 y1`, circle `cx cy r`, line `x0 y0 x1 y1 width`, polygon `n`
/// points as `x y` pairs.
pub(crate) const PRIM_RECORD: usize = 144;

fn put(values: &mut [i32], src: &[i32]) {
    for (d, s) in values.iter_mut().zip(src) {
        *d = *s;
    }
}

/// Appends `prim`'s record to `out`.
pub(crate) fn encode_prim(prim: &Prim, out: &mut Vec<u8>) {
    let mut values = [0_i32; 33];
    let (kind, fill, count) = match *prim {
        Prim::Rect {
            x0,
            y0,
            x1,
            y1,
            fill,
        } => {
            put(&mut values, &[x0.0, y0.0, x1.0, y1.0]);
            (0_u32, fill, 4_usize)
        }
        Prim::Circle { cx, cy, r, fill } => {
            put(&mut values, &[cx.0, cy.0, r.0]);
            (1, fill, 3)
        }
        Prim::Line {
            x0,
            y0,
            x1,
            y1,
            width,
            color,
        } => {
            put(&mut values, &[x0.0, y0.0, x1.0, y1.0, width.0]);
            (2, color, 5)
        }
        Prim::Polygon { points, n, fill } => {
            let mut used = 0;
            let (pairs, _) = values.as_chunks_mut::<2>();
            for (slot, &(x, y)) in pairs.iter_mut().zip(points.iter().take(usize::from(n))) {
                *slot = [x.0, y.0];
                used += 2;
            }
            (3, fill, used)
        }
    };
    out.extend_from_slice(&kind.to_le_bytes());
    out.extend_from_slice(&[fill.r, fill.g, fill.b, fill.a]);
    out.extend_from_slice(&u32::try_from(count).unwrap_or(0).to_le_bytes());
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}
