// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rendering from snapshots (ADR-0017, ADR-0083).
//!
//! An environment implements [`Render`](tessera_rl_core::Render): it draws
//! a snapshot's state into a vector scene. [`Renderer`] holds the scene
//! buffer for one configuration and rasterizes scenes into RGB8 frames with
//! an integer rasterizer ([`rasterize`]): one sample per pixel centre, no
//! anti-aliasing, integer blending. A scene built without floats gives the
//! same frame on every platform and thread count. Nothing on the
//! simulation path depends on this crate.

mod raster;
mod renderer;
mod video;

pub use raster::rasterize;
pub use renderer::{RenderError, Renderer, Scene};
pub use video::{VideoError, encode_gif};
