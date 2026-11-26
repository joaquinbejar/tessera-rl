// SPDX-License-Identifier: MIT OR Apache-2.0

//! `GridLayout`: the wall map of the reference grid world, a
//! content-addressed asset (ADR-0077).
//!
//! Inputs are either the canonical bytes or text: one line per row, `.` for
//! a free cell and `#` for a wall, with `\n` or `\r\n` line ends and an
//! optional final line end. Empty input is the open layout (no walls, any
//! size). The canonical form is the marker byte `0xff` (never in text),
//! `side: u16` little-endian, then the walls as a row-major bitset, least
//! significant bit first, `ceil(side² / 8)` bytes, with unused bits zero.
//!
//! In the grid, a move into a wall leaves the agent in place, goals are
//! never drawn on walls, and the third observation channel shows walls with
//! the border. The default, the bundled `"open"` layout, has no walls and
//! reproduces version 1's trajectories.

use std::borrow::Cow;

use tessera_rl_core::{AssetData, AssetError, ConfigError};

use tessera_rl_num::Fx32;
use tessera_rl_num::rng::StepRng;

use super::GridConfig;

/// First byte of the canonical form.
const CANONICAL: u8 = 0xff;

/// Largest side, the config's maximum `size`.
pub const MAX_SIDE: u16 = 64;

/// A square wall map; side 0 is the open layout.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GridLayout {
    side: u16,
    walls: Vec<u8>,
    /// Number of wall cells, derived from `walls` (not canonical bytes).
    wall_count: u32,
}

impl GridLayout {
    /// Cells on a side; 0 for the open layout.
    #[must_use]
    pub const fn side(&self) -> u16 {
        self.side
    }

    /// Whether row-major cell `k` of an `n`-wide grid is a wall.
    #[inline]
    #[must_use]
    pub fn is_wall_at(&self, k: u32, n: u32) -> bool {
        let at = |v: u32| i32::try_from(v).unwrap_or(i32::MAX);
        self.is_wall(at(k % n.max(1)), at(k / n.max(1)))
    }

    /// Number of wall cells.
    #[must_use]
    pub const fn wall_count(&self) -> u32 {
        self.wall_count
    }

    /// Whether `(col, row)` is a wall. Cells outside the layout are not.
    #[inline]
    #[must_use]
    pub fn is_wall(&self, col: i32, row: i32) -> bool {
        let side = i32::from(self.side);
        if !(0..side).contains(&col) || !(0..side).contains(&row) {
            return false;
        }
        // In range: row * side + col < side² <= 4096.
        let k = usize::try_from(row * side + col).unwrap_or(0);
        self.walls
            .get(k / 8)
            .is_some_and(|b| b & (1 << (k % 8)) != 0)
    }

    fn with_walls(side: u16, walls: Vec<u8>) -> Self {
        let wall_count = walls.iter().map(|b| b.count_ones()).sum();
        Self {
            side,
            walls,
            wall_count,
        }
    }

    fn bitset_len(side: u16) -> usize {
        (usize::from(side) * usize::from(side)).div_ceil(8)
    }

    fn from_text(text: &[u8]) -> Result<Self, AssetError> {
        let invalid = |reason| AssetError::Invalid { reason };
        let body = text.strip_suffix(b"\n").unwrap_or(text);
        let body = body.strip_suffix(b"\r").unwrap_or(body);
        let rows: Vec<&[u8]> = body
            .split(|&b| b == b'\n')
            .map(|r| r.strip_suffix(b"\r").unwrap_or(r))
            .collect();
        let side = u16::try_from(rows.len()).map_err(|_| invalid("too many rows"))?;
        if side > MAX_SIDE {
            return Err(AssetError::DecodedTooLarge {
                limit: Self::MAX_DECODED_BYTES,
            });
        }
        let mut walls = vec![0u8; Self::bitset_len(side)];
        for (r, row) in rows.iter().enumerate() {
            if row.len() != usize::from(side) {
                return Err(invalid("rows must be as long as there are rows"));
            }
            for (c, &cell) in row.iter().enumerate() {
                let k = r * usize::from(side) + c;
                match cell {
                    b'.' => {}
                    b'#' => {
                        if let Some(b) = walls.get_mut(k / 8) {
                            *b |= 1 << (k % 8);
                        }
                    }
                    _ => return Err(invalid("cells must be `.` or `#`")),
                }
            }
        }
        Ok(Self::with_walls(side, walls))
    }

    fn from_canonical(bytes: &[u8]) -> Result<Self, AssetError> {
        let invalid = |reason| AssetError::Invalid { reason };
        let Some((&[lo, hi], rest)) = bytes.split_first_chunk::<2>() else {
            return Err(invalid("truncated"));
        };
        let side = u16::from_le_bytes([lo, hi]);
        if side > MAX_SIDE {
            return Err(AssetError::DecodedTooLarge {
                limit: Self::MAX_DECODED_BYTES,
            });
        }
        let len = Self::bitset_len(side);
        if rest.len() != len {
            return Err(invalid("bitset length does not match the side"));
        }
        // Bits past side² must be zero, so the canonical form is unique.
        let used = usize::from(side) * usize::from(side);
        if used % 8 != 0 && rest.last().is_some_and(|b| b >> (used % 8) != 0) {
            return Err(invalid("bits past the last cell are set"));
        }
        Ok(Self::with_walls(side, rest.to_vec()))
    }
}

impl AssetData for GridLayout {
    const MAX_DECODED_BYTES: usize = 3 + (64 * 64) / 8;

    fn decode(bytes: &[u8]) -> Result<Self, AssetError> {
        match bytes.split_first() {
            None => Ok(Self::default()),
            Some((&CANONICAL, rest)) => Self::from_canonical(rest),
            Some(_) => Self::from_text(bytes),
        }
    }

    fn canonical_bytes(&self) -> Cow<'_, [u8]> {
        let mut out = Vec::with_capacity(3 + self.walls.len());
        out.push(CANONICAL);
        out.extend_from_slice(&self.side.to_le_bytes());
        out.extend_from_slice(&self.walls);
        Cow::Owned(out)
    }

    fn bundled(name: &str) -> Option<&'static [u8]> {
        match name {
            "open" => Some(include_bytes!("layouts/open.txt")),
            "rooms8" => Some(include_bytes!("layouts/rooms8.txt")),
            _ => None,
        }
    }
}

/// The layout fits the grid, the start cell is free and a goal can be drawn.
pub(super) fn check_layout(cfg: &GridConfig) -> Result<(), ConfigError> {
    let layout = cfg.layout.get();
    if layout.side() == 0 {
        return Ok(());
    }
    let err = |reason| {
        Err(ConfigError {
            field: "layout",
            reason,
        })
    };
    if layout.side() != cfg.size {
        return err("must be the open layout or have `size` cells on a side");
    }
    if layout.is_wall(0, 0) {
        return err("the start cell (0, 0) must be free");
    }
    if free_cells(cfg) == 0 {
        return err("needs a free cell besides the start for the goal");
    }
    Ok(())
}

/// Free cells other than the start, where a goal can be drawn. The start
/// cell is free (validated), so it is not among the walls.
pub(super) fn free_cells(cfg: &GridConfig) -> u32 {
    (u32::from(cfg.size) * u32::from(cfg.size))
        .saturating_sub(1)
        .saturating_sub(cfg.layout.get().wall_count())
}

/// The goal: uniform over the free cells after the start, row-major. With
/// no walls it is version 1's draw, cell `rng.below(n² - 1) + 1`.
pub(crate) fn goal(cfg: &GridConfig, rng: &mut StepRng) -> [i16; 2] {
    let n = u32::from(cfg.size);
    let layout = cfg.layout.get();
    let others = core::num::NonZeroU32::new(free_cells(cfg)).unwrap_or(core::num::NonZeroU32::MIN);
    let mut j = rng.below_u32(others);
    let mut k = 1 + j;
    if layout.wall_count() != 0 {
        // The j-th free cell after the start; validation guarantees one.
        k = 1;
        while k < n * n && (layout.is_wall_at(k, n) || j > 0) {
            j -= u32::from(!layout.is_wall_at(k, n));
            k += 1;
        }
    }
    let coord = |v: u32| i16::try_from(v).unwrap_or(0);
    [coord(k % n), coord(k / n)]
}

/// The position after a move from `from` to `to`: a wall leaves the agent
/// in place (game rule).
pub(crate) fn enter(cfg: &GridConfig, from: [Fx32; 2], to: [Fx32; 2]) -> [Fx32; 2] {
    let wall = cfg
        .layout
        .get()
        .is_wall(to[0].to_raw() >> 16, to[1].to_raw() >> 16);
    if wall { from } else { to }
}

/// The third observation channel: the border of the grid and its walls.
pub(crate) fn border_or_wall(cfg: &GridConfig, col: i32, row: i32) -> bool {
    let n = i32::from(cfg.size);
    let border = row == 0 || col == 0 || row == n - 1 || col == n - 1;
    border || cfg.layout.get().is_wall(col, row)
}
