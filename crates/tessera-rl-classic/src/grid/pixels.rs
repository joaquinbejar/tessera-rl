// SPDX-License-Identifier: MIT OR Apache-2.0

//! Grid's pixel observation, shared by `tessera/Grid-v1` and
//! `tessera/GridLanes-v1` (#751): `[channel, row, column]`, where channel 0
//! lights the agent's cell, channel 1 the goal's and channel 2 the border
//! and the walls, 255 where lit and 0 elsewhere.
//!
//! Every byte is lit by exactly one rule of its channel, so the observation
//! is written sparsely: zero everything, then light only the agent, the
//! goal, the border and the walls. That is O(size) writes for the open
//! layout instead of three writes per cell, with the same bytes as the
//! cell-by-cell loop (the `tests` module keeps that loop as its reference).

use tessera_rl_core::leaf::LeafMut;

use super::GridConfig;

/// Where an observation is written: a leaf of the observation, or the raw
/// bytes of one instance's leaf in a lane group.
pub(crate) trait PixelSink {
    /// Sets every byte to 0.
    fn zero(&mut self);
    /// Sets byte `k` to 255; an index past the end writes nothing.
    fn lit(&mut self, k: usize);
}

impl PixelSink for LeafMut<'_, u8> {
    #[inline]
    fn zero(&mut self) {
        self.fill(0);
    }

    #[inline]
    fn lit(&mut self, k: usize) {
        self.set(k, 255);
    }
}

impl PixelSink for [u8] {
    #[inline]
    fn zero(&mut self) {
        self.fill(0);
    }

    #[inline]
    fn lit(&mut self, k: usize) {
        if let Some(b) = self.get_mut(k) {
            *b = 255;
        }
    }
}

/// Writes the observation of an agent at `agent` and a goal at `goal`
/// (column, row). A cell off the grid lights nothing: an agent that has
/// walked off is not drawn.
#[inline]
pub(crate) fn write<S: PixelSink + ?Sized>(
    cfg: &GridConfig,
    agent: (i32, i32),
    goal: (i32, i32),
    out: &mut S,
) {
    let n = usize::from(cfg.size);
    let cells = n * n;
    out.zero();
    let cell = |(col, row): (i32, i32)| {
        let (col, row) = (usize::try_from(col).ok()?, usize::try_from(row).ok()?);
        (col < n && row < n).then_some(row * n + col)
    };
    if let Some(k) = cell(agent) {
        out.lit(k);
    }
    if let Some(k) = cell(goal) {
        out.lit(cells + k);
    }
    let Some(last) = n.checked_sub(1) else {
        return;
    };
    let edges = 2 * cells;
    for col in 0..n {
        out.lit(edges + col);
        out.lit(edges + last * n + col);
    }
    for row in 1..last {
        out.lit(edges + row * n);
        out.lit(edges + row * n + last);
    }
    let layout = cfg.layout.get();
    if layout.wall_count() != 0 {
        for k in layout.wall_cells(cfg.size) {
            out.lit(edges + k);
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use tessera_rl_core::Asset;

    use super::write;
    use crate::grid::GridConfig;
    use crate::grid::layout::{self, GridLayout};

    /// The cell-by-cell loop this module replaces, verbatim from
    /// `grid_lanes::pixels` at e9b84677 (`Grid::observe` ran the same loop
    /// through `LeafMut::set`).
    fn pixels_reference(
        cfg: &GridConfig,
        agent: (i32, i32),
        goal: (i32, i32),
        mut set: impl FnMut(usize, u8),
    ) {
        let n = i32::from(cfg.size);
        let cells = usize::from(cfg.size) * usize::from(cfg.size);
        let lit = |on: bool| if on { 255 } else { 0 };
        for (k, (row, col)) in (0..n).flat_map(|r| (0..n).map(move |c| (r, c))).enumerate() {
            set(k, lit((col, row) == agent));
            set(cells + k, lit((col, row) == goal));
            set(2 * cells + k, lit(layout::border_or_wall(cfg, col, row)));
        }
    }

    fn reference(cfg: &GridConfig, agent: (i32, i32), goal: (i32, i32)) -> Vec<u8> {
        // Prefilled with a sentinel: the reference writes every byte.
        let mut out = vec![7u8; 3 * usize::from(cfg.size) * usize::from(cfg.size)];
        pixels_reference(cfg, agent, goal, |k, v| out[k] = v);
        out
    }

    fn sparse(cfg: &GridConfig, agent: (i32, i32), goal: (i32, i32)) -> Vec<u8> {
        let mut out = vec![7u8; 3 * usize::from(cfg.size) * usize::from(cfg.size)];
        write(cfg, agent, goal, out.as_mut_slice());
        out
    }

    fn cfg(size: u16, text: &[u8]) -> GridConfig {
        GridConfig {
            size,
            max_steps: 50,
            layout: Asset::from_bytes(text).unwrap(),
        }
    }

    /// A walled layout of side `n`: a cross of walls, start and corners free.
    fn cross(n: u16) -> Vec<u8> {
        let mid = n / 2;
        (0..n)
            .flat_map(|r| {
                (0..n)
                    .map(move |c| {
                        if (r == mid || c == mid) && (r, c) != (mid, mid) {
                            b'#'
                        } else {
                            b'.'
                        }
                    })
                    .chain(*b"\n")
            })
            .collect()
    }

    /// Every interior cell a wall except the start and one goal cell.
    fn dense(n: u16) -> Vec<u8> {
        (0..n)
            .flat_map(|r| {
                (0..n)
                    .map(move |c| {
                        if (r, c) == (0, 0) || (r, c) == (0, 1) {
                            b'.'
                        } else {
                            b'#'
                        }
                    })
                    .chain(*b"\n")
            })
            .collect()
    }

    fn configs() -> Vec<GridConfig> {
        let mut out = Vec::new();
        for n in [4u16, 5, 16, 64] {
            out.push(cfg(n, b""));
            out.push(cfg(n, &cross(n)));
            out.push(cfg(n, &dense(n)));
        }
        out.push(GridConfig {
            size: 8,
            max_steps: 50,
            layout: Asset::<GridLayout>::bundled("rooms8").unwrap(),
        });
        out
    }

    #[test]
    fn test_pixels_sparse_write_equals_cell_loop_on_every_cell() {
        for cfg in configs() {
            assert!(tessera_rl_core::EnvConfig::validate(&cfg).is_ok());
            let n = i32::from(cfg.size);
            let goals = [(0, 1), (n - 1, n - 1), (n / 2, 0)];
            for row in -2..n + 2 {
                for col in -2..n + 2 {
                    for &goal in &goals {
                        assert_eq!(
                            sparse(&cfg, (col, row), goal),
                            reference(&cfg, (col, row), goal),
                            "size {n} agent ({col}, {row}) goal {goal:?}"
                        );
                        // The goal sweeps the cells too, the agent fixed.
                        assert_eq!(
                            sparse(&cfg, goal, (col, row)),
                            reference(&cfg, goal, (col, row)),
                            "size {n} goal ({col}, {row})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_pixels_wall_cells_are_exactly_the_walls_in_range() {
        for text in [cross(8), dense(8), cross(5), Vec::new()] {
            let layout: Asset<GridLayout> = Asset::from_bytes(&text).unwrap();
            let layout = layout.get();
            // n below, at and above the side: cells are row-major in n.
            for n in [3u16, 5, 8, 9] {
                let got: Vec<usize> = layout.wall_cells(n).collect();
                let n = i32::from(n);
                let want: Vec<usize> = (0..n)
                    .flat_map(|r| (0..n).map(move |c| (c, r)))
                    .filter(|&(c, r)| layout.is_wall(c, r))
                    .map(|(c, r)| usize::try_from(r * n + c).unwrap())
                    .collect();
                assert_eq!(got, want, "n {n}");
            }
        }
    }

    proptest! {
        #[test]
        fn test_pixels_sparse_write_equals_cell_loop_on_random_states(
            pick in 0usize..13,
            col in -70i32..70,
            row in -70i32..70,
            goal_col in -70i32..70,
            goal_row in -70i32..70,
        ) {
            let cfgs = configs();
            let cfg = &cfgs[pick % cfgs.len()];
            let (agent, goal) = ((col, row), (goal_col, goal_row));
            prop_assert_eq!(sparse(cfg, agent, goal), reference(cfg, agent, goal));
        }
    }
}
