// SPDX-License-Identifier: MIT OR Apache-2.0

//! `GridLanes`' lane kernels against its scalar `step` and `observe`, lane
//! by lane, at every level this CPU has (ADR-0090 4.8), on live, edge and
//! overflow-edge states; and the counter that proves the lane path runs.

#![expect(
    clippy::many_single_char_names,
    reason = "test vectors: one letter per column"
)]

use core::num::NonZeroU32;

use tessera_rl_core::{Asset, EpisodeEnd, LANES, LaneEnvironment, LaneOut};
use tessera_rl_num::Fx32;
use tessera_rl_simd::Resolved;

use super::{Ending, GridLanes, GridLanesState, GridOutcome, LANE_STEPS, pixels, transition};
use crate::grid::GridConfig;

const ROOMS: &[u8] =
    b"....#...\n....#...\n........\n....#...\n###.####\n....#...\n........\n....#...\n";
// Walls on the grid's last column and row, beside the edges.
const EDGES: &[u8] =
    b"...#...#\n.......#\n........\n.......#\n........\n.......#\n........\n#.#.#.#.\n";

fn configs() -> Vec<GridConfig> {
    vec![
        GridConfig {
            size: 6,
            max_steps: 40,
            ..GridConfig::default()
        },
        GridConfig {
            size: 8,
            max_steps: 3,
            layout: Asset::from_bytes(ROOMS).unwrap(),
        },
        GridConfig {
            size: 8,
            max_steps: 50,
            layout: Asset::from_bytes(EDGES).unwrap(),
        },
    ]
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 ^ self.0 >> 29
    }
    fn below(&mut self, n: i64) -> i64 {
        i64::try_from(self.next() % u64::try_from(n).unwrap()).unwrap()
    }
}

/// Raw positions: whole cells on and around the grid, half cells, and the
/// `i32` extremes a restored state could hold (the overflow mask).
fn position(r: &mut Rng, size: u16) -> Fx32 {
    let size = i64::from(size);
    let raw = match r.below(6) {
        0 => i32::MIN + i32::try_from(r.below(3)).unwrap() * (1 << 16),
        1 => i32::MAX - i32::try_from(r.below(3)).unwrap() * (1 << 16),
        2 => i32::try_from((r.below(size + 2) - 1) * (1 << 16) + (1 << 15)).unwrap(),
        3 => -(1 << 15),
        _ => i32::try_from((r.below(size + 2) - 1) * (1 << 16)).unwrap(),
    };
    Fx32::from_raw(raw)
}

fn state(r: &mut Rng, cfg: &GridConfig) -> GridLanesState {
    let size = i64::from(cfg.size);
    let steps = match r.below(4) {
        0 => u32::MAX - u32::try_from(r.below(2)).unwrap(),
        1 => u32::from(cfg.max_steps) - 1,
        _ => u32::try_from(r.below(i64::from(cfg.max_steps))).unwrap(),
    };
    GridLanesState {
        col: position(r, cfg.size),
        row: position(r, cfg.size),
        goal_col: i32::try_from(r.below(size)).unwrap(),
        goal_row: i32::try_from(r.below(size)).unwrap(),
        steps,
        outcome: GridOutcome::Running,
    }
}

/// The scalar step of one instance: the new state, reward and end, as
/// `GridLanes::step` writes them.
fn scalar(
    cfg: &GridConfig,
    s: GridLanesState,
    action: i32,
) -> (GridLanesState, f32, Option<EpisodeEnd>) {
    let mut s = s;
    let (reward, end) = match transition(cfg, &mut s, Some(action)) {
        Ending::Terminated(r) => (r, Some(EpisodeEnd::Terminated)),
        Ending::Truncated => (0.0, Some(EpisodeEnd::Truncated)),
        Ending::None => (0.0, None),
    };
    (s, reward, end)
}

/// `GridLanes::observe`'s bytes for one instance.
fn observe_scalar(cfg: &GridConfig, s: &GridLanesState) -> Vec<u8> {
    let mut row = vec![0u8; 3 * usize::from(cfg.size) * usize::from(cfg.size)];
    pixels(cfg, s.cell(), (s.goal_col, s.goal_row), |k, v| row[k] = v);
    row
}

fn lane_step(
    level: Resolved,
    cfg: &GridConfig,
    states: &[GridLanesState; LANES],
    actions: &[i32; LANES],
) -> ([GridLanesState; LANES], LaneOut) {
    let mut col = states.map(|s| s.col);
    let mut row = states.map(|s| s.row);
    let mut goal_col = states.map(|s| s.goal_col);
    let mut goal_row = states.map(|s| s.goal_row);
    let mut steps = states.map(|s| s.steps);
    let mut outcome = states.map(|s| s.outcome);
    let mut out = LaneOut::default();
    let mut group = (
        &mut col,
        &mut row,
        &mut goal_col,
        &mut goal_row,
        &mut steps,
        &mut outcome,
    );
    GridLanes::step_lanes(cfg, level, &mut group, actions, &mut out);
    let mut next = *states;
    for (l, s) in next.iter_mut().enumerate() {
        *s = GridLanesState {
            col: col[l],
            row: row[l],
            goal_col: goal_col[l],
            goal_row: goal_row[l],
            steps: steps[l],
            outcome: outcome[l],
        };
    }
    (next, out)
}

fn check_lanes(seed: u64, batches: usize) {
    let mut r = Rng(seed);
    for cfg in configs() {
        for _ in 0..batches {
            let states: [GridLanesState; LANES] = core::array::from_fn(|_| state(&mut r, &cfg));
            let actions: [i32; LANES] =
                core::array::from_fn(|_| i32::try_from(r.below(4)).unwrap());
            for level in Resolved::available() {
                let (lanes, out) = lane_step(level, &cfg, &states, &actions);
                for l in 0..LANES {
                    let (want, reward, end) = scalar(&cfg, states[l], actions[l]);
                    assert_eq!(lanes[l], want, "{level:?} lane {l}: {:?}", states[l]);
                    assert_eq!(out.reward[l].to_bits(), reward.to_bits());
                    assert_eq!(out.end[l], end, "{level:?} lane {l}");
                }
                // The observation rows, from the stepped columns.
                let len = 3 * usize::from(cfg.size) * usize::from(cfg.size);
                let mut rows = vec![0u8; len * LANES];
                let col = lanes.map(|s| s.col);
                let row = lanes.map(|s| s.row);
                let goal_col = lanes.map(|s| s.goal_col);
                let goal_row = lanes.map(|s| s.goal_row);
                let steps = lanes.map(|s| s.steps);
                let outcome = lanes.map(|s| s.outcome);
                let (mut a, mut b, mut c, mut d, mut e, mut f) =
                    (col, row, goal_col, goal_row, steps, outcome);
                let group = (&mut a, &mut b, &mut c, &mut d, &mut e, &mut f);
                GridLanes::observe_lanes(&cfg, &group, &mut [&mut rows[..]]);
                for l in 0..LANES {
                    assert_eq!(
                        &rows[l * len..(l + 1) * len],
                        observe_scalar(&cfg, &lanes[l]).as_slice(),
                        "{level:?} observation of lane {l}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_grid_lanes_kernel_matches_scalar_step_on_every_level() {
    check_lanes(1, 400);
}

#[test]
fn test_grid_lanes_kernel_matches_scalar_on_negative_half_cells() {
    // The floor of -0.5 and -1.0 cells is -1 (`to_raw() >> 16`), not 0.
    let cfg = &configs()[0];
    let mut states = [GridLanesState::default(); LANES];
    for (l, s) in states.iter_mut().enumerate() {
        s.goal_col = 3;
        s.goal_row = 3;
        s.col = Fx32::from_raw(if l % 2 == 0 { -(1 << 15) } else { -(1 << 16) });
        s.row = Fx32::from_raw(i32::try_from(l).unwrap() << 15);
    }
    let actions = [0, 1, 2, 3, 0, 1, 2, 3];
    for level in Resolved::available() {
        let (lanes, out) = lane_step(level, cfg, &states, &actions);
        for l in 0..LANES {
            let (want, _, end) = scalar(cfg, states[l], actions[l]);
            assert_eq!(lanes[l], want, "{level:?} lane {l}");
            assert_eq!(out.end[l], end);
        }
    }
}

#[test]
fn test_grid_lanes_kernel_floors_negative_cells_against_any_goal() {
    // The cell is `to_raw() >> 16`, an arithmetic shift: a position one cell
    // left of the grid is cell -1, never 65535 (a logical shift's result),
    // so a goal at 65535, outside any grid, is not reached. The lane contract
    // holds for every state, not only validated ones.
    let cfg = &configs()[0];
    let mut states = [GridLanesState::default(); LANES];
    for (l, s) in states.iter_mut().enumerate() {
        s.col = Fx32::from_raw(-(1 << 16));
        s.row = Fx32::from_raw(i32::try_from(l % 4).unwrap() << 16);
        s.goal_col = 65535;
        s.goal_row = s.row.to_raw() >> 16;
    }
    // North and south keep the agent one cell off the grid.
    let actions = [0, 1, 0, 1, 0, 1, 0, 1];
    for level in Resolved::available() {
        let (lanes, out) = lane_step(level, cfg, &states, &actions);
        for l in 0..LANES {
            let (want, reward, end) = scalar(cfg, states[l], actions[l]);
            assert_eq!(lanes[l], want, "{level:?} lane {l}");
            assert_eq!(want.outcome, GridOutcome::OutOfBounds);
            assert_eq!(out.reward[l].to_bits(), reward.to_bits());
            assert_eq!(out.end[l], end);
        }
    }
}

#[test]
fn test_grid_lanes_steps_full_groups_through_the_lane_kernel() {
    use tessera_rl_engine::{ActionInput, EnvPool, PoolOptions, Reseed};
    use tessera_rl_num::identity::Seed;
    // 16 instances, chunk 8: two full groups per step take step_lanes.
    let before = LANE_STEPS.load(core::sync::atomic::Ordering::Relaxed);
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(8).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(16).unwrap(), Seed::new(5))
    };
    let mut p = EnvPool::<GridLanes>::new(GridConfig::default(), options).unwrap();
    let _ = p.reset_all(Reseed::Keep).unwrap();
    let acts = vec![1i32; 16];
    let _ = p
        .step(ActionInput {
            discrete: &acts,
            continuous: &[],
        })
        .unwrap();
    let after = LANE_STEPS.load(core::sync::atomic::Ordering::Relaxed);
    assert!(
        after >= before + 2,
        "lane groups stepped: {}",
        after - before
    );
}
