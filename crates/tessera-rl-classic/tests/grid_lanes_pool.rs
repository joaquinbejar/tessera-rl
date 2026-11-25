// SPDX-License-Identifier: MIT OR Apache-2.0

//! `tessera/GridLanes-v1` (ADR-0090 stage 2, #345) at pool level.
//!
//! - Against Grid: the same dynamics on a lane-wise state. At every SIMD
//!   level, in Inline and Sync with 0, 1 and 2 workers, `SameStep` and
//!   `NextStep`, with truncation, mid-group resets, forks and tails, the
//!   outputs (observations, rewards, ends, headers) are byte-identical and
//!   the states agree through the field mapping.
//! - Against `GridLanesRows` (the same fields on the row store): records,
//!   persisted files, replay files and hashes are byte-identical, and
//!   records cross the stores both ways.
//! - Async and heuristic slots refuse it; `chunk_size % 8 == 0` is required;
//!   observation transforms take the per-instance path and still match.

#![cfg(not(loom))]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

#[path = "support/grid_lanes_rows.rs"]
mod grid_lanes_rows;

use grid_lanes_rows::{GridLanesRows, GridRowsState};
use tessera_rl_classic::{Grid, GridConfig, GridLanes, GridLanesState, GridOutcome, GridState};
use tessera_rl_core::{Asset, Bool8, Environment};
use tessera_rl_engine::replay::{Recorder, Replay, ReplayWriterOptions};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AutoReset, DecodeLimits, DecodedRecord, EngineError, EnvPool,
    Outputs, PipelinedOptions, PoolOptions, Reseed, SyncOptions, Transforms,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_simd::{Level, Resolved};

const ROOMS: &[u8] =
    b"....#...\n....#...\n........\n....#...\n###.####\n....#...\n........\n....#...\n";
// Walls beside the last column and row, so moves at the edges meet walls.
const EDGES: &[u8] =
    b"...#...#\n.......#\n........\n.......#\n........\n.......#\n........\n#.#.#.#.\n";

/// Grid's golden config, a walled one with a short `max_steps` (terminal and
/// truncation on the same step), and edge walls.
fn configs() -> Vec<GridConfig> {
    vec![
        GridConfig {
            size: 6,
            max_steps: 40,
            ..GridConfig::default()
        },
        GridConfig {
            size: 8,
            max_steps: 7,
            layout: Asset::from_bytes(ROOMS).unwrap(),
        },
        GridConfig {
            size: 8,
            max_steps: 50,
            layout: Asset::from_bytes(EDGES).unwrap(),
        },
    ]
}

#[derive(Debug, Clone, Copy)]
struct Case {
    n: u32,
    chunk: u32,
    workers: usize,
    auto_reset: AutoReset,
    limit: Option<u32>,
    steps: u32,
    level: Option<Level>,
}

fn options(case: Case) -> PoolOptions {
    PoolOptions {
        chunk_size: NonZeroU32::new(case.chunk).unwrap(),
        max_episode_steps: case.limit.and_then(NonZeroU32::new),
        auto_reset: case.auto_reset,
        simd_level: case.level,
        ..PoolOptions::new(NonZeroU32::new(case.n).unwrap(), Seed::new(29))
    }
}

macro_rules! pool {
    ($env:ty, $cfg:expr, $case:expr) => {{
        let case: Case = $case;
        let p = EnvPool::<$env>::new($cfg.clone(), options(case)).unwrap();
        if case.workers == 0 {
            p
        } else {
            p.with_workers(SyncOptions::new(NonZeroUsize::new(case.workers).unwrap()))
                .unwrap()
        }
    }};
}

macro_rules! records {
    ($p:expr, $n:expr) => {
        (0..$n)
            .map(|i| $p.snapshot(i).unwrap().to_bytes())
            .collect::<Vec<Vec<u8>>>()
    };
}

fn action(t: u32, i: u32) -> i32 {
    let h = (u64::from(i).wrapping_mul(0x9e37_79b9) ^ u64::from(t).wrapping_mul(0x85eb_ca6b))
        .wrapping_mul(0xc2b2_ae35);
    i32::try_from((h >> 31) & 3).unwrap()
}

fn flags(v: &[Bool8]) -> Vec<u64> {
    v.iter().map(|b| u64::from(b.get())).collect()
}

fn bits(v: &[f32]) -> Vec<u64> {
    v.iter().map(|x| u64::from(x.to_bits())).collect()
}

fn outputs(o: &Outputs, same_step: bool) -> Vec<Vec<u64>> {
    let mut all = vec![
        bits(o.rewards()),
        flags(o.agent_terminated()),
        flags(o.agent_truncated()),
        flags(o.episode_terminated()),
        flags(o.episode_truncated()),
        bits(o.episode_return()),
        bits(o.episode_length()),
        flags(o.alive()),
        flags(o.acting()),
        flags(o.pending_reset()),
        o.observation(0)
            .unwrap()
            .iter()
            .map(|&b| u64::from(b))
            .collect(),
    ];
    if same_step {
        all.push(
            o.terminal_observation(0)
                .unwrap()
                .iter()
                .map(|&b| u64::from(b))
                .collect(),
        );
    }
    all
}

/// Grid's state of a `GridLanesState`: the fields narrowed back, the
/// outcome split into Grid's two flags.
fn to_grid(s: GridLanesState) -> GridState {
    GridState {
        pos: [s.col, s.row],
        goal: [
            i16::try_from(s.goal_col).unwrap(),
            i16::try_from(s.goal_row).unwrap(),
        ],
        steps: u16::try_from(s.steps).unwrap(),
        reached_goal: Bool8::from(s.outcome == GridOutcome::ReachedGoal),
        out_of_bounds: Bool8::from(s.outcome == GridOutcome::OutOfBounds),
    }
}

fn of_grid(s: GridState) -> GridLanesState {
    assert!(!(s.reached_goal.get() && s.out_of_bounds.get()));
    GridLanesState {
        col: s.pos[0],
        row: s.pos[1],
        goal_col: i32::from(s.goal[0]),
        goal_row: i32::from(s.goal[1]),
        steps: u32::from(s.steps),
        outcome: if s.reached_goal.get() {
            GridOutcome::ReachedGoal
        } else if s.out_of_bounds.get() {
            GridOutcome::OutOfBounds
        } else {
            GridOutcome::Running
        },
    }
}

fn states_match(lanes: &EnvPool<GridLanes>, grid: &EnvPool<Grid>, n: u32, ctx: &str) {
    for i in 0..n {
        assert_eq!(
            to_grid(lanes.state(i).unwrap()),
            grid.state(i).unwrap(),
            "{ctx}: state {i}"
        );
    }
}

fn store_match(lanes: &EnvPool<GridLanes>, rows: &EnvPool<GridLanesRows>, n: u32, ctx: &str) {
    assert_eq!(records!(lanes, n), records!(rows, n), "{ctx}: records");
    for i in 0..n {
        assert_eq!(
            lanes.state_hash(i).unwrap(),
            rows.state_hash(i).unwrap(),
            "{ctx}: hash {i}"
        );
    }
}

fn differential(cfg: &GridConfig, case: Case) {
    let ctx = format!("{case:?} size {}", cfg.size);
    let mut lanes = pool!(GridLanes, cfg, case);
    let mut grid = pool!(Grid, cfg, case);
    let mut rows = pool!(GridLanesRows, cfg, case);
    let same = case.auto_reset == AutoReset::SameStep;
    let a = outputs(lanes.reset_all(Reseed::Keep).unwrap(), same);
    assert_eq!(
        a,
        outputs(grid.reset_all(Reseed::Keep).unwrap(), same),
        "{ctx}: reset"
    );
    assert_eq!(
        a,
        outputs(rows.reset_all(Reseed::Keep).unwrap(), same),
        "{ctx}: reset rows"
    );
    states_match(&lanes, &grid, case.n, &ctx);
    let ids: Vec<u32> = (0..case.n).collect();
    for t in 0..case.steps {
        let acts: Vec<i32> = (0..case.n).map(|i| action(t, i)).collect();
        let input = ActionInput {
            discrete: &acts,
            continuous: &[],
        };
        let x = outputs(lanes.step(input).unwrap(), same);
        assert_eq!(
            x,
            outputs(grid.step(input).unwrap(), same),
            "{ctx}: step {t}"
        );
        assert_eq!(
            x,
            outputs(rows.step(input).unwrap(), same),
            "{ctx}: rows step {t}"
        );
        states_match(&lanes, &grid, case.n, &ctx);
        if t % 23 == 22 {
            store_match(&lanes, &rows, case.n, &ctx);
            assert_eq!(
                lanes.encode(&ids).unwrap(),
                rows.encode(&ids).unwrap(),
                "{ctx}: persisted"
            );
            let dst: Vec<u32> = (case.n / 2..case.n).collect();
            lanes.fork(0, &dst).unwrap();
            grid.fork(0, &dst).unwrap();
            rows.fork(0, &dst).unwrap();
            if case.n > 2 {
                lanes.fork_branched(1, &[2], None).unwrap();
                grid.fork_branched(1, &[2], None).unwrap();
                rows.fork_branched(1, &[2], None).unwrap();
            }
            // A partial reset mixes pending and active instances in a group
            // (a mid-group reset).
            let partial: Vec<u32> = (0..case.n).step_by(5).collect();
            lanes.reset(&partial, Reseed::Keep).unwrap();
            grid.reset(&partial, Reseed::Keep).unwrap();
            rows.reset(&partial, Reseed::Keep).unwrap();
        }
        if t % 37 == 36 {
            // Records cross the stores through their bytes, both ways, and
            // Grid's records import into GridLanes through the mapping.
            let from_rows = rows.encode(&ids).unwrap();
            let from_lanes = lanes.encode(&ids).unwrap();
            let into_lanes = lanes.decode(&from_rows, DecodeLimits::default()).unwrap();
            let into_rows = rows.decode(&from_lanes, DecodeLimits::default()).unwrap();
            lanes.import_all(&into_lanes).unwrap();
            rows.import_all(&into_rows).unwrap();
            let r = grid.snapshot(0).unwrap();
            let rec = DecodedRecord::from_parts(*r.header(), of_grid(*r.state())).unwrap();
            lanes.import(&[0], &[rec]).unwrap();
            let rec =
                DecodedRecord::from_parts(*r.header(), GridRowsState::of(of_grid(*r.state())))
                    .unwrap();
            rows.import(&[0], &[rec]).unwrap();
        }
    }
    store_match(&lanes, &rows, case.n, &ctx);
    states_match(&lanes, &grid, case.n, &ctx);
}

fn matrix(ns: &[u32], chunks: &[u32], workers: &[usize], steps: u32) {
    for cfg in configs() {
        for &n in ns {
            for &chunk in chunks {
                for &workers in workers {
                    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
                        for limit in [None, Some(9)] {
                            for level in Resolved::available().map(Resolved::level) {
                                differential(
                                    &cfg,
                                    Case {
                                        n,
                                        chunk,
                                        workers,
                                        auto_reset,
                                        limit,
                                        steps,
                                        level: Some(level),
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_grid_lanes_pool_small_matches_grid() {
    matrix(&[1, 7, 8, 9, 20, 64, 65], &[8, 16, 64], &[0, 1, 2], 160);
}

#[test]
fn test_grid_lanes_pool_large_matches_grid() {
    matrix(&[1000], &[8, 1024], &[0, 2], 60);
}

#[test]
fn test_grid_lanes_pool_pipelined_matches_rows() {
    const N: u32 = 64;
    let cfg = &configs()[0];
    macro_rules! run {
        ($env:ty) => {{
            let case = Case {
                n: N,
                chunk: 8,
                workers: 0,
                auto_reset: AutoReset::SameStep,
                limit: Some(30),
                steps: 0,
                level: None,
            };
            let mut p = EnvPool::<$env>::new(cfg.clone(), options(case))
                .unwrap()
                .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
                .unwrap();
            let halves = p.halves();
            let [mut ta, mut tb] = p.reset_all(Reseed::Keep).unwrap();
            for t in 0..200 {
                for token in [ta, tb] {
                    let h = usize::try_from(token.half()).unwrap();
                    let a: Vec<i32> = halves[h].clone().map(|i| action(t, i)).collect();
                    assert!(
                        p.send(
                            token,
                            ActionInput {
                                discrete: &a,
                                continuous: &[],
                            },
                        )
                        .is_ok()
                    );
                }
                ta = p.recv().unwrap();
                tb = p.recv().unwrap();
            }
            drop((ta, tb));
            let mut out = vec![
                EnvPool::<$env>::new(cfg.clone(), options(case))
                    .unwrap()
                    .snapshot(0)
                    .unwrap();
                N as usize
            ];
            p.snapshot_all(&mut out).unwrap();
            out.iter().map(|r| r.to_bytes()).collect::<Vec<_>>()
        }};
    }
    assert_eq!(run!(GridLanes), run!(GridLanesRows));
}

#[test]
fn test_grid_lanes_pool_replay_files_are_identical_and_cross_replay() {
    let cfg = &configs()[1];
    macro_rules! record {
        ($env:ty) => {{
            let case = Case {
                n: 16,
                chunk: 8,
                workers: 2,
                auto_reset: AutoReset::SameStep,
                limit: Some(25),
                steps: 0,
                level: None,
            };
            let mut p = pool!($env, cfg, case);
            let options = ReplayWriterOptions {
                keyframe_every: 16,
                checkpoint_every: 4,
                outputs_hash: false,
            };
            let mut rec = Recorder::new(&mut p, Vec::new(), options).unwrap();
            rec.reset_all(Reseed::Base(Seed::new(5))).unwrap();
            for t in 0..60 {
                let a: Vec<i32> = (0..16).map(|i| action(t, i)).collect();
                rec.step(ActionInput {
                    discrete: &a,
                    continuous: &[],
                })
                .unwrap();
                if t == 30 {
                    rec.fork_branched(0, &[1, 2], None).unwrap();
                    rec.reset(&[3], Reseed::List(&[Seed::new(8)])).unwrap();
                }
            }
            let bytes = rec.finish().unwrap();
            let hashes: Vec<u64> = (0..16).map(|i| p.state_hash(i).unwrap().get()).collect();
            (bytes, hashes, case)
        }};
    }
    let (lanes_file, lanes_hashes, case) = record!(GridLanes);
    let (rows_file, rows_hashes, _) = record!(GridLanesRows);
    assert_eq!(lanes_file, rows_file);
    assert_eq!(lanes_hashes, rows_hashes);
    let mut lanes = pool!(GridLanes, cfg, case);
    Replay::open(&rows_file, DecodeLimits::default())
        .unwrap()
        .run(&mut lanes)
        .unwrap();
    let mut rows = pool!(GridLanesRows, cfg, case);
    Replay::open(&lanes_file, DecodeLimits::default())
        .unwrap()
        .run(&mut rows)
        .unwrap();
    let a: Vec<u64> = (0..16)
        .map(|i| lanes.state_hash(i).unwrap().get())
        .collect();
    let b: Vec<u64> = (0..16).map(|i| rows.state_hash(i).unwrap().get()).collect();
    assert_eq!(a, rows_hashes);
    assert_eq!(b, rows_hashes);
}

#[test]
fn test_grid_lanes_pool_async_is_refused() {
    let case = Case {
        n: 16,
        chunk: 8,
        workers: 0,
        auto_reset: AutoReset::SameStep,
        limit: None,
        steps: 0,
        level: None,
    };
    let p = pool!(GridLanes, &configs()[0], case);
    let err = p
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(8).unwrap(),
        ))
        .err()
        .unwrap();
    assert!(matches!(err.error, EngineError::LaneEnvUnsupported));
}

#[test]
fn test_grid_lanes_pool_chunk_size_must_be_a_multiple_of_8() {
    let cfg = &configs()[0];
    for chunk in [1, 4, 12, 63] {
        let options = PoolOptions {
            chunk_size: NonZeroU32::new(chunk).unwrap(),
            ..PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(1))
        };
        assert!(
            matches!(
                EnvPool::<GridLanes>::new(cfg.clone(), options),
                Err(EngineError::Options {
                    option: "chunk_size",
                    ..
                })
            ),
            "chunk {chunk}"
        );
    }
    for chunk in [8, 64, 1024] {
        let options = PoolOptions {
            chunk_size: NonZeroU32::new(chunk).unwrap(),
            ..PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(1))
        };
        EnvPool::<GridLanes>::new(cfg.clone(), options).unwrap();
    }
}

#[test]
fn test_grid_lanes_pool_heuristic_slots_are_refused() {
    let options = PoolOptions::new(NonZeroU32::new(16).unwrap(), Seed::new(3));
    let p = EnvPool::<GridLanes>::new(configs()[0].clone(), options).unwrap();
    assert!(matches!(
        p.with_heuristic_slots(&[0]),
        Err(EngineError::PolicyMap { .. })
    ));
}

#[test]
fn test_grid_lanes_pool_with_transforms_matches_grid() {
    // Observation transforms take the per-instance observe (ADR-0090 4.4).
    use tessera_rl_core::space::Space;
    let cfg = &configs()[0];
    let leaf = match GridLanes::observation_space(cfg) {
        Ok(Space::Dict(entries)) => entries.first().map(|e| e.0.clone()).unwrap_or_default(),
        _ => String::new(),
    };
    let transforms = || Transforms::new().frame_stack(leaf.clone(), 3);
    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
        let case = Case {
            n: 32,
            chunk: 8,
            workers: 0,
            auto_reset,
            limit: Some(30),
            steps: 0,
            level: None,
        };
        let mut lanes = pool!(GridLanes, cfg, case)
            .with_transforms(transforms())
            .unwrap();
        let mut grid = pool!(Grid, cfg, case)
            .with_transforms(transforms())
            .unwrap();
        let same = auto_reset == AutoReset::SameStep;
        assert_eq!(
            outputs(lanes.reset_all(Reseed::Keep).unwrap(), same),
            outputs(grid.reset_all(Reseed::Keep).unwrap(), same)
        );
        for t in 0..120 {
            let acts: Vec<i32> = (0..32).map(|i| action(t, i)).collect();
            let input = ActionInput {
                discrete: &acts,
                continuous: &[],
            };
            assert_eq!(
                outputs(lanes.step(input).unwrap(), same),
                outputs(grid.step(input).unwrap(), same),
                "{auto_reset:?} step {t}"
            );
        }
    }
}
