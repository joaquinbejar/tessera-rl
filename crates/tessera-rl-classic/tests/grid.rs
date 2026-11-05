// SPDX-License-Identifier: MIT OR Apache-2.0

//! The reference grid world through the engine, against a reference written
//! only for these tests, plus import validation, the G6 line count and the
//! baseline.

#![expect(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;
use std::collections::BTreeSet;

use tessera_rl_classic::{Grid, GridConfig, GridState};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{AgentSlot, Bool8, EnvConfig, Environment, Heuristic, Validate};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, NoMasks, PoolOptions, Reseed,
};
use tessera_rl_num::Fx32;
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

fn cfg() -> GridConfig {
    GridConfig::default()
}

fn pool_with(c: GridConfig, seed: u64, auto_reset: AutoReset) -> EnvPool<Grid, NoMasks> {
    let options = PoolOptions {
        auto_reset,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(seed))
    };
    let mut p = EnvPool::new(c, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn pool(seed: u64) -> EnvPool<Grid, NoMasks> {
    pool_with(cfg(), seed, AutoReset::Disabled)
}

fn step(p: &mut EnvPool<Grid, NoMasks>, a: i32) {
    p.step(ActionInput {
        discrete: &[a],
        continuous: &[],
    })
    .unwrap();
}

fn state(col: i32, row: i32, goal: [i16; 2], steps: u16) -> GridState {
    let fx = |v| Fx32::checked_from_int(v).unwrap();
    let on_goal = (col, row) == (i32::from(goal[0]), i32::from(goal[1]));
    let inside = (0..16).contains(&col) && (0..16).contains(&row);
    GridState {
        pos: [fx(col), fx(row)],
        goal,
        steps,
        reached_goal: Bool8::from(on_goal),
        out_of_bounds: Bool8::from(!inside),
    }
}

fn rejected(s: &GridState, code: &str) {
    assert_eq!(Grid::validate(&cfg(), s).unwrap_err().code, code);
}

fn invalid(r: &Result<(), EngineError>) -> bool {
    matches!(
        r,
        Err(EngineError::Invariant { index: 0, .. } | EngineError::InvalidRecord { index: 0, .. })
    )
}

/// SPEC 10's target for everything the environment needs to run, config and
/// observation included, is under 150 lines. Goal G6 is missed: this file holds
/// 156, formatted and with the docs public items require. Only validation and
/// the baseline, which an author may omit, live in `checks.rs`. The bound stops
/// it growing; lowering it to 149 is the way to record meeting G6.
#[test]
fn test_grid_main_file_stays_at_its_measured_line_count() {
    let lines = include_str!("../src/grid/mod.rs").lines().count();
    assert!(lines <= 156, "{lines} lines");
}

#[test]
fn test_grid_state_is_the_16_bytes_of_the_spec() {
    assert_eq!(size_of::<GridState>(), 16);
}

#[test]
fn test_grid_spaces_follow_the_size() {
    let c = GridConfig { size: 5, ..cfg() };
    let expected = Space::Dict(vec![(
        "pixels".to_owned(),
        Space::Box {
            shape: vec![3, 5, 5],
            dtype: DType::U8,
            bounds: BoxBounds::UInt {
                low: vec![0],
                high: vec![255],
            },
        },
    )]);
    assert_eq!(Grid::observation_space(&c).unwrap(), expected);
    assert_eq!(Grid::action_space(&c).unwrap(), Space::Discrete { n: 4 });
}

#[test]
fn test_grid_config_rejects_values_out_of_range() {
    let bad = [(3, 200), (65, 200), (16, 0), (16, 10_001)];
    for (size, max_steps) in bad {
        assert!(
            GridConfig { size, max_steps }.validate().is_err(),
            "{size} {max_steps}"
        );
    }
}

#[test]
fn test_grid_reset_is_seeded_covers_the_grid_and_avoids_the_start() {
    let c = GridConfig { size: 4, ..cfg() };
    let mut seen = BTreeSet::new();
    for seed in 0..500 {
        let goal = pool_with(c.clone(), seed, AutoReset::Disabled)
            .state(0)
            .unwrap()
            .goal;
        assert_eq!(
            goal,
            pool_with(c.clone(), seed, AutoReset::Disabled)
                .state(0)
                .unwrap()
                .goal,
            "seed {seed}"
        );
        seen.insert(goal);
    }
    assert_eq!(seen.len(), 15, "every cell but the start");
    assert!(!seen.contains(&[0, 0]));
}

/// The reference: plain integers and the rules written out once more.
#[test]
fn test_grid_random_walks_match_the_reference() {
    let mut rng = 0x1234_5678_u64;
    let (mut goals, mut falls, mut limits) = (0, 0, 0);
    for seed in 0..400 {
        let c = GridConfig {
            size: 4 + (seed % 5) as u16,
            max_steps: 12,
        };
        let size = i32::from(c.size);
        let mut p = pool_with(c.clone(), seed, AutoReset::Disabled);
        let start = *p.state(0).unwrap();
        let goal = (i32::from(start.goal[0]), i32::from(start.goal[1]));
        let (mut col, mut row, mut steps) = (0, 0, 0);
        loop {
            rng = rng.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
            let a = ((rng >> 40) % 4) as i32;
            match a {
                0 => row -= 1,
                1 => row += 1,
                2 => col += 1,
                _ => col -= 1,
            }
            steps += 1;
            step(&mut p, a);
            let out = p.outputs();
            let off = col < 0 || row < 0 || col >= size || row >= size;
            let won = (col, row) == goal;
            assert_eq!(p.state(0).unwrap().cell(), (col, row), "seed {seed}");
            let reward = if off {
                -1.0
            } else if won {
                1.0
            } else {
                0.0
            };
            assert_eq!(out.rewards()[0], reward, "seed {seed}");
            assert_eq!(out.episode_terminated()[0].get(), off || won, "seed {seed}");
            let limit = !off && !won && steps >= 12;
            assert_eq!(out.episode_truncated()[0].get(), limit, "seed {seed}");
            let obs = out.observation(0).unwrap();
            let n = usize::from(c.size);
            for r in 0..n {
                for k in 0..n {
                    let here = (k as i32, r as i32);
                    let lit = |on: bool| if on { 255 } else { 0 };
                    assert_eq!(obs[r * n + k], lit(here == (col, row)));
                    assert_eq!(obs[n * n + r * n + k], lit(here == goal));
                    let edge = r == 0 || k == 0 || r == n - 1 || k == n - 1;
                    assert_eq!(obs[2 * n * n + r * n + k], lit(edge));
                }
            }
            goals += i32::from(won);
            falls += i32::from(off);
            limits += i32::from(limit);
            if off || won || limit {
                break;
            }
        }
    }
    assert!(
        goals > 0 && falls > 0 && limits > 0,
        "{goals} {falls} {limits}"
    );
}

#[test]
fn test_grid_edge_cell_is_on_the_grid_and_one_more_step_falls_off() {
    let mut p = pool(3);
    step(&mut p, 2);
    assert!(!p.outputs().episode_terminated()[0].get() || p.state(0).unwrap().goal == [1, 0]);
    let mut q = pool(3);
    step(&mut q, 0);
    assert!(q.state(0).unwrap().out_of_bounds.get());
    assert!(q.outputs().episode_terminated()[0].get());
    assert_eq!(q.outputs().rewards()[0], -1.0);
}

#[test]
fn test_grid_auto_reset_draws_a_new_goal_from_the_stream() {
    let mut p = pool_with(cfg(), 7, AutoReset::SameStep);
    let first = p.state(0).unwrap().goal;
    let mut goals = BTreeSet::from([first]);
    for _ in 0..40 {
        step(&mut p, 0);
        assert!(p.outputs().episode_terminated()[0].get());
        assert_eq!(p.state(0).unwrap().cell(), (0, 0));
        goals.insert(p.state(0).unwrap().goal);
    }
    assert!(goals.len() > 1);
}

#[test]
fn test_grid_validate_accepts_real_states_and_names_each_rejection() {
    let good = state(2, 1, [5, 5], 3);
    assert_eq!(Grid::validate(&cfg(), &good), Ok(()));
    rejected(
        &state(2, 1, [16, 5], 3),
        "a goal off the grid or on the start",
    );
    rejected(
        &state(2, 1, [5, -1], 3),
        "a goal off the grid or on the start",
    );
    rejected(
        &state(2, 1, [0, 0], 3),
        "a goal off the grid or on the start",
    );
    let mut half = good;
    half.pos[0] = Fx32::from_raw(half.pos[0].to_raw() + 1);
    rejected(&half, "a position that is not a whole cell");
    rejected(
        &state(17, 1, [5, 5], 18),
        "a position beyond one cell off the grid",
    );
    rejected(
        &state(-1, -1, [5, 5], 4),
        "a position beyond one cell off the grid",
    );
    let mut flag = good;
    flag.out_of_bounds = Bool8::TRUE;
    rejected(
        &flag,
        "an out-of-bounds flag that disagrees with the position",
    );
    let mut reached = good;
    reached.reached_goal = Bool8::TRUE;
    rejected(&reached, "a reached flag that disagrees with the position");
    rejected(
        &state(2, 1, [5, 5], 1),
        "a step count that the walk does not allow",
    );
    rejected(
        &state(2, 1, [5, 5], 4),
        "a step count that the walk does not allow",
    );
    rejected(
        &state(2, 1, [5, 5], 201),
        "a step count that the walk does not allow",
    );
}

#[test]
fn test_grid_import_rejects_an_ended_episode_as_running_and_a_dead_agent() {
    let mut p = pool(1);
    step(&mut p, 1);
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let goal = good.state().goal;
    let fell = state(-1, 0, goal, 1);
    assert_eq!(Grid::validate(&cfg(), &fell), Ok(()));
    let rec = DecodedRecord::from_parts(*good.header(), fell).unwrap();
    let before = p.state_hash(0).unwrap();
    assert!(invalid(&p.import(&[0], &[rec])));
    assert_eq!(p.state_hash(0).unwrap(), before);
    let mut header = *good.header();
    header.agents[0].alive = Bool8::FALSE;
    let dead = DecodedRecord::from_parts(header, *good.state()).unwrap();
    assert!(invalid(&p.import(&[0], &[dead])));
}

#[test]
fn test_grid_terminal_record_imports_as_pending_reset() {
    let mut p = pool(1);
    step(&mut p, 0);
    assert!(p.header(0).unwrap().engine.pending_reset.get());
    let rec = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut q = pool(1);
    assert!(q.import(&[0], &[rec]).is_ok());
    assert_eq!(q.state(0), p.state(0));
}

#[test]
fn test_grid_never_reset_record_round_trips_and_refuses_anything_else() {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(11));
    let p = EnvPool::<Grid, NoMasks>::new(cfg(), options).unwrap();
    let rec = DecodedRecord::from(p.snapshot(0).unwrap());
    assert_eq!(*rec.state(), GridState::default());
    let header = *rec.header();
    let mut q = EnvPool::<Grid, NoMasks>::new(cfg(), options).unwrap();
    q.import(&[0], &[rec]).unwrap();
    let bad = DecodedRecord::from_parts(header, state(0, 0, [1, 1], 0)).unwrap();
    assert!(invalid(&q.import(&[0], &[bad])));
}

fn baseline(s: &GridState) -> i32 {
    let mut discrete = [0];
    let mut writer = ActionWriter::new(&mut discrete, &mut []);
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
    let slot = AgentSlot::new(0, 1).unwrap();
    Grid::act(&cfg(), s, slot, &mut rng, &mut writer);
    discrete[0]
}

#[test]
fn test_grid_baseline_reaches_every_goal_by_the_shortest_walk() {
    for seed in 0..300 {
        let mut p = pool(seed);
        let goal = p.state(0).unwrap().goal;
        let shortest = i32::from(goal[0]) + i32::from(goal[1]);
        let mut steps = 0;
        while !p.outputs().episode_terminated()[0].get() {
            let a = baseline(p.state(0).unwrap());
            step(&mut p, a);
            steps += 1;
            assert!(steps <= shortest, "seed {seed}");
        }
        assert_eq!(steps, shortest, "seed {seed}");
        assert_eq!(p.outputs().rewards()[0], 1.0);
    }
}
