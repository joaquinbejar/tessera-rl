// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tag through the engine: spaces, initial agents, mask, egocentric
//! observations, captures and rewards, episode ends, determinism and the
//! scripted baseline.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "test helpers: a failure is the assertion; small action indices"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_classic::tag::grid::moved;
use tessera_rl_classic::{Tag, TagConfig, TagState};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{AgentSlot, Bool8, Environment, Heuristic, StateHash};
use tessera_rl_engine::{
    ActionInput, EngineError, EnvPool, Masked, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

const A: usize = 8;

fn pool(n: u32, seed: u64, cfg: TagConfig) -> EnvPool<Tag, Masked> {
    let options = PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed));
    EnvPool::with_masks(cfg, options).unwrap()
}

fn act(values: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: values,
        continuous: &[],
    }
}

fn heuristic(cfg: &TagConfig, state: &TagState, agent: usize) -> i32 {
    let mut discrete = [0];
    let mut writer = ActionWriter::new(&mut discrete, &mut []);
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
    let slot = AgentSlot::new(u16::try_from(agent).unwrap(), A).unwrap();
    Tag::act(cfg, state, slot, &mut rng, &mut writer);
    discrete[0]
}

/// Heuristic actions for every slot of every instance; evaders stay when
/// `evaders_stay`.
fn scripted(p: &EnvPool<Tag, Masked>, cfg: &TagConfig, n: u32, evaders_stay: bool) -> Vec<i32> {
    let mut out = vec![0; usize::try_from(n).unwrap() * A];
    for i in 0..n {
        let state = p.state(i).unwrap();
        for a in 0..cfg.agents() {
            if !evaders_stay || cfg.is_pursuer(a) {
                out[usize::try_from(i).unwrap() * A + a] = heuristic(cfg, state, a);
            }
        }
    }
    out
}

/// `actions` with every move the current mask forbids replaced by stay, so
/// the tests also pass with the strict `debug-checks` feature.
fn legal(p: &EnvPool<Tag, Masked>, mut actions: Vec<i32>) -> Vec<i32> {
    let masks = p.outputs().masks();
    for (row, a) in actions.iter_mut().enumerate() {
        let index = usize::try_from(*a).unwrap();
        if !masks[row * 5 + index].get() {
            *a = 0;
        }
    }
    actions
}

/// An independent observation reference: `[3, side, side]` counts.
fn expected_window(cfg: &TagConfig, s: &TagState, me: usize) -> Vec<u8> {
    let r = i32::try_from(cfg.radius).unwrap();
    let side = usize::try_from(2 * r + 1).unwrap();
    let mut w = vec![0u8; 3 * side * side];
    let [cx, cy] = s.pos[me].map(i32::from);
    for row in 0..side {
        for col in 0..side {
            let x = cx - r + i32::try_from(col).unwrap();
            let y = cy - r + i32::try_from(row).unwrap();
            let outside = x < 0
                || y < 0
                || x >= i32::try_from(cfg.width).unwrap()
                || y >= i32::try_from(cfg.height).unwrap();
            w[row * side + col] = u8::from(outside);
            for j in 0..cfg.agents() {
                if s.caught[j].get() || s.pos[j].map(i32::from) != [x, y] {
                    continue;
                }
                let channel = if cfg.is_pursuer(j) { 1 } else { 2 };
                w[channel * side * side + row * side + col] += 1;
            }
        }
    }
    w
}

#[test]
fn test_tag_spaces_are_an_egocentric_u8_window_and_five_moves() {
    let cfg = TagConfig {
        radius: 3,
        ..TagConfig::default()
    };
    let expected = Space::Dict(vec![(
        "window".to_owned(),
        Space::Box {
            shape: vec![3, 7, 7],
            dtype: DType::U8,
            bounds: BoxBounds::UInt {
                low: vec![0],
                high: vec![255],
            },
        },
    )]);
    assert_eq!(Tag::observation_space(&cfg).unwrap(), expected);
    assert_eq!(Tag::action_space(&cfg).unwrap(), Space::Discrete { n: 5 });
}

#[test]
fn test_tag_more_than_eight_agents_is_config_error() {
    let cfg = TagConfig {
        pursuers: 5,
        evaders: 4,
        ..TagConfig::default()
    };
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    assert!(matches!(
        EnvPool::<Tag, Masked>::with_masks(cfg, options).map(|_| ()),
        Err(EngineError::Config(e)) if e.field == "initial_agents"
    ));
}

#[test]
fn test_tag_reset_places_used_slots_distinctly_and_leaves_the_rest_empty() {
    let cfg = TagConfig {
        width: 3,
        height: 3,
        pursuers: 2,
        evaders: 3,
        ..TagConfig::default()
    };
    let mut p = pool(16, 2, cfg.clone());
    let out = p.reset_all(Reseed::Keep).unwrap();
    for i in 0..16 {
        let alive = &out.alive()[i * A..(i + 1) * A];
        assert!(alive.iter().take(5).all(|f| f.get()));
        assert!(alive.iter().skip(5).all(|f| !f.get()));
    }
    for i in 0..16 {
        let s = p.state(i).unwrap();
        for a in 0..5 {
            assert!(s.pos[a].iter().all(|v| (0..3).contains(v)));
            assert!((a + 1..5).all(|b| s.pos[b] != s.pos[a]), "{s:?}");
        }
        assert!(s.pos[5..].iter().all(|c| *c == [0, 0]));
    }
}

#[test]
fn test_tag_mask_marks_moves_off_the_grid_illegal() {
    let cfg = TagConfig {
        width: 3,
        height: 3,
        pursuers: 4,
        evaders: 4,
        ..TagConfig::default()
    };
    let mut p = pool(8, 3, cfg.clone());
    let out = p.reset_all(Reseed::Keep).unwrap();
    let masks = out.masks().to_vec();
    for i in 0..8 {
        let s = p.state(i).unwrap();
        for a in 0..8 {
            let row = &masks[(usize::try_from(i).unwrap() * A + a) * 5..][..5];
            for (action, legal) in row.iter().enumerate() {
                let action = i32::try_from(action).unwrap();
                assert_eq!(legal.get(), moved(s.pos[a], action, 3, 3).is_some());
            }
            assert!(row[0].get(), "stay is always legal");
        }
    }
}

#[test]
fn test_tag_observations_match_an_independent_reference() {
    let cfg = TagConfig {
        width: 5,
        height: 4,
        pursuers: 3,
        evaders: 3,
        radius: 2,
        ..TagConfig::default()
    };
    let n = 8;
    let mut p = pool(n, 4, cfg.clone());
    p.reset_all(Reseed::Keep).unwrap();
    let leaf = 3 * 5 * 5;
    for t in 0..30 {
        let raw = (0..n as usize * A).map(|k| ((k + t) % 5) as i32).collect();
        let actions = legal(&p, raw);
        let out = p.step(act(&actions)).unwrap();
        let obs = out.observation(0).unwrap().to_vec();
        let alive = out.alive().to_vec();
        for i in 0..n {
            let s = p.state(i).unwrap();
            for a in 0..cfg.agents() {
                let row = usize::try_from(i).unwrap() * A + a;
                if !alive[row].get() {
                    continue;
                }
                let got = &obs[row * leaf..(row + 1) * leaf];
                assert_eq!(got, expected_window(&cfg, s, a), "t {t} env {i} agent {a}");
            }
        }
    }
}

#[test]
fn test_tag_capture_rewards_pursuers_and_kills_the_evader() {
    let cfg = TagConfig {
        width: 3,
        height: 3,
        pursuers: 1,
        evaders: 2,
        slip: 0,
        ..TagConfig::default()
    };
    let n = 32;
    let mut p = pool(n, 5, cfg.clone());
    p.reset_all(Reseed::Keep).unwrap();
    let mut captures = 0;
    let mut finished = vec![false; usize::try_from(n).unwrap()];
    for _ in 0..40 {
        let actions = scripted(&p, &cfg, n, true);
        let before: Vec<TagState> = (0..n).map(|i| *p.state(i).unwrap()).collect();
        let out = p.step(act(&actions)).unwrap();
        for i in 0..usize::try_from(n).unwrap() {
            if finished[i] {
                continue;
            }
            let rows = i * A..(i + 1) * A;
            let rewards = &out.rewards()[rows.clone()];
            let terminated = &out.agent_terminated()[rows.clone()];
            for e in 1..3 {
                if terminated[e].get() {
                    captures += 1;
                    assert!(!before[i].caught[e].get());
                    assert_eq!(rewards[e], -1.0);
                    assert!(!out.alive()[rows.start + e].get());
                } else {
                    assert_eq!(rewards[e], 0.0);
                }
            }
            let caught_now = (1..3).filter(|&e| terminated[e].get()).count();
            #[expect(clippy::cast_precision_loss, reason = "small counts")]
            let expected = caught_now as f32;
            assert_eq!(rewards[0], expected);
            if out.episode_terminated()[i].get() {
                finished[i] = true;
            }
        }
    }
    // Stationary evaders on a 3x3 grid are all caught well within 40 steps.
    assert!(finished.iter().all(|f| *f));
    assert_eq!(captures, 2 * n);
}

#[test]
fn test_tag_truncates_at_max_steps_while_evaders_remain() {
    let cfg = TagConfig {
        width: 16,
        height: 16,
        slip: 0,
        max_steps: 5,
        ..TagConfig::default()
    };
    let mut p = pool(1, 6, cfg);
    p.reset_all(Reseed::Keep).unwrap();
    for t in 0..5 {
        // Everyone stays: no capture.
        let out = p.step(act(&[0; A])).unwrap();
        assert_eq!(out.episode_truncated()[0].get(), t == 4);
        assert_eq!(out.episode_terminated(), [Bool8::FALSE]);
    }
}

fn hashes(p: &EnvPool<Tag, Masked>, n: u32) -> Vec<u64> {
    (0..n)
        .map(|i| StateHash::of(p.state(i).unwrap()).get())
        .collect()
}

fn run(seed: u64, workers: Option<usize>) -> Vec<u64> {
    let cfg = TagConfig::default();
    let n = 24;
    let mut p = pool(n, seed, cfg.clone());
    if let Some(w) = workers {
        p = p
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap();
    }
    p.reset_all(Reseed::Keep).unwrap();
    for _ in 0..120 {
        let actions = scripted(&p, &cfg, n, false);
        p.step(act(&actions)).unwrap();
    }
    hashes(&p, n)
}

#[test]
fn test_tag_slip_draws_are_deterministic_per_seed_and_mode() {
    let inline = run(7, None);
    assert_eq!(inline, run(7, None));
    assert_eq!(inline, run(7, Some(3)));
    assert_ne!(inline, run(8, None));
}

#[test]
fn test_tag_heuristic_pursuers_catch_more_than_random_play() {
    let cfg = TagConfig {
        pursuers: 3,
        evaders: 3,
        slip: 0,
        max_steps: 100,
        ..TagConfig::default()
    };
    let n = 32;
    let episodes = |scripted_play: bool| {
        let mut p = pool(n, 10, cfg.clone());
        p.reset_all(Reseed::Keep).unwrap();
        let mut caught = 0;
        for t in 0..100 {
            let actions = if scripted_play {
                scripted(&p, &cfg, n, false)
            } else {
                let raw = (0..n as usize * A)
                    .map(|k| ((k * 7 + t * 3) % 5) as i32)
                    .collect();
                legal(&p, raw)
            };
            let out = p.step(act(&actions)).unwrap();
            caught += out.agent_terminated().iter().filter(|f| f.get()).count();
        }
        caught
    };
    assert!(episodes(true) > episodes(false));
}
