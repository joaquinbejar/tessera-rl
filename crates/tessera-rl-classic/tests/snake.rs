// SPDX-License-Identifier: MIT OR Apache-2.0

//! Snake through the engine, against a reference written only for these tests:
//! each collision rule by name, food, ends, the mask, both observations, import
//! validation and the baseline.

#![expect(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::large_types_passed_by_value,
    reason = "test helpers on a small board: a failure is the assertion"
)]

use core::num::NonZeroU32;
use std::collections::{BTreeSet, VecDeque};

use tessera_rl_classic::snake::{
    NO_FOOD, OBS_PIXELS, OBS_STATE, PIXELS, RING, SHADE_FOOD, SHADE_FREE, SHADE_OTHER_BODY,
    SHADE_OTHER_HEAD, SHADE_OUTSIDE, SHADE_OWN_BODY, SHADE_OWN_HEAD,
};
use tessera_rl_classic::{Snake, SnakeConfig, SnakeState};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{AgentSlot, Bool8, EnvConfig, Environment, Heuristic, Validate};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, Masked, NoMasks, PoolOptions,
    Reseed,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

type Pool = EnvPool<Snake, Masked>;

const N: u8 = 0;
const E: u8 = 1;
const S: u8 = 2;
const W: u8 = 3;

fn config(size: u32, agents: u32, obs: u32) -> SnakeConfig {
    SnakeConfig { size, agents, obs }
}

fn pool_with(c: &SnakeConfig, seed: u64, auto_reset: AutoReset) -> Pool {
    let options = PoolOptions {
        auto_reset,
        max_episode_steps: NonZeroU32::new(100_000),
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(seed))
    };
    let mut p = EnvPool::with_masks(c.clone(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn pool(c: &SnakeConfig, seed: u64) -> Pool {
    pool_with(c, seed, AutoReset::Disabled)
}

fn state(p: &Pool) -> SnakeState {
    *p.state(0).unwrap()
}

/// Steps with one direction per slot. A reversal, which the mask refuses and
/// strict checks reject, is sent as the current direction, as the kernel
/// reads it.
fn step(p: &mut Pool, acts: [u8; 2]) {
    let s = state(p);
    let row = [0, 1].map(|k| {
        let a = if acts[k] == (s.facing[k] + 2) % 4 {
            s.facing[k]
        } else {
            acts[k]
        };
        i32::from(a)
    });
    p.step(ActionInput {
        discrete: &row,
        continuous: &[],
    })
    .unwrap();
}

fn rewards(p: &Pool) -> [f32; 2] {
    let r = p.outputs().unwrap().rewards();
    [r[0], r[1]]
}

fn terminated(p: &Pool) -> bool {
    p.outputs().unwrap().episode_terminated()[0].get()
}

fn import(p: &mut Pool, s: SnakeState) -> Result<(), EngineError> {
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let rec = DecodedRecord::from_parts(*good.header(), s).unwrap();
    p.import(&[0], &[rec])
}

fn invalid(r: &Result<(), EngineError>) -> bool {
    matches!(
        r,
        Err(EngineError::Invariant { index: 0, .. } | EngineError::InvalidRecord { index: 0, .. })
    )
}

fn at(size: u32, (row, col): (u32, u32)) -> u16 {
    (row * size + col) as u16
}

/// A state from bodies given head first as `(row, col)`; the facing is read
/// from the neck. Ring cursors start at an offset so wrapping is exercised.
fn board(size: u32, bodies: &[&[(u32, u32)]], food: Option<(u32, u32)>) -> SnakeState {
    let mut s = SnakeState::default();
    for (k, body) in bodies.iter().enumerate() {
        let len = body.len();
        let head = (RING - 2 + len) % RING;
        for (i, &cell) in body.iter().enumerate() {
            s.body[k][(head + RING - i) % RING] = at(size, cell);
        }
        let (h, n) = (body[0], body[1]);
        s.facing[k] = match (h.0 as i32 - n.0 as i32, h.1 as i32 - n.1 as i32) {
            (-1, 0) => N,
            (0, 1) => E,
            (1, 0) => S,
            _ => W,
        };
        s.head[k] = head as u16;
        s.len[k] = len as u16;
        s.alive[k] = Bool8::TRUE;
    }
    s.food = food.map_or(NO_FOOD, |f| at(size, f));
    s
}

fn fixture(size: u32, bodies: &[&[(u32, u32)]], food: Option<(u32, u32)>) -> Pool {
    let c = config(size, bodies.len() as u32, OBS_STATE);
    let mut p = pool(&c, 1);
    let s = board(size, bodies, food);
    assert_eq!(Snake::validate(&c, &s), Ok(()));
    import(&mut p, s).unwrap();
    p
}

fn cells(s: &SnakeState, k: usize) -> Vec<u16> {
    s.cells_of(k).collect()
}

const FAR: Option<(u32, u32)> = Some((7, 7));

#[test]
fn test_snake_state_is_4110_bytes_with_no_padding() {
    assert_eq!(size_of::<SnakeState>(), 4110);
    assert_eq!(align_of::<SnakeState>(), 2);
}

#[test]
fn test_snake_spaces_follow_obs_and_the_action_is_four_directions() {
    let pixels = Snake::observation_space(&config(16, 1, OBS_PIXELS)).unwrap();
    let flags = Snake::observation_space(&config(10, 2, OBS_STATE)).unwrap();
    let leaf = |space: Space| match space {
        Space::Dict(mut entries) => entries.remove(0).1,
        other => other,
    };
    assert_eq!(
        leaf(pixels),
        Space::Box {
            shape: vec![1, PIXELS, PIXELS],
            dtype: DType::U8,
            bounds: BoxBounds::UInt {
                low: vec![0],
                high: vec![255]
            },
        }
    );
    assert_eq!(
        leaf(flags),
        Space::Box {
            shape: vec![500],
            dtype: DType::F32,
            bounds: BoxBounds::Float {
                low: vec![0.0],
                high: vec![1.0]
            },
        }
    );
    assert_eq!(
        Snake::action_space(&SnakeConfig::default()).unwrap(),
        Space::Discrete { n: 4 }
    );
}

#[test]
fn test_snake_config_rejects_values_out_of_range() {
    for c in [
        config(5, 1, 0),
        config(33, 1, 0),
        config(8, 0, 0),
        config(8, 3, 0),
        config(8, 1, 2),
    ] {
        assert!(c.validate().is_err(), "{c:?}");
    }
    assert!(SnakeConfig::default().validate().is_ok());
}

#[test]
fn test_snake_reset_spawns_are_fixed_symmetric_and_food_is_free() {
    for size in [6, 7, 16, 32] {
        for agents in [1, 2] {
            let c = config(size, agents, OBS_STATE);
            let a = state(&pool(&c, 3));
            let b = state(&pool(&c, 4));
            assert_eq!(Snake::validate(&c, &a), Ok(()), "{size} {agents}");
            for k in 0..agents as usize {
                assert_eq!(
                    cells(&a, k),
                    cells(&b, k),
                    "spawns do not depend on the seed"
                );
                assert_eq!(a.len[k], 3);
            }
            if agents == 2 {
                // A half turn about the centre maps one body onto the other.
                let turn = |cell: u16| (size * size - 1) as u16 - cell;
                let spun: Vec<u16> = cells(&a, 0).into_iter().map(turn).collect();
                assert_eq!(spun, cells(&a, 1));
                assert_eq!(a.facing, [E, W]);
            }
        }
    }
    let c = config(6, 2, OBS_STATE);
    let foods: BTreeSet<u16> = (0..200).map(|seed| state(&pool(&c, seed)).food).collect();
    assert_eq!(foods.len(), 36 - 6, "food reaches every free cell");
}

#[test]
fn test_snake_mask_forbids_reversing_and_a_reversal_goes_straight() {
    let mut p = fixture(8, &[&[(3, 3), (3, 2), (3, 1)]], FAR);
    let mask: Vec<bool> = p.outputs().unwrap().masks()[..4]
        .iter()
        .map(|f| f.get())
        .collect();
    assert_eq!(
        mask,
        [true, true, true, false],
        "west reverses an east head"
    );
    step(&mut p, [S, N]);
    assert_eq!(state(&p).facing[0], S);
    // Without masks the kernel itself reads a reversal as going straight.
    let c = config(8, 1, OBS_STATE);
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    let mut q = EnvPool::<Snake, NoMasks>::new(c, options).unwrap();
    q.reset_all(Reseed::Keep).unwrap();
    let good = DecodedRecord::from(q.snapshot(0).unwrap());
    let east = board(8, &[&[(3, 3), (3, 2), (3, 1)]], FAR);
    let rec = DecodedRecord::from_parts(*good.header(), east).unwrap();
    q.import(&[0], &[rec]).unwrap();
    let west = [i32::from(W), 0];
    q.step(ActionInput {
        discrete: &west,
        continuous: &[],
    })
    .unwrap();
    let moved: Vec<u16> = q.state(0).unwrap().cells_of(0).collect();
    assert_eq!(moved, [at(8, (3, 4)), at(8, (3, 3)), at(8, (3, 2))]);
}

#[test]
fn test_snake_eats_grows_by_keeping_its_tail_and_new_food_is_free() {
    let mut p = fixture(8, &[&[(3, 3), (3, 2), (3, 1)]], Some((3, 4)));
    step(&mut p, [E, N]);
    let s = state(&p);
    assert_eq!(
        cells(&s, 0),
        [at(8, (3, 4)), at(8, (3, 3)), at(8, (3, 2)), at(8, (3, 1))]
    );
    assert_eq!(rewards(&p)[0], 1.0);
    assert!(s.food != NO_FOOD && !cells(&s, 0).contains(&s.food));
    assert_eq!(Snake::validate(&config(8, 1, OBS_STATE), &s), Ok(()));
}

#[test]
fn test_snake_off_the_board_dies_and_ends_a_lone_episode() {
    let mut p = fixture(8, &[&[(0, 3), (1, 3), (2, 3)]], FAR);
    step(&mut p, [N, N]);
    assert_eq!(rewards(&p)[0], -1.0);
    assert!(terminated(&p));
    assert!(!state(&p).alive[0].get() && state(&p).len[0] == 0);
}

#[test]
fn test_snake_into_its_own_body_dies() {
    // A loop of 5 whose head turns into its third segment.
    let body = [(2, 2), (2, 3), (3, 3), (3, 2), (3, 1)];
    let mut p = fixture(8, &[&body], FAR);
    step(&mut p, [S, N]);
    assert_eq!(rewards(&p)[0], -1.0);
}

#[test]
fn test_snake_into_its_own_leaving_tail_lives() {
    let body = [(2, 2), (2, 3), (3, 3), (3, 2)];
    let mut p = fixture(8, &[&body], FAR);
    step(&mut p, [S, N]);
    assert_eq!(rewards(&p)[0], 0.0);
    assert_eq!(cells(&state(&p), 0)[0], at(8, (3, 2)));
}

#[test]
fn test_snake_into_its_own_tail_dies_when_it_also_eats() {
    // The tail stays when its owner eats, but the food can never be on the
    // tail, so the owner cannot both eat and enter its tail: the other agent
    // does, and dies, as the eater keeps that cell.
    let a: &[(u32, u32)] = &[(1, 2), (1, 1), (1, 0)];
    let b: &[(u32, u32)] = &[(2, 0), (3, 0), (4, 0)];
    let mut p = fixture(8, &[a, b], Some((1, 3)));
    step(&mut p, [E, N]);
    assert_eq!(rewards(&p), [1.0, -1.0], "b enters a's kept tail");
    let mut q = fixture(8, &[a, b], FAR);
    step(&mut q, [E, N]);
    assert_eq!(
        rewards(&q),
        [0.0, 0.0],
        "a's tail leaves when a does not eat"
    );
}

#[test]
fn test_snake_two_heads_into_one_cell_both_die_even_on_food() {
    for food in [FAR, Some((3, 3))] {
        let a: &[(u32, u32)] = &[(3, 2), (3, 1), (3, 0)];
        let b: &[(u32, u32)] = &[(3, 4), (3, 5), (3, 6)];
        let mut p = fixture(8, &[a, b], food);
        step(&mut p, [E, W]);
        assert_eq!(rewards(&p), [-1.0, -1.0], "{food:?}");
        assert!(terminated(&p));
    }
}

#[test]
fn test_snake_swapped_heads_both_die() {
    let a: &[(u32, u32)] = &[(3, 3), (3, 2), (3, 1)];
    let b: &[(u32, u32)] = &[(3, 4), (3, 5), (3, 6)];
    let mut p = fixture(8, &[a, b], FAR);
    step(&mut p, [E, W]);
    assert_eq!(rewards(&p), [-1.0, -1.0]);
}

#[test]
fn test_snake_two_heads_each_entering_the_others_leaving_tail_both_live() {
    // Two bodies of 4 in a 2 x 4 cycle: each head enters the other's tail.
    let a: &[(u32, u32)] = &[(2, 3), (2, 2), (2, 1), (2, 0)];
    let b: &[(u32, u32)] = &[(3, 0), (3, 1), (3, 2), (3, 3)];
    let mut p = fixture(8, &[a, b], FAR);
    step(&mut p, [S, N]);
    assert_eq!(rewards(&p), [0.0, 0.0]);
    let s = state(&p);
    assert_eq!(cells(&s, 0)[0], at(8, (3, 3)));
    assert_eq!(cells(&s, 1)[0], at(8, (2, 0)));
}

#[test]
fn test_snake_entrant_follows_the_tail_of_an_agent_that_dies_on_a_wall() {
    // b's head leaves the board; a enters b's tail, which leaves as b does not
    // eat, and lives. b's body is gone after the step.
    let a: &[(u32, u32)] = &[(4, 3), (5, 3), (6, 3)];
    let b: &[(u32, u32)] = &[(3, 0), (3, 1), (3, 2), (3, 3)];
    let mut p = fixture(8, &[a, b], FAR);
    step(&mut p, [N, W]);
    assert_eq!(rewards(&p), [0.0, -1.0]);
    let s = state(&p);
    assert_eq!(cells(&s, 0)[0], at(8, (3, 3)));
    assert!(!s.alive[1].get() && s.body[1].iter().all(|&c| c == 0));
    assert!(!terminated(&p), "a survivor plays on");
    assert_eq!(Snake::validate(&config(8, 2, OBS_STATE), &s), Ok(()));
}

#[test]
fn test_snake_into_the_others_body_dies() {
    let a: &[(u32, u32)] = &[(2, 2), (1, 2), (0, 2)];
    let b: &[(u32, u32)] = &[(3, 4), (3, 3), (3, 2), (3, 1)];
    let mut p = fixture(8, &[a, b], FAR);
    step(&mut p, [S, E]);
    assert_eq!(rewards(&p), [-1.0, 0.0]);
}

#[test]
fn test_snake_dead_body_frees_its_cells_for_the_next_step() {
    let a: &[(u32, u32)] = &[(5, 2), (5, 1), (5, 0)];
    let b: &[(u32, u32)] = &[(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4)];
    let mut p = fixture(8, &[a, b], FAR);
    step(&mut p, [E, N]);
    assert_eq!(rewards(&p), [0.0, -1.0]);
    step(&mut p, [E, N]);
    assert_eq!(rewards(&p)[0], 0.0, "no corpse at (5, 4)");
    assert_eq!(cells(&state(&p), 0)[0], at(8, (5, 4)));
}

#[test]
fn test_snake_a_full_board_has_no_food_and_terminates() {
    // Side 6, one body of 35 in a snake path, the last free cell in front.
    let mut path = Vec::new();
    for row in 0..6 {
        for i in 0..6 {
            path.push((row, if row % 2 == 0 { i } else { 5 - i }));
        }
    }
    let food = path.pop();
    path.reverse();
    let c = config(6, 1, OBS_STATE);
    let mut p = pool(&c, 1);
    let s = board(6, &[&path], food);
    assert_eq!(Snake::validate(&c, &s), Ok(()));
    import(&mut p, s).unwrap();
    step(&mut p, [W, N]);
    let s = state(&p);
    assert_eq!(rewards(&p)[0], 1.0);
    assert!(terminated(&p));
    assert_eq!(s.food, NO_FOOD);
    assert_eq!(s.len[0], 36);
    assert_eq!(Snake::validate(&c, &s), Ok(()));
}

/// The reference: bodies as queues of `(row, col)`, head first, and the rules
/// written out once more, in a different order.
#[derive(Clone, Debug)]
struct Ref {
    n: i32,
    bodies: Vec<Option<VecDeque<(i32, i32)>>>,
    facing: Vec<u8>,
    food: Option<(i32, i32)>,
}

impl Ref {
    fn of(c: &SnakeConfig, s: &SnakeState) -> Self {
        let n = c.size as i32;
        let cell = |v: u16| (i32::from(v) / n, i32::from(v) % n);
        Self {
            n,
            bodies: (0..2)
                .map(|k| s.alive[k].get().then(|| s.cells_of(k).map(cell).collect()))
                .collect(),
            facing: s.facing.to_vec(),
            food: (s.food != NO_FOOD).then(|| cell(s.food)),
        }
    }

    /// Applies the actions; returns the rewards and the free cells after.
    fn step(&mut self, acts: [u8; 2]) -> ([f32; 2], BTreeSet<(i32, i32)>) {
        let d = |a: u8| [(-1, 0), (0, 1), (1, 0), (0, -1)][usize::from(a)];
        let mut target = [None, None];
        for k in 0..2 {
            if let Some(b) = &self.bodies[k] {
                let a = if acts[k] == (self.facing[k] + 2) % 4 {
                    self.facing[k]
                } else {
                    acts[k]
                };
                self.facing[k] = a;
                let h = b[0];
                target[k] = Some((h.0 + d(a).0, h.1 + d(a).1));
            }
        }
        let shared = target[0].is_some() && target[0] == target[1];
        let eat: Vec<bool> = (0..2)
            .map(|k| !shared && target[k].is_some() && target[k] == self.food)
            .collect();
        let mut blocked = BTreeSet::new();
        for k in 0..2 {
            if let Some(b) = &self.bodies[k] {
                let keep = if eat[k] { b.len() } else { b.len() - 1 };
                blocked.extend(b.iter().take(keep).copied());
            }
        }
        let mut dead = [false; 2];
        for k in 0..2 {
            let Some(t) = target[k] else { continue };
            let off = t.0 < 0 || t.1 < 0 || t.0 >= self.n || t.1 >= self.n;
            let swap = target[1 - k].is_some()
                && self.bodies[1 - k].as_ref().map(|b| b[0]) == Some(t)
                && self.bodies[k].as_ref().map(|b| b[0]) == target[1 - k];
            dead[k] = off || shared || swap || blocked.contains(&t);
        }
        let mut r = [0.0; 2];
        for k in 0..2 {
            if self.bodies[k].is_none() {
                continue;
            }
            if dead[k] {
                self.bodies[k] = None;
                r[k] = -1.0;
                continue;
            }
            let b = self.bodies[k].as_mut().unwrap();
            b.push_front(target[k].unwrap());
            if eat[k] {
                r[k] = 1.0;
            } else {
                b.pop_back();
            }
        }
        let mut free = BTreeSet::new();
        for row in 0..self.n {
            for col in 0..self.n {
                free.insert((row, col));
            }
        }
        for b in self.bodies.iter().flatten() {
            for c in b {
                free.remove(c);
            }
        }
        (r, free)
    }
}

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(0x5851_f42d_4c95_7f2d)
            .wrapping_add(0x1405_7b7e_f767_814f);
        (self.0 >> 33) % n
    }
}

#[test]
fn test_snake_random_play_matches_the_reference() {
    let mut rng = Lcg(0x5eed);
    let (mut eats, mut deaths, mut both) = (0, 0, 0);
    for seed in 0..300 {
        let c = config(6 + (seed % 4) as u32, 1 + (seed % 2) as u32, OBS_STATE);
        let mut p = pool(&c, seed);
        let mut r = Ref::of(&c, &state(&p));
        for _ in 0..400 {
            // Mostly the baseline, so bodies grow; sometimes anything.
            let s = state(&p);
            let acts = [0, 1].map(|k| {
                if rng.below(4) == 0 {
                    rng.below(4) as u8
                } else {
                    baseline(&c, &s, k)
                }
            });
            step(&mut p, acts);
            let (want, free) = r.step(acts);
            let s = state(&p);
            assert_eq!(
                rewards(&p)[..c.agents as usize],
                want[..c.agents as usize],
                "seed {seed}"
            );
            let fresh = Ref::of(&c, &s);
            assert_eq!(fresh.bodies, r.bodies, "seed {seed}");
            assert_eq!(Snake::validate(&c, &s), Ok(()), "seed {seed}");
            eats += want.iter().filter(|&&x| x == 1.0).count();
            deaths += want.iter().filter(|&&x| x == -1.0).count();
            both += usize::from(want == [-1.0, -1.0]);
            if want.contains(&1.0) {
                assert!(free.contains(&fresh.food.unwrap()), "seed {seed}");
            }
            r.food = fresh.food;
            let over = r.bodies.iter().all(Option::is_none) || free.is_empty();
            assert_eq!(terminated(&p), over, "seed {seed}");
            if over {
                break;
            }
        }
    }
    assert!(
        eats > 100 && deaths > 100 && both > 5,
        "{eats} {deaths} {both}"
    );
}

fn pixels(p: &Pool, k: usize) -> Vec<u8> {
    let side = (PIXELS * PIXELS) as usize;
    p.outputs().unwrap().observation(0).unwrap()[k * side..(k + 1) * side].to_vec()
}

fn flags(p: &Pool, k: usize, cells: usize) -> Vec<f32> {
    let n = 5 * cells;
    p.outputs().unwrap().observation(0).unwrap()[k * n * 4..(k + 1) * n * 4]
        .chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// A rasterizer of its own: every pixel asks which cell it lies in.
fn raster(c: &SnakeConfig, s: &SnakeState, me: usize) -> Vec<u8> {
    let n = c.size;
    let scale = PIXELS / n;
    let margin = (PIXELS - scale * n) / 2;
    let mut out = Vec::new();
    for y in 0..PIXELS {
        for x in 0..PIXELS {
            let inside = (margin..margin + scale * n).contains(&y)
                && (margin..margin + scale * n).contains(&x);
            if !inside {
                out.push(SHADE_OUTSIDE);
                continue;
            }
            let cell = at(n, ((y - margin) / scale, (x - margin) / scale));
            let own: Vec<u16> = s.cells_of(me).collect();
            let other: Vec<u16> = s.cells_of(1 - me).collect();
            out.push(match () {
                () if own.first() == Some(&cell) => SHADE_OWN_HEAD,
                () if own.contains(&cell) => SHADE_OWN_BODY,
                () if other.first() == Some(&cell) => SHADE_OTHER_HEAD,
                () if other.contains(&cell) => SHADE_OTHER_BODY,
                () if s.food == cell => SHADE_FOOD,
                () => SHADE_FREE,
            });
        }
    }
    out
}

#[test]
fn test_snake_pixels_match_the_reference_rasterizer_for_each_observer() {
    let mut rng = Lcg(9);
    for size in [6, 7, 13, 16, 32] {
        let c = config(size, 2, OBS_PIXELS);
        let mut p = pool(&c, u64::from(size));
        for _ in 0..30 {
            let s = state(&p);
            // The engine observes live agents only.
            for k in (0..2).filter(|&k| s.alive[k].get()) {
                assert_eq!(pixels(&p, k), raster(&c, &s, k), "size {size} slot {k}");
            }
            step(&mut p, [rng.below(4) as u8, rng.below(4) as u8]);
            if terminated(&p) {
                break;
            }
        }
    }
}

#[test]
fn test_snake_state_planes_are_observer_relative_flags() {
    let a: &[(u32, u32)] = &[(2, 3), (2, 2), (2, 1)];
    let b: &[(u32, u32)] = &[(5, 1), (5, 2), (5, 3)];
    let p = fixture(8, &[a, b], Some((7, 0)));
    let s = state(&p);
    for me in 0..2 {
        let v = flags(&p, me, 64);
        let plane = |k: usize| -> Vec<u16> {
            (0..64u16)
                .filter(|&c| v[k * 64 + usize::from(c)] == 1.0)
                .collect()
        };
        let own = cells(&s, me);
        let other = cells(&s, 1 - me);
        assert_eq!(plane(0), [own[0]]);
        assert_eq!(
            plane(1),
            own[1..]
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
        );
        assert_eq!(plane(2), [other[0]]);
        assert_eq!(plane(3).len(), 2);
        assert_eq!(plane(4), [at(8, (7, 0))]);
        assert!(v.iter().all(|&x| x == 0.0 || x == 1.0));
    }
}

#[test]
fn test_snake_validate_names_each_rejection() {
    let c = config(8, 2, OBS_STATE);
    let a: &[(u32, u32)] = &[(2, 3), (2, 2), (2, 1)];
    let b: &[(u32, u32)] = &[(5, 1), (5, 2), (5, 3)];
    let good = board(8, &[a, b], FAR);
    assert_eq!(Snake::validate(&c, &good), Ok(()));
    let rejected = |s: &SnakeState, code: &str| {
        assert_eq!(Snake::validate(&c, s).unwrap_err().code, code);
    };
    let mut s = good;
    s.len[0] = 2;
    rejected(&s, "a body length out of range");
    let mut s = good;
    s.len[0] = 65;
    rejected(&s, "a body length out of range");
    let mut s = good;
    s.head[0] = RING as u16;
    rejected(&s, "a head cursor out of range");
    let mut s = good;
    s.facing[0] = 4;
    rejected(&s, "a facing out of range");
    let mut s = good;
    s.facing[0] = S;
    rejected(&s, "a facing that does not lead from the neck");
    let mut s = good;
    s.body[0][usize::from(s.head[0])] = 64;
    rejected(&s, "a body cell off the board");
    let mut s = good;
    s.body[0][(usize::from(s.head[0]) + RING - 2) % RING] = at(8, (4, 1));
    rejected(&s, "a body that is not connected");
    let crossing: &[(u32, u32)] = &[(5, 2), (4, 2), (3, 2), (2, 2)];
    rejected(
        &board(8, &[a, crossing], FAR),
        "two body segments on one cell",
    );
    let mut s = good;
    s.body[0][(usize::from(s.head[0]) + 1) % RING] = 9;
    rejected(&s, "a value in an unused ring cell");
    let mut s = good;
    s.alive[1] = Bool8::FALSE;
    rejected(&s, "a value in a dead or unused slot");
    rejected(&board(8, &[a, b], Some((2, 2))), "food on a body");
    let mut s = good;
    s.food = 64;
    rejected(&s, "food off the board");
    rejected(
        &board(8, &[a, b], None),
        "no food on a board with free cells",
    );
    let one = config(8, 1, OBS_STATE);
    assert_eq!(
        Snake::validate(&one, &good).unwrap_err().code,
        "an unused slot that is alive"
    );
}

#[test]
fn test_snake_import_checks_alive_flags_and_running_episodes() {
    let c = config(8, 2, OBS_STATE);
    let mut p = pool(&c, 1);
    let before = p.state_hash(0).unwrap();
    // Slot 1 dead in the state but alive in the header.
    let mut s = board(8, &[&[(2, 3), (2, 2), (2, 1)]], FAR);
    s.alive[1] = Bool8::FALSE;
    assert_eq!(Snake::validate(&c, &s), Ok(()));
    assert!(invalid(&import(&mut p, s)));
    assert_eq!(p.state_hash(0).unwrap(), before);
    // Nobody alive, in a running episode.
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut header = *good.header();
    header.agents[0].alive = Bool8::FALSE;
    header.agents[1].alive = Bool8::FALSE;
    let rec = DecodedRecord::from_parts(header, SnakeState::default()).unwrap();
    assert!(invalid(&p.import(&[0], &[rec])));
    // An unused slot alive or acting.
    let one = config(8, 1, OBS_STATE);
    let mut q = pool(&one, 1);
    let good = DecodedRecord::from(q.snapshot(0).unwrap());
    for (alive, acting) in [(true, true), (true, false), (false, true)] {
        let mut header = *good.header();
        header.agents[1].alive = Bool8::new(alive);
        header.agents[1].acting = Bool8::new(acting);
        let bad = DecodedRecord::from_parts(header, *good.state()).unwrap();
        assert!(invalid(&q.import(&[0], &[bad])), "{alive} {acting}");
    }
}

#[test]
fn test_snake_never_reset_record_round_trips_and_refuses_anything_else() {
    let c = config(8, 2, OBS_STATE);
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    let p = Pool::with_masks(c.clone(), options).unwrap();
    let rec = DecodedRecord::from(p.snapshot(0).unwrap());
    let header = *rec.header();
    let mut q = Pool::with_masks(c, options).unwrap();
    q.import(&[0], &[rec]).unwrap();
    let bad =
        DecodedRecord::from_parts(header, board(8, &[&[(2, 3), (2, 2), (2, 1)]], FAR)).unwrap();
    assert!(invalid(&q.import(&[0], &[bad])));
}

/// Perturbations of reachable states: what `Validate` lets through plays on
/// without panicking and stays valid.
#[test]
fn test_snake_perturbed_imports_stay_valid_when_accepted() {
    let mut rng = Lcg(77);
    let (mut no, mut yes) = (0, 0);
    let c = config(6, 2, OBS_STATE);
    for seed in 0..80 {
        let mut p = pool(&c, seed);
        for _ in 0..20 {
            step(&mut p, [rng.below(4) as u8, rng.below(4) as u8]);
            if terminated(&p) {
                break;
            }
            let mut t = state(&p);
            let k = rng.below(2) as usize;
            match rng.below(6) {
                0 => t.body[k][rng.below(RING as u64) as usize] = rng.below(40) as u16,
                1 => t.head[k] = rng.below(RING as u64 + 2) as u16,
                2 => t.len[k] = rng.below(40) as u16,
                3 => t.facing[k] = rng.below(5) as u8,
                4 => t.food = rng.below(40) as u16,
                _ => t.alive[k] = Bool8::new(rng.below(2) == 0),
            }
            if Snake::validate(&c, &t).is_err() {
                no += 1;
                continue;
            }
            yes += 1;
            let mut q = pool(&c, seed);
            if import(&mut q, t).is_err() {
                continue;
            }
            for _ in 0..20 {
                step(&mut q, [rng.below(4) as u8, rng.below(4) as u8]);
                assert_eq!(Snake::validate(&c, &state(&q)), Ok(()));
                if terminated(&q) {
                    break;
                }
            }
        }
    }
    assert!(no > 100 && yes > 50, "{no} {yes}");
}

fn baseline(c: &SnakeConfig, s: &SnakeState, k: u16) -> u8 {
    let mut discrete = [0];
    let mut writer = ActionWriter::new(&mut discrete, &mut []);
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
    Snake::act(c, s, AgentSlot::new(k, 2).unwrap(), &mut rng, &mut writer);
    discrete[0] as u8
}

/// The baseline's mean food per episode within a bound, pinned per agent count.
#[test]
fn test_snake_baseline_food_is_pinned() {
    let mut food = [0u32; 2];
    for agents in 1..=2u32 {
        let c = config(10, agents, OBS_STATE);
        for seed in 0..100 {
            let mut p = pool(&c, seed);
            for _ in 0..300 {
                let s = state(&p);
                let acts = [baseline(&c, &s, 0), baseline(&c, &s, 1)];
                step(&mut p, acts);
                food[agents as usize - 1] +=
                    rewards(&p).iter().filter(|&&r| r == 1.0).count() as u32;
                if terminated(&p) {
                    break;
                }
            }
        }
    }
    assert_eq!(food, [1939, 647]);
}
