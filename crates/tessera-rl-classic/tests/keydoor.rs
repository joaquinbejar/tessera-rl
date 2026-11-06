// SPDX-License-Identifier: MIT OR Apache-2.0

//! Key and door through the engine, against a reference written only for these
//! tests: generation, a solver that plays every world to the goal, the mask,
//! slot-order conflicts, the egocentric view, import validation and the
//! baseline.

#![expect(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::float_cmp,
    reason = "test helpers on a small room: a failure is the assertion"
)]

use core::num::NonZeroU32;
use std::collections::{BTreeSet, VecDeque};

use tessera_rl_classic::keydoor::{FORWARD, INTERACT, LEFT, NO_HOLDER, RIGHT, WAIT};
use tessera_rl_classic::{KeyDoor, KeyDoorConfig, KeyDoorState};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{AgentSlot, Bool8, EnvConfig, Environment, Heuristic, Validate};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, Masked, PoolOptions, Reseed,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

type Pool = EnvPool<KeyDoor, Masked>;

fn config(size: u32, agents: u32) -> KeyDoorConfig {
    KeyDoorConfig { size, agents }
}

fn pool_with(c: &KeyDoorConfig, seed: u64, auto_reset: AutoReset) -> Pool {
    let options = PoolOptions {
        auto_reset,
        max_episode_steps: NonZeroU32::new(100_000),
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(seed))
    };
    let mut p = EnvPool::with_masks(c.clone(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn pool(c: &KeyDoorConfig, seed: u64) -> Pool {
    pool_with(c, seed, AutoReset::Disabled)
}

/// Steps with one action per slot; slots beyond the agents wait.
fn step(p: &mut Pool, acts: &[i32]) {
    let mut row = [WAIT; 4];
    row[..acts.len()].copy_from_slice(acts);
    p.step(ActionInput {
        discrete: &row,
        continuous: &[],
    })
    .unwrap();
}

fn step_one(p: &mut Pool, slot: usize, action: i32) {
    let mut row = [WAIT; 4];
    row[slot] = action;
    step(p, &row);
}

/// Imports a state, the header's acting flags already fit a live record.
fn import(p: &mut Pool, s: KeyDoorState) -> Result<(), EngineError> {
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

/// A fixed world of side 8: wall at column 3, door at row 4, key at (0,0), goal
/// at (6,6), with agents placed by the caller, facing north.
fn world(agents: &[[u8; 2]]) -> KeyDoorState {
    let mut s = KeyDoorState {
        key: [0, 0],
        goal: [6, 6],
        wall_col: 3,
        door_row: 4,
        key_holder: NO_HOLDER,
        ..KeyDoorState::default()
    };
    for (k, &p) in agents.iter().enumerate() {
        s.pos[k] = p;
    }
    s
}

fn rejected(c: &KeyDoorConfig, s: &KeyDoorState, code: &str) {
    assert_eq!(KeyDoor::validate(c, s).unwrap_err().code, code);
}

/// The reference: a cell array and the rules written out once more.
#[derive(Clone)]
struct Ref {
    n: i32,
    agents: usize,
    /// Each cell: b'#' wall, b'D' closed door, b'.' floor, b'G' goal, b'K' key.
    cells: Vec<u8>,
    pos: Vec<(i32, i32)>,
    facing: Vec<usize>,
    done: Vec<bool>,
    holder: Option<usize>,
}

const DIRS: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

impl Ref {
    fn of(c: &KeyDoorConfig, s: &KeyDoorState) -> Self {
        let n = c.size as i32;
        let mut cells = vec![b'.'; (n * n) as usize];
        for r in 0..n {
            cells[(r * n + i32::from(s.wall_col)) as usize] = b'#';
        }
        cells[(i32::from(s.door_row) * n + i32::from(s.wall_col)) as usize] = b'D';
        cells[(i32::from(s.key[1]) * n + i32::from(s.key[0])) as usize] = b'K';
        cells[(i32::from(s.goal[1]) * n + i32::from(s.goal[0])) as usize] = b'G';
        let agents = c.agents as usize;
        Self {
            n,
            agents,
            cells,
            pos: (0..agents)
                .map(|k| (i32::from(s.pos[k][0]), i32::from(s.pos[k][1])))
                .collect(),
            facing: (0..agents).map(|k| usize::from(s.facing[k])).collect(),
            done: vec![false; agents],
            holder: None,
        }
    }

    fn cell(&self, c: i32, r: i32) -> u8 {
        if c < 0 || r < 0 || c >= self.n || r >= self.n {
            b'#'
        } else {
            self.cells[(r * self.n + c) as usize]
        }
    }

    fn front(&self, k: usize) -> (i32, i32) {
        let (dc, dr) = DIRS[self.facing[k]];
        (self.pos[k].0 + dc, self.pos[k].1 + dr)
    }

    fn legal(&self, k: usize) -> Vec<i32> {
        if self.done[k] {
            return vec![WAIT];
        }
        let mut out = vec![LEFT, RIGHT];
        let (c, r) = self.front(k);
        let v = self.cell(c, r);
        let agent_there =
            (0..self.agents).any(|j| j != k && !self.done[j] && self.pos[j] == (c, r));
        if (v == b'.' || v == b'G') && !agent_there {
            out.push(FORWARD);
        }
        if v == b'K' || (v == b'D' && self.holder == Some(k)) {
            out.push(INTERACT);
        }
        out.push(WAIT);
        out
    }

    fn apply(&mut self, k: usize, a: i32) {
        if !self.legal(k).contains(&a) {
            return;
        }
        let (c, r) = self.front(k);
        match a {
            LEFT => self.facing[k] = (self.facing[k] + 3) % 4,
            RIGHT => self.facing[k] = (self.facing[k] + 1) % 4,
            FORWARD => {
                self.pos[k] = (c, r);
                self.done[k] = self.cell(c, r) == b'G';
            }
            INTERACT => {
                if self.cell(c, r) == b'K' {
                    self.holder = Some(k);
                }
                self.cells[(r * self.n + c) as usize] = b'.';
            }
            _ => {}
        }
    }

    /// The expected view of slot `k`: written out per facing, by hand.
    fn view(&self, k: usize) -> Vec<u8> {
        let mut out = vec![0u8; 5 * 49];
        for vr in 0..7i32 {
            for vc in 0..7i32 {
                let (a, b) = (6 - vr, vc - 3);
                let (x, y) = self.pos[k];
                let (c, r) = match self.facing[k] {
                    0 => (x + b, y - a),
                    1 => (x + a, y + b),
                    2 => (x - b, y + a),
                    _ => (x - a, y - b),
                };
                let at = (vr * 7 + vc) as usize;
                let v = self.cell(c, r);
                let others = (0..self.agents).any(|j| j != k && self.pos[j] == (c, r));
                let on = [v == b'#', v == b'D', v == b'K', v == b'G', others];
                for (ch, flag) in on.iter().enumerate() {
                    out[ch * 49 + at] = u8::from(*flag);
                }
            }
        }
        out
    }
}

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self.0.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
        (self.0 >> 33) % n
    }
}

fn masks(p: &Pool, k: usize) -> Vec<i32> {
    p.outputs().masks()[k * 5..(k + 1) * 5]
        .iter()
        .enumerate()
        .filter(|(_, f)| f.get())
        .map(|(a, _)| a as i32)
        .collect()
}

/// Slot `k`'s observation: the `view` leaf and the `carrying` flag.
fn view(p: &Pool, k: usize) -> (Vec<u8>, u8) {
    let out = p.outputs();
    let views = out.observation(0).unwrap();
    let flags = out.observation(1).unwrap();
    (views[k * 245..(k + 1) * 245].to_vec(), flags[k])
}

#[test]
fn test_keydoor_state_is_24_bytes() {
    assert_eq!(size_of::<KeyDoorState>(), 24);
}

#[test]
fn test_keydoor_spaces_are_the_view_the_flag_and_five_actions() {
    let c = KeyDoorConfig::default();
    let leaf = |shape: Vec<u32>| Space::Box {
        shape,
        dtype: DType::U8,
        bounds: BoxBounds::UInt {
            low: vec![0],
            high: vec![1],
        },
    };
    let Space::Dict(mut got) = KeyDoor::observation_space(&c).unwrap() else {
        panic!("a dict");
    };
    got.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        got,
        vec![
            ("carrying".to_owned(), leaf(vec![1])),
            ("view".to_owned(), leaf(vec![5, 7, 7])),
        ]
    );
    assert_eq!(KeyDoor::action_space(&c).unwrap(), Space::Discrete { n: 5 });
}

#[test]
fn test_keydoor_config_rejects_values_out_of_range() {
    for (size, agents) in [(5, 1), (17, 1), (8, 0), (8, 5)] {
        assert!(config(size, agents).validate().is_err(), "{size} {agents}");
    }
}

/// Generation is a function of the seed alone, stays in range, and reaches
/// every wall column and door row.
#[test]
fn test_keydoor_generation_is_seeded_valid_and_covers_the_ranges() {
    for agents in 1..=4 {
        for size in [6, 8, 16] {
            let c = config(size, agents);
            let (mut walls, mut doors) = (BTreeSet::new(), BTreeSet::new());
            let mut worlds = BTreeSet::new();
            for seed in 0..600 {
                let s = *pool(&c, seed).state(0).unwrap();
                assert_eq!(*pool(&c, seed).state(0).unwrap(), s, "seed {seed}");
                assert_eq!(KeyDoor::validate(&c, &s), Ok(()), "seed {seed}");
                walls.insert(s.wall_col);
                doors.insert(s.door_row);
                worlds.insert((s.wall_col, s.door_row, s.key, s.goal, s.pos));
            }
            assert_eq!(walls.len() as u32, size - 4, "{size}");
            assert_eq!(doors.len() as u32, size, "{size}");
            assert!(worlds.len() > 100, "{}", worlds.len());
        }
    }
}

/// Moves on the floor from `from` to any of `targets`, around walls, the closed
/// door, the key and the other agents standing still.
fn path(
    c: &KeyDoorConfig,
    s: &KeyDoorState,
    me: usize,
    targets: &[(i32, i32)],
) -> Option<Vec<usize>> {
    let from = (i32::from(s.pos[me][0]), i32::from(s.pos[me][1]));
    let mut seen = BTreeSet::from([from]);
    let mut queue = VecDeque::from([(from, Vec::new())]);
    while let Some((at, dirs)) = queue.pop_front() {
        if targets.contains(&at) {
            return Some(dirs);
        }
        for (d, (dc, dr)) in DIRS.iter().enumerate() {
            let next = (at.0 + dc, at.1 + dr);
            if s.can_enter(c, next.0, next.1, me) && seen.insert(next) {
                let mut more = dirs.clone();
                more.push(d);
                queue.push_back((next, more));
            }
        }
    }
    None
}

fn state(p: &Pool) -> KeyDoorState {
    *p.state(0).unwrap()
}

fn face(p: &mut Pool, slot: usize, dir: usize) {
    while usize::from(state(p).facing[slot]) != dir {
        step_one(p, slot, RIGHT);
    }
}

fn walk(p: &mut Pool, slot: usize, dirs: &[usize]) {
    for &d in dirs {
        face(p, slot, d);
        step_one(p, slot, FORWARD);
    }
}

/// Walks `slot` to one of `targets`. While the others block every route, they
/// shuffle by seeded legal moves; a bounded search, and reaching the end of a
/// world through the engine is the proof that it is solvable.
fn reach(
    p: &mut Pool,
    c: &KeyDoorConfig,
    slot: usize,
    targets: &[(i32, i32)],
    rng: &mut Lcg,
) -> bool {
    for _ in 0..20_000 {
        if let Some(route) = path(c, &state(p), slot, targets) {
            walk(p, slot, &route);
            return true;
        }
        let other = rng.below(u64::from(c.agents)) as usize;
        if other != slot && !state(p).is_finished(other) {
            let a = [LEFT, RIGHT, FORWARD, FORWARD][rng.below(4) as usize];
            if state(p).is_legal(c, other, a) {
                step_one(p, other, a);
            }
        }
    }
    false
}

/// Plays one world to the end with moves by one agent at a time: slot 0 takes
/// the key and opens the door, then every agent walks to the goal. Returns
/// whether it ended with +1 for all.
fn solve(c: &KeyDoorConfig, seed: u64) -> bool {
    let mut rng = Lcg(seed ^ 0x501e);
    let mut p = pool(c, seed);
    let s = state(&p);
    let (kc, kr) = (i32::from(s.key[0]), i32::from(s.key[1]));
    let beside: Vec<(i32, i32)> = DIRS.iter().map(|d| (kc + d.0, kr + d.1)).collect();
    if !reach(&mut p, c, 0, &beside, &mut rng) {
        return false;
    }
    let at = (
        i32::from(state(&p).pos[0][0]),
        i32::from(state(&p).pos[0][1]),
    );
    let Some(dir) = DIRS
        .iter()
        .position(|d| (at.0 + d.0, at.1 + d.1) == (kc, kr))
    else {
        return false;
    };
    face(&mut p, 0, dir);
    step_one(&mut p, 0, INTERACT);
    let before = (i32::from(s.wall_col) - 1, i32::from(s.door_row));
    if !reach(&mut p, c, 0, &[before], &mut rng) {
        return false;
    }
    face(&mut p, 0, 1);
    step_one(&mut p, 0, INTERACT);
    let goal = (i32::from(s.goal[0]), i32::from(s.goal[1]));
    for k in 0..c.agents as usize {
        if !reach(&mut p, c, k, &[goal], &mut rng) {
            return false;
        }
    }
    let out = p.outputs();
    out.episode_terminated()[0].get()
        && out.rewards()[..c.agents as usize].iter().all(|&r| r == 1.0)
}

/// Sampled solvability: the solver finishes all 2,400 generated worlds drawn
/// here, across agent counts and sizes. This is not a proof for every seed.
#[test]
fn test_keydoor_solver_finishes_all_sampled_worlds() {
    for agents in 1..=4 {
        for size in [6, 7, 8, 16] {
            for seed in 0..150 {
                let c = config(size, agents);
                assert!(solve(&c, seed), "size {size}, agents {agents}, seed {seed}");
            }
        }
    }
}

/// Random play compared with the reference: state, masks, views, rewards, ends.
#[test]
fn test_keydoor_random_play_matches_the_reference() {
    let mut rng = Lcg(0xfeed);
    let (mut keys, mut doors, mut ends) = (0, 0, 0);
    for seed in 0..120 {
        let c = config(6 + (seed % 3) as u32, 1 + (seed % 4) as u32);
        let mut p = pool(&c, seed);
        let mut r = Ref::of(&c, &state(&p));
        for _ in 0..800 {
            let mut acts = [WAIT; 4];
            for (k, a) in acts.iter_mut().enumerate().take(r.agents) {
                // Mostly legal, leaning to moves that make progress.
                let legal = r.legal(k);
                let best = legal.iter().copied().find(|&x| x == INTERACT);
                *a = best.unwrap_or_else(|| legal[rng.below(legal.len() as u64) as usize]);
            }
            step(&mut p, &acts);
            for (k, &a) in acts.iter().enumerate().take(r.agents) {
                r.apply(k, a);
            }
            let s = state(&p);
            for k in 0..r.agents {
                assert_eq!(
                    (i32::from(s.pos[k][0]), i32::from(s.pos[k][1])),
                    r.pos[k],
                    "seed {seed}"
                );
                assert_eq!(usize::from(s.facing[k]), r.facing[k], "seed {seed}");
                assert_eq!(s.is_finished(k), r.done[k], "seed {seed}");
                assert_eq!(masks(&p, k), r.legal(k), "seed {seed}");
                let (seen, carrying) = view(&p, k);
                assert_eq!(seen, r.view(k), "seed {seed}");
                assert_eq!(carrying, u8::from(r.holder == Some(k)), "seed {seed}");
            }
            assert_eq!(KeyDoor::validate(&c, &s), Ok(()), "seed {seed}");
            let all = r.done.iter().all(|&d| d);
            assert_eq!(
                p.outputs().episode_terminated()[0].get(),
                all,
                "seed {seed}"
            );
            let pay = if all { 1.0 } else { 0.0 };
            assert!(p.outputs().rewards()[..r.agents].iter().all(|&x| x == pay));
            keys += i32::from(r.holder.is_some());
            doors += i32::from(s.door_open.get());
            if all {
                ends += 1;
                break;
            }
        }
    }
    assert!(keys > 0 && doors > 0 && ends > 0, "{keys} {doors} {ends}");
}

fn two_agents_at(a: [u8; 2], fa: u8, b: [u8; 2], fb: u8) -> (KeyDoorConfig, Pool) {
    let c = config(8, 2);
    let mut p = pool(&c, 1);
    let mut s = world(&[a, b]);
    s.facing[0] = fa;
    s.facing[1] = fb;
    import(&mut p, s).unwrap();
    (c, p)
}

#[test]
fn test_keydoor_two_agents_into_one_cell_the_lower_slot_wins() {
    let (_, mut p) = two_agents_at([0, 2], 1, [2, 2], 3);
    step(&mut p, &[FORWARD, FORWARD]);
    assert_eq!(state(&p).pos[0], [1, 2]);
    assert_eq!(state(&p).pos[1], [2, 2], "the second move found it taken");
}

/// The mask is read before the step, so a cell occupied then is refused even if
/// its agent leaves in that step, whichever slot leaves.
#[test]
fn test_keydoor_following_is_refused_in_both_slot_orders() {
    for (leader, follower, cells) in [(0, 1, [[1, 2], [0, 2]]), (1, 0, [[0, 2], [1, 2]])] {
        let (_, mut p) = two_agents_at(cells[0], 1, cells[1], 1);
        assert!(masks(&p, leader).contains(&FORWARD));
        assert!(!masks(&p, follower).contains(&FORWARD));
        let before = state(&p).pos[follower];
        let mut row = [WAIT; 4];
        row[leader] = FORWARD;
        row[follower] = FORWARD;
        let stepped = p
            .step(ActionInput {
                discrete: &row,
                continuous: &[],
            })
            .map(|_| ());
        if let Err(e) = stepped {
            // `debug-checks`: rejected atomically.
            assert!(matches!(e, EngineError::IllegalAction { .. }), "{e:?}");
        } else {
            // Default: replaced by the no-op.
            assert_eq!(state(&p).pos[follower], before);
            assert_eq!(state(&p).pos[leader], [2, 2]);
        }
    }
}

#[test]
fn test_keydoor_two_agents_on_the_key_the_lower_slot_takes_it() {
    let (_, mut p) = two_agents_at([0, 1], 0, [1, 0], 3);
    step(&mut p, &[INTERACT, INTERACT]);
    assert_eq!(state(&p).key_holder, 0);
    assert!(!masks(&p, 1).contains(&INTERACT));
    assert_eq!(view(&p, 0).1, 1);
    assert_eq!(view(&p, 1).1, 0);
}

#[test]
fn test_keydoor_door_needs_the_key_and_stays_open_for_all() {
    let c = config(8, 2);
    let mut p = pool(&c, 1);
    let mut s = world(&[[2, 4], [2, 3]]);
    s.facing = [1, 1, 0, 0];
    import(&mut p, s).unwrap();
    assert!(!masks(&p, 0).contains(&FORWARD), "the closed door blocks");
    assert!(!masks(&p, 0).contains(&INTERACT), "no key");
    let mut held = world(&[[2, 4], [2, 3]]);
    held.facing = [1, 1, 0, 0];
    held.key_holder = 1;
    import(&mut p, held).unwrap();
    assert!(!masks(&p, 0).contains(&INTERACT), "the other agent has it");
    held.key_holder = 0;
    import(&mut p, held).unwrap();
    assert!(masks(&p, 0).contains(&INTERACT));
    step_one(&mut p, 0, INTERACT);
    assert!(state(&p).door_open.get());
    assert!(!masks(&p, 0).contains(&INTERACT), "already open");
    step_one(&mut p, 0, FORWARD);
    assert_eq!(state(&p).pos[0], [3, 4]);
}

#[test]
fn test_keydoor_the_goal_pays_only_when_the_last_agent_arrives() {
    let c = config(8, 2);
    let mut p = pool(&c, 1);
    let mut s = world(&[[5, 6], [6, 5]]);
    s.facing = [1, 2, 0, 0];
    s.door_open = Bool8::TRUE;
    s.key_holder = 0;
    import(&mut p, s).unwrap();
    step_one(&mut p, 0, FORWARD);
    assert!(state(&p).is_finished(0));
    assert_eq!(masks(&p, 0), [WAIT], "a finished agent may only wait");
    assert!(!p.outputs().episode_terminated()[0].get());
    assert_eq!(p.outputs().rewards()[..2], [0.0, 0.0]);
    assert!(masks(&p, 1).contains(&FORWARD), "the goal holds both");
    step_one(&mut p, 1, FORWARD);
    assert!(p.outputs().episode_terminated()[0].get());
    assert_eq!(p.outputs().rewards()[..2], [1.0, 1.0]);
}

#[test]
fn test_keydoor_key_blocks_until_taken() {
    let c = config(8, 1);
    let mut p = pool(&c, 1);
    let mut s = world(&[[0, 1]]);
    s.facing[0] = 0;
    import(&mut p, s).unwrap();
    assert_eq!(masks(&p, 0), [LEFT, RIGHT, INTERACT, WAIT]);
    step_one(&mut p, 0, INTERACT);
    assert_eq!(masks(&p, 0), [LEFT, RIGHT, FORWARD, WAIT]);
}

/// The view rotates with the facing: the key two cells ahead appears straight
/// up from the agent, whichever way it looks.
#[test]
fn test_keydoor_view_rotates_with_the_facing() {
    let c = config(8, 1);
    let cases = [
        ([1, 1], [1, 3], 0u8),
        ([2, 1], [0, 1], 1),
        ([1, 2], [1, 0], 2),
        ([0, 1], [2, 1], 3),
    ];
    for (key, agent, f) in cases {
        let mut p = pool(&c, 1);
        let mut s = world(&[agent]);
        s.key = key;
        s.facing[0] = f;
        import(&mut p, s).unwrap();
        // Channel 2 is the key: two cells ahead of row 6, column 3.
        let (seen, _) = view(&p, 0);
        let on: Vec<usize> = (0..49).filter(|&i| seen[2 * 49 + i] == 1).collect();
        assert_eq!(on, [4 * 7 + 3], "facing {f}");
    }
}

#[test]
fn test_keydoor_auto_reset_draws_a_new_world() {
    let c = config(8, 1);
    let mut p = pool_with(&c, 3, AutoReset::SameStep);
    let first = state(&p);
    let mut s = first;
    s.pos[0] = [s.goal[0] - 1, s.goal[1]];
    if s.goal[0] - 1 == s.wall_col {
        s.pos[0] = [s.goal[0], s.goal[1] + u8::from(s.goal[1] == 0) * 2 - 1];
    }
    s.door_open = Bool8::TRUE;
    s.key_holder = 0;
    s.facing[0] = if s.pos[0][0] < s.goal[0] {
        1
    } else if s.pos[0][1] < s.goal[1] {
        2
    } else {
        0
    };
    import(&mut p, s).unwrap();
    step_one(&mut p, 0, FORWARD);
    assert!(p.outputs().episode_terminated()[0].get());
    assert_eq!(state(&p).key_holder, NO_HOLDER);
    assert!(!state(&p).door_open.get());
}

#[test]
fn test_keydoor_validate_names_each_rejection() {
    let c = config(8, 2);
    let good = world(&[[0, 2], [1, 2]]);
    assert_eq!(KeyDoor::validate(&c, &good), Ok(()));
    let with = |f: &dyn Fn(&mut KeyDoorState)| {
        let mut s = good;
        f(&mut s);
        s
    };
    let wall_low = with(&|s| s.wall_col = 1);
    rejected(&c, &wall_low, "a wall or door out of range");
    let wall_high = with(&|s| s.wall_col = 6);
    rejected(&c, &wall_high, "a wall or door out of range");
    rejected(
        &c,
        &with(&|s| s.door_row = 8),
        "a wall or door out of range",
    );
    rejected(&c, &with(&|s| s.key = [3, 0]), "a key off the left side");
    rejected(&c, &with(&|s| s.key = [0, 8]), "a key off the left side");
    rejected(&c, &with(&|s| s.goal = [3, 0]), "a goal off the right side");
    rejected(&c, &with(&|s| s.goal = [8, 0]), "a goal off the right side");
    rejected(&c, &with(&|s| s.goal = [6, 8]), "a goal off the right side");
    rejected(
        &c,
        &with(&|s| s.key_holder = 2),
        "a holder that is not an agent",
    );
    rejected(
        &c,
        &with(&|s| s.door_open = Bool8::TRUE),
        "an open door that nobody had the key for",
    );
    rejected(
        &c,
        &with(&|s| s.pos[2] = [1, 1]),
        "a value in an unused slot",
    );
    rejected(&c, &with(&|s| s.facing[3] = 1), "a value in an unused slot");
    rejected(
        &c,
        &with(&|s| s.finished[2] = Bool8::TRUE),
        "a value in an unused slot",
    );
    rejected(&c, &with(&|s| s.pos[0] = [8, 0]), "an agent off the room");
    rejected(&c, &with(&|s| s.pos[0] = [0, 8]), "an agent off the room");
    rejected(&c, &with(&|s| s.facing[0] = 4), "a facing out of range");
    rejected(
        &c,
        &with(&|s| s.pos[0] = [3, 0]),
        "an agent inside the wall",
    );
    rejected(
        &c,
        &with(&|s| s.pos[0] = [3, 4]),
        "an agent past a closed door",
    );
    rejected(
        &c,
        &with(&|s| s.pos[0] = [5, 0]),
        "an agent past a closed door",
    );
    rejected(&c, &with(&|s| s.pos[0] = [0, 0]), "an agent on the key");
    rejected(
        &c,
        &with(&|s| s.finished[0] = Bool8::TRUE),
        "a finished flag that disagrees with the goal",
    );
    rejected(
        &c,
        &with(&|s| {
            s.pos[0] = [6, 6];
            s.door_open = Bool8::TRUE;
            s.key_holder = 0;
        }),
        "a finished flag that disagrees with the goal",
    );
    rejected(
        &c,
        &with(&|s| s.pos[1] = [0, 2]),
        "two agents on one cell off the goal",
    );
    let both = with(&|s| {
        s.pos = [[6, 6], [6, 6], [0, 0], [0, 0]];
        s.finished = [Bool8::TRUE, Bool8::TRUE, Bool8::FALSE, Bool8::FALSE];
        s.door_open = Bool8::TRUE;
        s.key_holder = 0;
    });
    assert_eq!(KeyDoor::validate(&c, &both), Ok(()), "the goal holds both");
}

#[test]
fn test_keydoor_import_rejects_everyone_finished_as_running_and_a_dead_agent() {
    let c = config(8, 1);
    let mut p = pool(&c, 1);
    let mut done = world(&[[6, 6]]);
    done.finished[0] = Bool8::TRUE;
    done.door_open = Bool8::TRUE;
    done.key_holder = 0;
    assert_eq!(KeyDoor::validate(&c, &done), Ok(()));
    let before = p.state_hash(0).unwrap();
    assert!(invalid(&import(&mut p, done)));
    assert_eq!(p.state_hash(0).unwrap(), before);
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut header = *good.header();
    header.agents[0].alive = Bool8::FALSE;
    let dead = DecodedRecord::from_parts(header, *good.state()).unwrap();
    assert!(invalid(&p.import(&[0], &[dead])));
}

#[test]
fn test_keydoor_import_rejects_an_unused_slot_alive_or_acting() {
    let c = config(8, 1);
    let mut p = pool(&c, 1);
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let before = p.state_hash(0).unwrap();
    for (alive, acting) in [(true, true), (true, false), (false, true)] {
        let mut header = *good.header();
        header.agents[1].alive = Bool8::new(alive);
        header.agents[1].acting = Bool8::new(acting);
        let bad = DecodedRecord::from_parts(header, *good.state()).unwrap();
        assert!(invalid(&p.import(&[0], &[bad])), "{alive} {acting}");
        assert_eq!(p.state_hash(0).unwrap(), before);
    }
}

#[test]
fn test_keydoor_never_reset_record_round_trips_and_refuses_anything_else() {
    let c = config(8, 2);
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    let p = Pool::with_masks(c.clone(), options).unwrap();
    let rec = DecodedRecord::from(p.snapshot(0).unwrap());
    let header = *rec.header();
    let mut q = Pool::with_masks(c, options).unwrap();
    q.import(&[0], &[rec]).unwrap();
    let bad = DecodedRecord::from_parts(header, world(&[[0, 2], [1, 2]])).unwrap();
    assert!(invalid(&q.import(&[0], &[bad])));
}

/// Bit perturbations of reachable states: what `Validate` lets through keeps a
/// legal action for every agent and plays on without panicking.
#[test]
fn test_keydoor_perturbed_imports_stay_total_when_accepted() {
    let mut rng = Lcg(77);
    let (mut no, mut yes) = (0, 0);
    let c = config(8, 3);
    for seed in 0..60 {
        let mut s = state(&pool(&c, seed));
        for _ in 0..40 {
            let k = rng.below(3) as usize;
            s.apply(&c, k, rng.below(5) as i32);
            let mut t = s;
            match rng.below(7) {
                0 => t.pos[rng.below(4) as usize][rng.below(2) as usize] = rng.below(10) as u8,
                1 => t.facing[rng.below(4) as usize] = rng.below(6) as u8,
                2 => t.key[rng.below(2) as usize] = rng.below(10) as u8,
                3 => t.goal[rng.below(2) as usize] = rng.below(10) as u8,
                4 => t.wall_col = rng.below(9) as u8,
                5 => t.door_row = rng.below(9) as u8,
                _ => t.key_holder = rng.below(5) as u8,
            }
            if KeyDoor::validate(&c, &t).is_err() {
                no += 1;
                continue;
            }
            yes += 1;
            for _ in 0..30 {
                for k in 0..3 {
                    assert!(t.is_legal(&c, k, WAIT));
                    t.apply(&c, k, rng.below(5) as i32);
                }
                assert_eq!(KeyDoor::validate(&c, &t), Ok(()), "{t:?}");
            }
        }
    }
    assert!(no > 100 && yes > 100, "{no} {yes}");
}

fn baseline(c: &KeyDoorConfig, s: &KeyDoorState, k: u16) -> i32 {
    let mut discrete = [0];
    let mut writer = ActionWriter::new(&mut discrete, &mut []);
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
    let slot = AgentSlot::new(k, 4).unwrap();
    KeyDoor::act(c, s, slot, &mut rng, &mut writer);
    discrete[0]
}

/// The share of worlds the baseline finishes within a bound, per agent count,
/// pinned; one agent must finish all.
#[test]
fn test_keydoor_baseline_solved_share_is_pinned() {
    let mut solved = [0u32; 4];
    for agents in 1..=4u32 {
        for seed in 0..200 {
            let c = config(8, agents);
            let mut p = pool(&c, seed);
            for _ in 0..400 {
                let s = state(&p);
                let acts: Vec<i32> = (0..agents as u16).map(|k| baseline(&c, &s, k)).collect();
                for (k, &a) in acts.iter().enumerate() {
                    assert!(s.is_legal(&c, k, a), "{s:?} {k} {a}");
                }
                step(&mut p, &acts);
                if p.outputs().episode_terminated()[0].get() {
                    solved[agents as usize - 1] += 1;
                    break;
                }
            }
        }
    }
    assert_eq!(solved, [200, 200, 199, 192], "of 200 each");
}
