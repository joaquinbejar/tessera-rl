// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-built states for the order semantics of `rts-design.md` r6 section
//! 13: attacks, blocked steps, the movement counter, deposit, the end in a
//! tick, contested and joint spending, and placement.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::large_types_passed_by_value,
    reason = "test fixtures"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::MaskWriter;
use tessera_rl_core::{ActionMask, AgentSlot, Bool8, Reserved};
use tessera_rl_engine::{
    ActionInput, AutoReset, DEBUG_CHECKS, DecodedRecord, EngineError, EnvPool, IllegalActionPolicy,
    Masked, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_rts::env::{
    HEADS_PER_UNIT, MASK_PER_UNIT, N_TYPES, SIDE_SLOTS, WINDOW, WINDOW_CELLS, order_type,
};
use tessera_rl_rts::units::{Kind, Unit};
use tessera_rl_rts::world::{Grid, Team, Terrain};
use tessera_rl_rts::{Rts, RtsConfig, RtsState};

type Pool = EnvPool<Rts, Masked>;

const SIZE: u16 = 12;

fn config(units: u16, ticks: u16) -> RtsConfig {
    RtsConfig {
        size: SIZE,
        decision_ticks: ticks,
        max_units: units,
        step_limit: 400,
        seat_swap: false,
    }
}

fn grid() -> Grid {
    Grid::new(SIZE).unwrap()
}

fn at(x: u16, y: u16) -> u16 {
    grid().index(x, y).unwrap()
}

/// A unit of `kind` at `(x, y)`, complete and at full hit points.
fn unit(kind: Kind, x: u16, y: u16) -> Unit {
    Unit {
        cell: at(x, y),
        kind: kind as u8,
        hp: kind.stats().hp,
        alive: Bool8::TRUE,
        finished: Bool8::TRUE,
        carry: 0,
        counter: 0,
        progress: 0,
        producing: 0,
        pad: Reserved::default(),
        generation: 1,
    }
}

/// An open map with the given units: side Zero's from slot 0, side One's
/// from [`SIDE_SLOTS`]. Each side gets a far Core unless it has one.
fn world(zero: &[Unit], one: &[Unit], ore: [u32; 2]) -> RtsState {
    let mut s = RtsState {
        ore,
        decisions: 0,
        ticks: 0,
        ..RtsState::default()
    };
    for (k, u) in zero.iter().enumerate() {
        s.units[k] = *u;
    }
    for (k, u) in one.iter().enumerate() {
        s.units[SIDE_SLOTS + k] = *u;
    }
    s
}

/// `s` with each side's explored bits set to what it sees now, as reset
/// would leave them; a hand-built state is otherwise refused (visible
/// cells must be explored).
fn with_explored(cfg: &RtsConfig, mut s: RtsState) -> RtsState {
    for team in [Team::Zero, Team::One] {
        let seen = tessera_rl_rts::fog::visible(cfg, &s, team);
        for (bits, &v) in s.explored.iter_mut().zip(seen.iter()) {
            if v {
                *bits |= tessera_rl_rts::fog::side_bit(team);
            }
        }
    }
    s
}

struct Game {
    pool: Pool,
    cfg: RtsConfig,
    /// Whether the last step truncated the episode.
    truncated: bool,
    /// Every agent's observation bytes after the last step.
    obs: Vec<u8>,
}

impl Game {
    fn new(cfg: RtsConfig, s: RtsState) -> Self {
        Self::with(cfg, s, IllegalActionPolicy::Noop, 1)
    }

    fn with(cfg: RtsConfig, s: RtsState, policy: IllegalActionPolicy, seed: u64) -> Self {
        let options = PoolOptions {
            auto_reset: AutoReset::NextStep,
            illegal_action: policy,
            chunk_size: NonZeroU32::new(1).unwrap(),
            ..PoolOptions::new(NonZeroU32::new(1).unwrap(), Seed::new(seed))
        };
        let mut pool: Pool = EnvPool::with_masks(cfg.clone(), options).unwrap();
        pool.reset_all(Reseed::Keep).unwrap();
        let header = *DecodedRecord::from(pool.snapshot(0).unwrap()).header();
        let s = with_explored(&cfg, s);
        pool.import(&[0], &[DecodedRecord::from_parts(header, s).unwrap()])
            .unwrap();
        Self {
            pool,
            cfg,
            truncated: false,
            obs: Vec::new(),
        }
    }

    /// Agent `agent`'s observation bytes after the last step.
    fn observation(&self, agent: usize) -> Vec<u8> {
        let per = self.obs.len() / 2;
        self.obs[agent * per..(agent + 1) * per].to_vec()
    }

    fn state(&self) -> RtsState {
        *self.pool.state(0).unwrap()
    }

    fn illegal(&self) -> [u32; 2] {
        let r = DecodedRecord::from(self.pool.snapshot(0).unwrap());
        [
            r.header().agents[0].illegal_actions,
            r.header().agents[1].illegal_actions,
        ]
    }

    /// One step with `orders`: `(side, slot in side, type, target cell or
    /// None for window 0, product)`.
    fn step(&mut self, orders: &[(usize, usize, i32, Option<u16>, i32)]) -> Vec<f32> {
        let per = usize::from(self.cfg.max_units) * HEADS_PER_UNIT;
        let mut a = vec![0; 2 * per];
        let s = self.state();
        for &(side, k, t, target, product) in orders {
            let from = s.units[side * SIDE_SLOTS + k].cell;
            let index = target.map_or(0, |c| window(side, from, c));
            let base = side * per + k * HEADS_PER_UNIT;
            a[base] = t;
            a[base + 1] = i32::from(index);
            a[base + 2] = product;
        }
        let out = self
            .pool
            .step(ActionInput {
                discrete: &a,
                continuous: &[],
            })
            .unwrap();
        self.truncated = out.episode_truncated()[0].get();
        self.obs = out.observation(0).unwrap().to_vec();
        out.rewards().to_vec()
    }
}

/// The window index, in `side`'s frame, of `to` seen from `from`.
fn window(side: usize, from: u16, to: u16) -> u16 {
    let g = grid();
    let ((fx, fy), (tx, ty)) = (g.coords(from), g.coords(to));
    let (mut dx, mut dy) = (i32::from(tx) - i32::from(fx), i32::from(ty) - i32::from(fy));
    if side == 1 {
        (dx, dy) = (-dx, -dy);
    }
    let half = i32::from(WINDOW / 2);
    u16::try_from((dy + half) * i32::from(WINDOW) + dx + half).unwrap()
}

const DRONE: i32 = 0;
const STRIKER: i32 = 1;
const FOUNDRY: i32 = 3;

#[test]
fn test_scenario_range_is_at_most_sight_for_every_kind() {
    for k in Kind::ALL {
        assert!(k.stats().range <= k.stats().sight, "{k:?}");
    }
}

#[test]
fn test_scenario_two_strikers_kill_each_other_in_one_tick() {
    let mut a = unit(Kind::Striker, 5, 5);
    let mut b = unit(Kind::Striker, 6, 5);
    a.hp = 2;
    b.hp = 2;
    let s = world(
        &[unit(Kind::Core, 1, 1), a],
        &[unit(Kind::Core, 10, 10), b],
        [0, 0],
    );
    let mut g = Game::new(config(4, 4), s);
    g.step(&[
        (0, 1, order_type::ATTACK, Some(at(6, 5)), 0),
        (1, 1, order_type::ATTACK, Some(at(5, 5)), 0),
    ]);
    let s = g.state();
    assert!(!s.units[1].alive.get());
    assert!(!s.units[SIDE_SLOTS + 1].alive.get());
    assert_eq!(s.units[1].generation, 1);
}

#[test]
fn test_scenario_second_step_into_a_wall_is_blocked_not_illegal() {
    let mut s = world(
        &[unit(Kind::Core, 1, 1), unit(Kind::Drone, 5, 5)],
        &[unit(Kind::Core, 10, 10)],
        [0, 0],
    );
    s.terrain[usize::from(at(7, 5))] = Terrain::Wall as u8;
    // Eight ticks at speed 4: a second step, into the wall.
    let mut g = Game::new(config(4, 8), s);
    g.step(&[(0, 1, order_type::MOVE, Some(at(6, 5)), 0)]);
    let s = g.state();
    assert_eq!(s.units[1].cell, at(6, 5));
    assert_eq!(s.units[1].counter, 0);
    assert_eq!(g.illegal(), [0, 0]);
}

#[test]
fn test_scenario_counter_carries_across_moves_and_resets_otherwise() {
    let lancer = unit(Kind::Lancer, 5, 5);
    let base = world(
        &[unit(Kind::Core, 1, 1), lancer],
        &[unit(Kind::Core, 10, 10)],
        [0, 0],
    );
    // Two ticks per decision at speed 6: three moves make one step.
    let mut g = Game::new(config(4, 2), base);
    for _ in 0..3 {
        let from = g.state().units[1].cell;
        let (x, y) = grid().coords(from);
        g.step(&[(0, 1, order_type::MOVE, Some(at(x + 1, y)), 0)]);
    }
    assert_eq!(g.state().units[1].cell, at(6, 5));
    // With a no-op in the middle the counter restarts: no step.
    let mut h = Game::new(config(4, 2), base);
    h.step(&[(0, 1, order_type::MOVE, Some(at(6, 5)), 0)]);
    h.step(&[]);
    h.step(&[(0, 1, order_type::MOVE, Some(at(6, 5)), 0)]);
    assert_eq!(h.state().units[1].cell, at(5, 5));
}

#[test]
fn test_scenario_deposit_to_a_core_destroyed_in_tick_1_does_nothing() {
    let mut core_a = unit(Kind::Core, 6, 6);
    core_a.hp = 2;
    let mut drone = unit(Kind::Drone, 6, 7);
    drone.carry = 1;
    let s = world(
        &[unit(Kind::Core, 1, 1), unit(Kind::Striker, 5, 6)],
        &[unit(Kind::Core, 10, 10), core_a, drone],
        [0, 3],
    );
    let mut g = Game::new(config(4, 4), s);
    let r = g.step(&[
        (0, 1, order_type::ATTACK, Some(at(6, 6)), 0),
        (1, 2, order_type::DEPOSIT, Some(at(6, 6)), 0),
    ]);
    let s = g.state();
    assert!(!s.units[SIDE_SLOTS + 1].alive.get());
    assert_eq!(s.units[SIDE_SLOTS + 2].carry, 1);
    assert_eq!(s.ore[1], 3);
    assert_eq!(r, [0.0, 0.0]);
}

#[test]
fn test_scenario_game_ending_in_tick_1_skips_the_rest_of_the_decision() {
    let mut core = unit(Kind::Core, 6, 6);
    core.hp = 2;
    let mut s = world(
        &[
            unit(Kind::Core, 1, 1),
            unit(Kind::Striker, 5, 6),
            unit(Kind::Drone, 3, 3),
        ],
        &[core],
        [0, 0],
    );
    s.terrain[usize::from(at(4, 3))] = Terrain::Ore as u8;
    let mut g = Game::new(config(4, 4), s);
    let before = g.state().ticks;
    let r = g.step(&[
        (0, 1, order_type::ATTACK, Some(at(6, 6)), 0),
        (0, 2, order_type::HARVEST, Some(at(4, 3)), 0),
    ]);
    assert_eq!(r, [1.0, -1.0]);
    let s = g.state();
    // The episode reset happens on the next step (NextStep), so the
    // terminal state is still there: one tick ran, no harvest.
    assert_eq!(s.ticks, before + 1);
    assert_eq!(s.units[2].carry, 0);
}

#[test]
fn test_scenario_two_own_drones_building_on_one_cell_are_both_refused() {
    let s = world(
        &[
            unit(Kind::Core, 1, 1),
            unit(Kind::Drone, 5, 5),
            unit(Kind::Drone, 7, 5),
        ],
        &[unit(Kind::Core, 10, 10)],
        [20, 0],
    );
    let mut g = Game::new(config(6, 4), s);
    g.step(&[
        (0, 1, order_type::BUILD, Some(at(6, 5)), FOUNDRY),
        (0, 2, order_type::BUILD, Some(at(6, 5)), FOUNDRY),
    ]);
    let s = g.state();
    assert_eq!(s.ore[0], 20);
    assert!(
        s.units[..SIDE_SLOTS]
            .iter()
            .all(|u| u.kind != Kind::Foundry as u8)
    );
    assert_eq!(g.illegal(), [0, 0]);
}

fn two_builds(ore: u32, units: u16) -> (RtsState, Game) {
    let s = world(
        &[
            unit(Kind::Core, 1, 1),
            unit(Kind::Drone, 5, 5),
            unit(Kind::Drone, 5, 8),
        ],
        &[unit(Kind::Core, 10, 10)],
        [ore, 0],
    );
    let mut g = Game::new(config(units, 4), s);
    g.step(&[
        (0, 1, order_type::BUILD, Some(at(6, 5)), FOUNDRY),
        (0, 2, order_type::BUILD, Some(at(6, 8)), FOUNDRY),
    ]);
    (g.state(), g)
}

#[test]
fn test_scenario_joint_ore_overspend_is_refused_whole() {
    let cost = u32::from(Kind::Foundry.stats().cost);
    let (s, g) = two_builds(2 * cost - 1, 6);
    assert_eq!(s.ore[0], 2 * cost - 1);
    assert_eq!(
        s.units[..SIDE_SLOTS]
            .iter()
            .filter(|u| u.kind == Kind::Foundry as u8)
            .count(),
        0
    );
    assert_eq!(g.illegal(), [0, 0]);
    let (s, _) = two_builds(2 * cost, 6);
    assert_eq!(s.ore[0], 0);
    assert_eq!(
        s.units[..SIDE_SLOTS]
            .iter()
            .filter(|u| u.kind == Kind::Foundry as u8)
            .count(),
        2
    );
}

#[test]
fn test_scenario_joint_slot_overspend_is_refused_whole() {
    // Four slots: Core and two Drones leave one free for two builds.
    let (s, g) = two_builds(100, 4);
    assert_eq!(s.ore[0], 100);
    assert_eq!(
        s.units[..SIDE_SLOTS]
            .iter()
            .filter(|u| u.kind == Kind::Foundry as u8)
            .count(),
        0
    );
    assert_eq!(g.illegal(), [0, 0]);
}

/// A Core and a Foundry whose only free cells are one shared cell and, for
/// the Core, one more; both complete in the decision's first tick.
fn contest(core_first: bool) -> RtsState {
    let mut core = unit(Kind::Core, 4, 4);
    core.producing = Kind::Drone as u8;
    core.progress = u16::from(Kind::Drone.stats().build_ticks) - 1;
    let mut foundry = unit(Kind::Foundry, 6, 4);
    foundry.producing = Kind::Striker as u8;
    foundry.progress = u16::from(Kind::Striker.stats().build_ticks) - 1;
    let zero = if core_first {
        [core, foundry]
    } else {
        [foundry, core]
    };
    let mut s = world(&zero, &[unit(Kind::Core, 10, 10)], [0, 0]);
    for (x, y) in [(4, 3), (3, 4), (6, 3), (7, 4), (6, 5)] {
        s.terrain[usize::from(at(x, y))] = Terrain::Wall as u8;
    }
    s
}

#[test]
fn test_scenario_placement_contest_has_no_winner_and_does_not_depend_on_slots() {
    for core_first in [true, false] {
        let mut g = Game::new(config(6, 1), contest(core_first));
        // Tick 1: both pick (5, 4); refused to both; the Core takes its next
        // cell, (4, 5); the Foundry has none and waits.
        g.step(&[]);
        let s = g.state();
        let kinds_at = |x, y| {
            s.units[..SIDE_SLOTS]
                .iter()
                .find(|u| u.alive.get() && u.cell == at(x, y))
                .map(|u| Kind::from_byte(u.kind).unwrap())
        };
        assert_eq!(kinds_at(4, 5), Some(Kind::Drone));
        assert_eq!(kinds_at(5, 4), None);
        // Tick 2: the Foundry alone picks (5, 4).
        g.step(&[]);
        let s = g.state();
        let striker = s.units[..SIDE_SLOTS]
            .iter()
            .find(|u| u.alive.get() && u.kind == Kind::Striker as u8)
            .unwrap();
        assert_eq!(striker.cell, at(5, 4));
    }
}

#[test]
fn test_scenario_a_waiting_producer_is_busy() {
    let mut s = contest(true);
    // Wall the Core's last cell too: both producers wait.
    s.terrain[usize::from(at(4, 5))] = Terrain::Wall as u8;
    s.terrain[usize::from(at(5, 4))] = Terrain::Wall as u8;
    s.ore = [50, 0];
    let mut g = Game::new(config(6, 1), s);
    g.step(&[]);
    let s = g.state();
    assert_ne!(s.units[0].producing, 0);
    // A produce order to the waiting Core is illegal: one fallback.
    if !DEBUG_CHECKS {
        g.step(&[(0, 0, order_type::BUILD, None, DRONE)]);
        assert_eq!(g.illegal(), [1, 0]);
    }
}

#[test]
fn test_scenario_foundry_build_then_striker_production_spends_and_places() {
    let s = world(
        &[unit(Kind::Core, 1, 1), unit(Kind::Drone, 5, 5)],
        &[unit(Kind::Core, 10, 10)],
        [20, 0],
    );
    let mut g = Game::new(config(6, 4), s);
    // 16 build ticks; the placing tick adds none: 3 + 4 + 4 + 4 + 1.
    for _ in 0..5 {
        g.step(&[(0, 1, order_type::BUILD, Some(at(6, 5)), FOUNDRY)]);
    }
    let s = g.state();
    let f = s.units[..SIDE_SLOTS]
        .iter()
        .position(|u| u.kind == Kind::Foundry as u8)
        .unwrap();
    assert!(s.units[f].finished.get());
    assert_eq!(s.ore[0], 20 - u32::from(Kind::Foundry.stats().cost));
    g.step(&[(0, f, order_type::BUILD, None, STRIKER)]);
    for _ in 0..3 {
        g.step(&[]);
    }
    let s = g.state();
    assert!(
        s.units[..SIDE_SLOTS]
            .iter()
            .any(|u| u.alive.get() && u.kind == Kind::Striker as u8)
    );
    assert_eq!(g.illegal(), [0, 0]);
}

/// `s` seen from the other side: units swapped between sides and
/// mirrored, ore swapped, terrain and explored mirrored (explored bits
/// swapped with the sides).
fn mirrored(s: &RtsState) -> RtsState {
    let g = grid();
    let mut m = *s;
    for i in 0..2 * SIDE_SLOTS {
        let mut u = s.units[i];
        if u.alive.get() {
            u.cell = g.mirror(u.cell);
        }
        m.units[(i + SIDE_SLOTS) % (2 * SIDE_SLOTS)] = u;
    }
    m.ore = [s.ore[1], s.ore[0]];
    for cell in 0..g.cells() {
        m.terrain[usize::from(g.mirror(cell))] = s.terrain[usize::from(cell)];
        let e = s.explored[usize::from(cell)];
        m.explored[usize::from(g.mirror(cell))] = ((e & 1) << 1) | ((e >> 1) & 1);
    }
    m
}

type Orders = Vec<(usize, usize, i32, Option<u16>, i32)>;

/// Steps `s` and its mirror `steps` times with the same orders, each side's
/// given to the other with mirrored target cells, so that the window
/// indices are identical; the results must be mirrors.
fn check_symmetry(cfg: &RtsConfig, s: RtsState, orders: &Orders, steps: usize) {
    let g = grid();
    let swapped: Orders = orders
        .iter()
        .map(|&(side, k, t, c, p)| (1 - side, k, t, c.map(|c| g.mirror(c)), p))
        .collect();
    let mut a = Game::new(cfg.clone(), s);
    let mut b = Game::new(cfg.clone(), mirrored(&s));
    for k in 0..steps {
        let ra = a.step(if k == 0 { orders } else { &[] });
        let rb = b.step(if k == 0 { &swapped } else { &[] });
        assert_eq!(mirrored(&a.state()), b.state(), "step {k}");
        assert_eq!((ra[0], ra[1]), (rb[1], rb[0]), "step {k}");
        assert_eq!(a.illegal(), [b.illegal()[1], b.illegal()[0]], "step {k}");
    }
}

#[test]
fn test_scenario_team_symmetry_with_fighting_building_and_placement() {
    if DEBUG_CHECKS {
        return;
    }
    // Attacks and a kill on both sides.
    let mut b = unit(Kind::Striker, 6, 5);
    b.hp = 2;
    let fight = world(
        &[
            unit(Kind::Core, 1, 1),
            unit(Kind::Striker, 5, 5),
            unit(Kind::Lancer, 5, 7),
        ],
        &[unit(Kind::Core, 10, 10), b],
        [0, 0],
    );
    check_symmetry(
        &config(4, 4),
        fight,
        &vec![
            (0, 1, order_type::ATTACK, Some(at(6, 5)), 0),
            (0, 2, order_type::ATTACK, Some(at(6, 5)), 0),
            (1, 1, order_type::ATTACK, Some(at(5, 5)), 0),
        ],
        3,
    );
    // A contested placement in each producer's frame.
    check_symmetry(&config(6, 1), contest(true), &vec![], 3);
    // A build and a production in the same decision.
    let econ = world(
        &[unit(Kind::Core, 1, 1), unit(Kind::Drone, 5, 5)],
        &[unit(Kind::Core, 10, 10), unit(Kind::Drone, 6, 6)],
        [20, 20],
    );
    check_symmetry(
        &config(6, 4),
        econ,
        &vec![
            (0, 1, order_type::BUILD, Some(at(6, 5)), FOUNDRY),
            (0, 0, order_type::BUILD, None, DRONE),
            (1, 1, order_type::MOVE, Some(at(6, 5)), 0),
        ],
        6,
    );
}

#[test]
fn test_scenario_a_production_in_progress_reserves_its_slot() {
    // The build is illegal by mask, which `debug-checks` turns into an
    // error.
    if DEBUG_CHECKS {
        return;
    }
    // Three slots: a producing Core, a Drone and one empty slot, which the
    // Core's production holds, so a new build has no slot.
    let mut core = unit(Kind::Core, 1, 1);
    core.producing = Kind::Drone as u8;
    let s = world(
        &[core, unit(Kind::Drone, 5, 5)],
        &[unit(Kind::Core, 10, 10)],
        [50, 0],
    );
    let mut g = Game::new(config(3, 1), s);
    g.step(&[(0, 1, order_type::BUILD, Some(at(6, 5)), FOUNDRY)]);
    let s = g.state();
    assert!(
        s.units[..SIDE_SLOTS]
            .iter()
            .all(|u| u.kind != Kind::Foundry as u8)
    );
    assert_eq!(s.ore[0], 50);
    // Refused by the mask: the order was illegal and fell back.
    assert_eq!(g.illegal(), [1, 0]);
}

// Rules of section 13 the first round left untested (TR, 68555).

#[test]
fn test_scenario_attack_hits_once_per_decision() {
    let s = world(
        &[unit(Kind::Core, 1, 1), unit(Kind::Striker, 5, 5)],
        &[unit(Kind::Core, 6, 5)],
        [0, 0],
    );
    let mut g = Game::new(config(4, 4), s);
    g.step(&[(0, 1, order_type::ATTACK, Some(at(6, 5)), 0)]);
    let core = Kind::Core.stats().hp - Kind::Striker.stats().damage;
    assert_eq!(g.state().units[SIDE_SLOTS].hp, core);
}

/// Side Zero's mask block of its slot `k`.
fn mask_block(cfg: &RtsConfig, s: &RtsState, k: usize) -> Vec<bool> {
    let per = usize::from(cfg.max_units) * MASK_PER_UNIT;
    let mut buf = vec![Bool8::FALSE; per];
    <Rts as ActionMask>::action_mask(
        cfg,
        s,
        AgentSlot::new(0, 2).unwrap(),
        &mut MaskWriter::new(&mut buf),
    );
    buf[k * MASK_PER_UNIT..(k + 1) * MASK_PER_UNIT]
        .iter()
        .map(|b| b.get())
        .collect()
}

#[test]
fn test_scenario_attack_mask_is_range_and_never_an_own_unit() {
    let c = config(4, 4);
    let s = world(
        &[
            unit(Kind::Core, 1, 1),
            unit(Kind::Striker, 5, 5),
            unit(Kind::Drone, 5, 6),
        ],
        &[unit(Kind::Core, 10, 10)],
        [0, 0],
    );
    let block = mask_block(&c, &s, 1);
    let cell = |x, y| {
        usize::from(N_TYPES)
            + 2 * usize::from(WINDOW_CELLS)
            + usize::from(window(0, at(5, 5), at(x, y)))
    };
    // Range 1, Chebyshev: diagonals in, distance 2 out.
    assert!(block[cell(6, 6)]);
    assert!(block[cell(6, 5)]);
    assert!(!block[cell(7, 5)]);
    // An own unit is never an attack target.
    assert!(!block[cell(5, 6)]);
}

#[test]
fn test_scenario_counter_resets_on_a_no_op_between_moves() {
    // Two ticks per decision at speed 6: move, no-op, move, move would step
    // on its sixth tick without the reset; with it, the count is 4.
    let lancer = unit(Kind::Lancer, 5, 5);
    let s = world(
        &[unit(Kind::Core, 1, 1), lancer],
        &[unit(Kind::Core, 10, 10)],
        [0, 0],
    );
    let mut g = Game::new(config(4, 2), s);
    g.step(&[(0, 1, order_type::MOVE, Some(at(6, 5)), 0)]);
    g.step(&[]);
    g.step(&[(0, 1, order_type::MOVE, Some(at(6, 5)), 0)]);
    g.step(&[(0, 1, order_type::MOVE, Some(at(6, 5)), 0)]);
    let u = g.state().units[1];
    assert_eq!(u.cell, at(5, 5));
    assert_eq!(u.counter, 4);
}

#[test]
fn test_scenario_a_unit_killed_in_tick_1_neither_moves_builds_nor_places() {
    // A Drone killed in tick 1 does not claim the cell another Drone wants.
    let mut dying = unit(Kind::Drone, 6, 5);
    dying.counter = 3;
    let mut other = unit(Kind::Drone, 7, 6);
    other.counter = 3;
    let s = world(
        &[unit(Kind::Core, 1, 1), unit(Kind::Striker, 5, 5)],
        &[unit(Kind::Core, 10, 10), dying, other],
        [0, 0],
    );
    let mut g = Game::new(config(4, 1), s);
    g.step(&[
        (0, 1, order_type::ATTACK, Some(at(6, 5)), 0),
        (1, 1, order_type::MOVE, Some(at(7, 5)), 0),
        (1, 2, order_type::MOVE, Some(at(7, 5)), 0),
    ]);
    let s = g.state();
    assert!(!s.units[SIDE_SLOTS + 1].alive.get());
    assert_eq!(s.units[SIDE_SLOTS + 2].cell, at(7, 5));

    // A dying Drone adds no build progress; a dying Foundry places nothing.
    let mut site = unit(Kind::Foundry, 6, 6);
    site.finished = Bool8::FALSE;
    site.progress = 5;
    let mut foundry = unit(Kind::Foundry, 3, 7);
    foundry.hp = 1;
    foundry.producing = Kind::Striker as u8;
    foundry.progress = u16::from(Kind::Striker.stats().build_ticks) - 1;
    let s = world(
        &[
            unit(Kind::Core, 1, 1),
            unit(Kind::Striker, 6, 4),
            unit(Kind::Lancer, 3, 5),
        ],
        &[
            unit(Kind::Core, 10, 10),
            unit(Kind::Drone, 6, 5),
            site,
            foundry,
        ],
        [0, 0],
    );
    let mut g = Game::new(config(5, 1), s);
    g.step(&[
        (0, 1, order_type::ATTACK, Some(at(6, 5)), 0),
        (0, 2, order_type::ATTACK, Some(at(3, 7)), 0),
        (1, 1, order_type::BUILD, Some(at(6, 6)), FOUNDRY),
    ]);
    let s = g.state();
    assert_eq!(s.units[SIDE_SLOTS + 2].progress, 5);
    assert!(!s.units[SIDE_SLOTS + 3].alive.get());
    assert!(
        s.units[SIDE_SLOTS..]
            .iter()
            .all(|u| u.kind != Kind::Striker as u8)
    );
}

#[test]
fn test_scenario_truncation_comes_at_the_step_limit_exactly() {
    let s = world(
        &[unit(Kind::Core, 1, 1)],
        &[unit(Kind::Core, 10, 10)],
        [0, 0],
    );
    let mut g = Game::new(
        RtsConfig {
            step_limit: 3,
            ..config(4, 1)
        },
        s,
    );
    g.step(&[]);
    g.step(&[]);
    assert!(!g.truncated);
    g.step(&[]);
    assert!(g.truncated);
}

#[test]
fn test_scenario_observation_is_in_the_side_frame() {
    let s = world(
        &[unit(Kind::Core, 1, 1), unit(Kind::Lancer, 3, 7)],
        &[unit(Kind::Core, 10, 10), unit(Kind::Striker, 8, 2)],
        [0, 0],
    );
    let mut a = Game::new(config(4, 1), s);
    let mut b = Game::new(config(4, 1), mirrored(&s));
    a.step(&[]);
    b.step(&[]);
    assert_eq!(a.observation(1), b.observation(0));
    assert_eq!(a.observation(0), b.observation(1));
    assert_ne!(a.observation(0), a.observation(1));
}

fn refused(s: RtsState) {
    let mut g = Game::new(
        config(4, 1),
        world(
            &[unit(Kind::Core, 1, 1)],
            &[unit(Kind::Core, 10, 10)],
            [0, 0],
        ),
    );
    let header = *DecodedRecord::from(g.pool.snapshot(0).unwrap()).header();
    assert!(
        g.pool
            .import(&[0], &[DecodedRecord::from_parts(header, s).unwrap()])
            .is_err()
    );
}

#[test]
fn test_scenario_records_breaking_invariants_are_refused() {
    let base = world(
        &[unit(Kind::Core, 1, 1), unit(Kind::Drone, 4, 4)],
        &[unit(Kind::Core, 10, 10)],
        [0, 0],
    );
    let mut hp = base;
    hp.units[1].hp = Kind::Drone.stats().hp + 1;
    refused(hp);
    let mut wall = base;
    wall.terrain[usize::from(at(4, 4))] = Terrain::Wall as u8;
    refused(wall);
    let mut off = base;
    off.units[1].cell = grid().cells();
    refused(off);
    let mut carry = base;
    carry.units[0].carry = 1;
    refused(carry);
    let mut counter = base;
    counter.units[1].counter = Kind::Drone.stats().speed;
    refused(counter);
    let mut no_core = base;
    no_core.units[SIDE_SLOTS] = Unit::default();
    refused(no_core);
    let mut beyond = base;
    beyond.units[5] = unit(Kind::Drone, 6, 6);
    refused(beyond);
    // Four slots, two of them used, three productions: one cannot place.
    let mut over = base;
    over.units[0].producing = Kind::Drone as u8;
    over.units[2] = unit(Kind::Foundry, 7, 7);
    over.units[2].producing = Kind::Striker as u8;
    over.units[3] = unit(Kind::Foundry, 8, 7);
    over.units[3].producing = Kind::Lancer as u8;
    refused(over);
}

#[test]
fn test_scenario_penalize_counts_per_unit_and_penalizes_once_per_agent() {
    if DEBUG_CHECKS {
        return;
    }
    let s = world(
        &[
            unit(Kind::Core, 1, 1),
            unit(Kind::Drone, 5, 5),
            unit(Kind::Drone, 5, 8),
        ],
        &[unit(Kind::Core, 10, 10)],
        [0, 0],
    );
    let mut g = Game::with(config(4, 1), s, IllegalActionPolicy::Penalize(-0.5), 1);
    // Two units order a build they cannot pay for: illegal by mask.
    let r = g.step(&[
        (0, 1, order_type::BUILD, Some(at(6, 5)), FOUNDRY),
        (0, 2, order_type::BUILD, Some(at(6, 8)), FOUNDRY),
    ]);
    assert_eq!(g.illegal(), [2, 0]);
    assert_eq!(r, [-0.5, 0.0]);
}

#[test]
fn test_scenario_the_step_draws_nothing() {
    let s = world(
        &[
            unit(Kind::Core, 1, 1),
            unit(Kind::Drone, 5, 5),
            unit(Kind::Striker, 6, 6),
        ],
        &[unit(Kind::Core, 10, 10), unit(Kind::Lancer, 7, 7)],
        [9, 9],
    );
    let orders = [
        (0, 1, order_type::BUILD, Some(at(5, 4)), FOUNDRY),
        (0, 2, order_type::ATTACK, Some(at(7, 7)), 0),
        (1, 1, order_type::MOVE, Some(at(7, 8)), 0),
    ];
    let mut a = Game::with(config(4, 4), s, IllegalActionPolicy::Noop, 1);
    let mut b = Game::with(config(4, 4), s, IllegalActionPolicy::Noop, 999);
    for k in 0..4 {
        a.step(if k == 0 { &orders } else { &[] });
        b.step(if k == 0 { &orders } else { &[] });
        assert_eq!(a.state(), b.state(), "step {k}");
    }
}

#[test]
fn test_scenario_permuting_slots_gives_the_same_world_modulo_naming() {
    let units = [
        unit(Kind::Core, 1, 1),
        unit(Kind::Striker, 5, 5),
        unit(Kind::Drone, 6, 5),
        unit(Kind::Lancer, 5, 6),
        unit(Kind::Drone, 3, 3),
    ];
    let enemy = [unit(Kind::Core, 10, 10), unit(Kind::Striker, 4, 4)];
    let orders = |slot: &dyn Fn(usize) -> usize| {
        vec![
            (0, slot(1), order_type::ATTACK, Some(at(4, 4)), 0),
            (0, slot(2), order_type::MOVE, Some(at(6, 4)), 0),
            (0, slot(3), order_type::MOVE, Some(at(6, 6)), 0),
            (0, slot(4), order_type::MOVE, Some(at(4, 3)), 0),
            (1, 1, order_type::ATTACK, Some(at(5, 5)), 0),
        ]
    };
    let perm = [0usize, 3, 1, 4, 2];
    let mut permuted = [Unit::default(); 5];
    for (k, &p) in perm.iter().enumerate() {
        permuted[p] = units[k];
    }
    let mut a = Game::new(config(6, 4), world(&units, &enemy, [0, 0]));
    let mut b = Game::new(config(6, 4), world(&permuted, &enemy, [0, 0]));
    a.step(&orders(&|k| k));
    b.step(&orders(&|k| perm[k]));
    let sorted = |s: RtsState| {
        let mut v: Vec<(u16, u8, u8)> = s
            .units
            .iter()
            .filter(|u| u.alive.get())
            .map(|u| (u.cell, u.kind, u.hp))
            .collect();
        v.sort_unstable();
        v
    };
    assert_eq!(sorted(a.state()), sorted(b.state()));
}

#[test]
fn test_scenario_slot_reuse_takes_the_lowest_slot_with_generation_plus_one() {
    let mut core = unit(Kind::Core, 1, 1);
    core.producing = Kind::Drone as u8;
    core.progress = u16::from(Kind::Drone.stats().build_ticks) - 1;
    let dead = Unit {
        generation: 4,
        ..Unit::default()
    };
    let s = world(
        &[core, dead, unit(Kind::Drone, 5, 5)],
        &[unit(Kind::Core, 10, 10)],
        [0, 0],
    );
    let mut g = Game::new(config(4, 1), s);
    g.step(&[]);
    let u = g.state().units[1];
    assert!(u.alive.get());
    assert_eq!(Kind::from_byte(u.kind), Some(Kind::Drone));
    assert_eq!(u.generation, 5);
}

#[test]
fn test_scenario_mask_budget_fits_128_units_and_refuses_past_the_budget() {
    let full = RtsConfig {
        max_units: 128,
        ..config(4, 4)
    };
    let options = PoolOptions::new(NonZeroU32::new(4).unwrap(), Seed::new(1));
    assert!(EnvPool::<Rts, Masked>::with_masks(full.clone(), options).is_ok());
    let tight = PoolOptions {
        mask_budget_bytes: 1024,
        ..PoolOptions::new(NonZeroU32::new(4).unwrap(), Seed::new(1))
    };
    assert!(matches!(
        EnvPool::<Rts, Masked>::with_masks(full, tight).map(|_| ()),
        Err(EngineError::MaskBudgetExceeded { .. })
    ));
}
