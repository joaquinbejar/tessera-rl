// SPDX-License-Identifier: MIT OR Apache-2.0

//! The RTS environment through the engine: starts, legal play, validation,
//! determinism, continuation and team symmetry.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::large_types_passed_by_value,
    reason = "test fixtures"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::Bool8;
use tessera_rl_engine::{
    ActionInput, AutoReset, DEBUG_CHECKS, DecodedRecord, EnvPool, IllegalActionPolicy, Masked,
    PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_rts::env::{
    HEADS_PER_UNIT, MASK_PER_UNIT, N_PRODUCTS, N_TYPES, SIDE_SLOTS, START_DRONES, START_ORE,
    UNIT_SLOTS, WINDOW, WINDOW_CELLS, move_index, side_of_agent,
};
use tessera_rl_rts::units::{Kind, Unit};
use tessera_rl_rts::world::{Dir, Grid, Team, mirror_window};
use tessera_rl_rts::{Rts, RtsConfig, RtsState};

type Pool = EnvPool<Rts, Masked>;

fn cfg(size: u16, max_units: u16) -> RtsConfig {
    RtsConfig {
        size,
        decision_ticks: 4,
        max_units,
        step_limit: 400,
        seat_swap: false,
    }
}

fn pool(c: RtsConfig, n: u32, seed: u64, workers: Option<usize>) -> Pool {
    let options = PoolOptions {
        auto_reset: AutoReset::NextStep,
        illegal_action: IllegalActionPolicy::Noop,
        chunk_size: NonZeroU32::new(1).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed))
    };
    let mut p: Pool = EnvPool::with_masks(c, options).unwrap();
    if let Some(w) = workers {
        p = p
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap();
    }
    p
}

/// Resets every instance once and returns the masks.
fn ready(p: &mut Pool) -> Vec<Bool8> {
    p.reset_all(Reseed::Keep).unwrap().masks().to_vec()
}

/// A seeded generator for tests.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        usize::try_from(self.0 % u64::try_from(n).unwrap()).unwrap()
    }

    /// A uniformly drawn true index of `row`; every row has one.
    fn pick(&mut self, row: &[Bool8]) -> i32 {
        let legal: Vec<usize> = (0..row.len()).filter(|&i| row[i].get()).collect();
        i32::try_from(legal[self.below(legal.len())]).unwrap()
    }
}

/// A legal action row for every agent of every instance, drawn from masks.
fn legal_actions(rng: &mut Rng, masks: &[Bool8], n: usize, max_units: usize) -> Vec<i32> {
    let types = usize::from(N_TYPES);
    let cells = usize::from(WINDOW_CELLS);
    let products = usize::from(N_PRODUCTS);
    let per_agent = max_units * MASK_PER_UNIT;
    let mut out = Vec::new();
    for a in 0..n * 2 {
        let agent = &masks[a * per_agent..(a + 1) * per_agent];
        for u in 0..max_units {
            let block = &agent[u * MASK_PER_UNIT..(u + 1) * MASK_PER_UNIT];
            let t = rng.pick(&block[..types]);
            let tu = usize::try_from(t).unwrap();
            let target = rng.pick(&block[types + tu * cells..types + (tu + 1) * cells]);
            let base = types + types * cells;
            let product = rng.pick(&block[base + tu * products..base + (tu + 1) * products]);
            out.extend([t, target, product]);
        }
    }
    out
}

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

/// The state seen from the other side: units swapped between sides and
/// mirrored, ore swapped, terrain mirrored.
fn mirrored(c: &RtsConfig, s: &RtsState) -> RtsState {
    let g = Grid::new(c.size).unwrap();
    let mut m = *s;
    for i in 0..UNIT_SLOTS {
        let j = (i + SIDE_SLOTS) % UNIT_SLOTS;
        let mut u = s.units[i];
        if u.alive.get() {
            u.cell = g.mirror(u.cell);
        }
        m.units[j] = u;
    }
    m.ore = [s.ore[1], s.ore[0]];
    for cell in 0..g.cells() {
        m.terrain[usize::from(g.mirror(cell))] = s.terrain[usize::from(cell)];
    }
    m
}

fn record(p: &Pool, state: RtsState) -> DecodedRecord<Rts> {
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    DecodedRecord::from_parts(*good.header(), state).unwrap()
}

#[test]
fn test_rts_starts_are_mirror_images_with_their_ore_and_drones() {
    for size in [8, 9, 13, 16, 24, 32] {
        let c = cfg(size, 8);
        for seed in 0..20 {
            let mut p = pool(c.clone(), 1, seed, None);
            ready(&mut p);
            let s = *p.state(0).unwrap();
            assert_eq!(s, mirrored(&c, &s), "size {size} seed {seed}");
            assert_eq!(s.ore, [START_ORE; 2]);
            let alive: Vec<&Unit> = s.units[..SIDE_SLOTS]
                .iter()
                .filter(|u| u.alive.get())
                .collect();
            assert_eq!(alive.len(), 1 + START_DRONES);
            assert_eq!(Kind::from_byte(s.units[0].kind), Some(Kind::Core));
            assert_eq!(Kind::from_byte(s.units[1].kind), Some(Kind::Drone));
            assert_eq!(s.units[0].generation, 1);
        }
    }
}

#[test]
fn test_rts_legal_play_is_accepted_and_every_state_validates() {
    if DEBUG_CHECKS {
        return;
    }
    for (size, units) in [(8, 4), (16, 12)] {
        let c = cfg(size, units);
        let mut p = pool(c.clone(), 2, 11, None);
        let mut rng = Rng(0x5eed_u64 + u64::from(size));
        let mut masks = ready(&mut p);
        for _ in 0..300 {
            let a = legal_actions(&mut rng, &masks, 2, usize::from(units));
            assert_eq!(p.check_actions(input(&a)), Ok(()));
            masks = p.step(input(&a)).unwrap().masks().to_vec();
            for i in 0..2 {
                let s = *p.state(i).unwrap();
                p.import(&[i], &[record(&p, s)]).unwrap();
            }
        }
    }
}

fn trajectory(c: &RtsConfig, workers: Option<usize>, steps: usize) -> Vec<u64> {
    let mut p = pool(c.clone(), 3, 5, workers);
    let mut rng = Rng(0x1234_5678);
    let mut masks = ready(&mut p);
    let mut hashes = Vec::new();
    for _ in 0..steps {
        let a = legal_actions(&mut rng, &masks, 3, usize::from(c.max_units));
        masks = p.step(input(&a)).unwrap().masks().to_vec();
        hashes.extend((0..3).map(|i| p.state_hash(i).unwrap().get()));
    }
    hashes
}

#[test]
fn test_rts_trajectory_is_identical_across_workers() {
    let c = cfg(12, 6);
    let inline = trajectory(&c, None, 120);
    assert_eq!(inline, trajectory(&c, Some(1), 120));
    assert_eq!(inline, trajectory(&c, Some(3), 120));
}

#[test]
fn test_rts_continues_through_snapshot_and_restore() {
    let c = cfg(12, 6);
    let mut p = pool(c.clone(), 1, 9, None);
    let mut rng = Rng(77);
    let mut masks = ready(&mut p);
    let mut rows = Vec::new();
    for _ in 0..80 {
        let a = legal_actions(&mut rng, &masks, 1, 6);
        rows.push(a.clone());
        masks = p.step(input(&a)).unwrap().masks().to_vec();
    }
    let whole = p.state_hash(0).unwrap();
    let mut q = pool(c, 1, 9, None);
    ready(&mut q);
    for a in &rows[..30] {
        q.step(input(a)).unwrap();
    }
    let mid = q.snapshot(0).unwrap();
    let mut r = pool(cfg(12, 6), 1, 9, None);
    ready(&mut r);
    r.restore(0, &mid).unwrap();
    for a in &rows[30..] {
        r.step(input(a)).unwrap();
    }
    assert_eq!(r.state_hash(0).unwrap(), whole);
}

#[test]
fn test_rts_side_one_window_index_acts_on_the_mirror_window_cell() {
    let g = Grid::new(16).unwrap();
    let from = g.index(7, 9).unwrap();
    for d in Dir::order(Team::Zero) {
        let zero = move_index(Team::Zero, d);
        let one = move_index(Team::One, d);
        // The same map step is the mirrored index for side One.
        assert_eq!(one, mirror_window(zero, WINDOW));
        assert!(g.step(from, d).is_some());
    }
    // Side Zero's North is the window cell straight above the centre.
    let half = WINDOW / 2;
    assert_eq!(
        move_index(Team::Zero, Dir::North),
        (half - 1) * WINDOW + half
    );
}

#[test]
fn test_rts_team_order_for_side_zero_is_north_east_south_west() {
    assert_eq!(
        Dir::order(Team::Zero),
        [Dir::North, Dir::East, Dir::South, Dir::West]
    );
}

/// One step from `s` with `a0` for agent 0 and `a1` for agent 1.
fn step_from(c: &RtsConfig, s: RtsState, a0: &[i32], a1: &[i32]) -> (RtsState, Vec<f32>) {
    let mut p = pool(c.clone(), 1, 3, None);
    ready(&mut p);
    p.import(&[0], &[record(&p, s)]).unwrap();
    let a = [a0, a1].concat();
    let rewards = p.step(input(&a)).unwrap().rewards().to_vec();
    (*p.state(0).unwrap(), rewards)
}

#[test]
fn test_rts_team_symmetry_under_identical_indices() {
    if DEBUG_CHECKS {
        return;
    }
    let c = cfg(12, 6);
    let per = 6 * HEADS_PER_UNIT;
    let mut p = pool(c.clone(), 1, 21, None);
    let mut rng = Rng(4242);
    let mut masks = ready(&mut p);
    for k in 0..150 {
        let s = *p.state(0).unwrap();
        let a = legal_actions(&mut rng, &masks, 1, 6);
        let (a0, a1) = (&a[..per], &a[per..]);
        let (next, r) = step_from(&c, s, a0, a1);
        let (mnext, mr) = step_from(&c, mirrored(&c, &s), a1, a0);
        assert_eq!(mirrored(&c, &next), mnext, "step {k}");
        assert_eq!((r[0], r[1]), (mr[1], mr[0]), "step {k}");
        masks = p.step(input(&a)).unwrap().masks().to_vec();
    }
}

#[test]
fn test_rts_seat_swap_changes_only_naming() {
    let c = cfg(12, 6);
    let swapped = RtsConfig {
        seat_swap: true,
        ..c.clone()
    };
    assert_eq!(side_of_agent(&c, 0), Team::Zero);
    assert_eq!(side_of_agent(&swapped, 0), Team::One);
    let mut p = pool(c.clone(), 1, 8, None);
    let mut q = pool(swapped, 1, 8, None);
    let mut rng = Rng(99);
    let per = 6 * HEADS_PER_UNIT;
    let mut masks = ready(&mut p);
    let mut qmasks = ready(&mut q);
    for _ in 0..60 {
        let half = qmasks.len() / 2;
        assert_eq!(masks[..half], qmasks[half..]);
        let a = legal_actions(&mut rng, &masks, 1, 6);
        let b = [&a[per..], &a[..per]].concat();
        masks = p.step(input(&a)).unwrap().masks().to_vec();
        qmasks = q.step(input(&b)).unwrap().masks().to_vec();
        assert_eq!(p.state(0).unwrap(), q.state(0).unwrap());
    }
}

#[test]
fn test_rts_record_with_two_units_on_one_cell_is_refused() {
    let c = cfg(12, 6);
    let mut p = pool(c, 1, 2, None);
    ready(&mut p);
    let mut s = *p.state(0).unwrap();
    s.units[1].cell = s.units[0].cell;
    let mut q = pool(cfg(12, 6), 1, 2, None);
    ready(&mut q);
    assert!(q.import(&[0], &[record(&p, s)]).is_err());
}
