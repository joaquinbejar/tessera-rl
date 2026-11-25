// SPDX-License-Identifier: MIT OR Apache-2.0

//! Survival-v1 rules (STAGE1-r1): map generation and connectivity, start
//! positions, the tie rules (T1 contested target, T3 start-of-step
//! occupancy, swaps), mutual attacks and shared kill counts, forage, drink
//! and starvation, death and respawn, the episode bound, validation, and
//! the slot-permutation tests the review asked for.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]
#![expect(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::large_types_passed_by_value,
    clippy::large_stack_arrays,
    clippy::needless_range_loop,
    reason = "hand-built test states: fixed small coordinates and slots"
)]

use core::num::NonZeroU32;

use tessera_rl_core::Validate;
use tessera_rl_engine::{ActionInput, DecodedRecord, EnvPool, Masked, NoMasks, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;
use tessera_rl_world::survival::{
    ATTACK, FOREST, GRASS, MAX_AGENTS, MAX_HEALTH, STONE, WATER, map,
};
use tessera_rl_world::{Survival, SurvivalConfig, SurvivalState};

fn cfg(agents: u32, size: u32, respawn: u32) -> SurvivalConfig {
    SurvivalConfig {
        agents,
        size,
        respawn,
        ..SurvivalConfig::default()
    }
}

fn pool(c: SurvivalConfig, n: u32, seed: u64) -> EnvPool<Survival, NoMasks> {
    let options = PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed));
    let mut p = EnvPool::new(c, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn state(p: &EnvPool<Survival, NoMasks>) -> SurvivalState {
    p.state(0).unwrap()
}

/// One step of instance 0 with `(move, act)` per agent slot; the rewards.
fn step(p: &mut EnvPool<Survival, NoMasks>, acts: &[(i32, i32)]) -> Vec<f32> {
    let mut a = vec![0i32; MAX_AGENTS * 2];
    for (i, &(m, x)) in acts.iter().enumerate() {
        a[2 * i] = m;
        a[2 * i + 1] = x;
    }
    let out = p
        .step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
    out.rewards()[..MAX_AGENTS].to_vec()
}

/// Replaces instance 0's state with `s` (the header kept, so the alive
/// flags of `s` must match the pool's).
fn put(p: &mut EnvPool<Survival, NoMasks>, s: SurvivalState) {
    let header = *p.snapshot(0).unwrap().header();
    let rec = DecodedRecord::from_parts(header, s).unwrap();
    p.import(&[0], &[rec]).unwrap();
}

/// A blank 32x32 grass map with `agents` agents parked in a row on the
/// bottom edge (x = 0, 1, ...), alive with full stocks, from a reset pool.
fn blank(p: &mut EnvPool<Survival, NoMasks>) -> SurvivalState {
    let mut s = state(p);
    for row in s.tiles.iter_mut().take(32) {
        row[..32].fill(GRASS);
    }
    for row in &mut s.regrow {
        row.fill(0);
    }
    for a in 0..s.agents as usize {
        s.x[a] = i16::try_from(a % 32).unwrap();
        s.y[a] = 31 - i16::try_from(a / 32).unwrap();
    }
    s
}

#[test]
fn test_survival_reset_map_is_connected_and_agents_start_apart() {
    let c = cfg(32, 48, 16);
    let p = pool(c.clone(), 4, 5);
    for i in 0..4 {
        let s = p.state(i).unwrap();
        assert!(Survival::validate(&c, &s).is_ok());
        // Every passable tile is reachable from every other (one region).
        let mut seen = [[false; 96]; 96];
        let start = (0..48 * 48)
            .map(|k| (k % 48, k / 48))
            .find(|&(x, y)| map::passable(s.tiles[y][x]))
            .unwrap();
        let mut todo = vec![start];
        seen[start.1][start.0] = true;
        while let Some((x, y)) = todo.pop() {
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if !(0..48).contains(&nx) || !(0..48).contains(&ny) {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                if map::passable(s.tiles[ny][nx]) && !seen[ny][nx] {
                    seen[ny][nx] = true;
                    todo.push((nx, ny));
                }
            }
        }
        for y in 0..48 {
            for x in 0..48 {
                assert_eq!(map::passable(s.tiles[y][x]), seen[y][x], "({x}, {y})");
            }
        }
        // Every class appears, and agents stand on distinct grass tiles.
        for t in [GRASS, WATER, FOREST, STONE] {
            assert!(
                s.tiles[..48].iter().any(|r| r[..48].contains(&t)),
                "tile {t}"
            );
        }
        for a in 0..32 {
            assert_eq!(s.tiles[s.y[a] as usize][s.x[a] as usize], GRASS);
            assert!((0..a).all(|b| (s.x[b], s.y[b]) != (s.x[a], s.y[a])));
        }
    }
    assert_ne!(p.state(0).unwrap().tiles, p.state(1).unwrap().tiles);
}

#[test]
fn test_survival_t1_contested_target_nobody_moves() {
    let mut p = pool(cfg(16, 32, 16), 1, 1);
    let mut s = blank(&mut p);
    // Agents 0 and 1 at (10, 10) and (12, 10) both move to (11, 10).
    (s.x[0], s.y[0], s.x[1], s.y[1]) = (10, 10, 12, 10);
    put(&mut p, s);
    let mut acts = vec![(0, 0); 16];
    acts[0] = (2, 0); // E
    acts[1] = (4, 0); // W
    step(&mut p, &acts);
    let t = state(&p);
    assert_eq!((t.x[0], t.x[1]), (10, 12), "both stay");
}

#[test]
fn test_survival_t3_occupied_at_start_fails_even_if_the_occupant_leaves() {
    let mut p = pool(cfg(16, 32, 16), 1, 1);
    let mut s = blank(&mut p);
    // 0 at (10, 10) moves E into (11, 10), where 1 stands and moves E too.
    (s.x[0], s.y[0], s.x[1], s.y[1]) = (10, 10, 11, 10);
    put(&mut p, s);
    let mut acts = vec![(0, 0); 16];
    acts[0] = (2, 0);
    acts[1] = (2, 0);
    step(&mut p, &acts);
    let t = state(&p);
    assert_eq!(
        (t.x[0], t.x[1]),
        (10, 12),
        "1 leaves, 0 stays (start-of-step occupancy)"
    );
    // A swap: both stay.
    let mut s = t;
    (s.x[0], s.y[0], s.x[1], s.y[1]) = (20, 20, 21, 20);
    put(&mut p, s);
    acts[0] = (2, 0);
    acts[1] = (4, 0);
    step(&mut p, &acts);
    let t = state(&p);
    assert_eq!((t.x[0], t.x[1]), (20, 21));
    // Water, stone and the edge block a move.
    let mut s = t;
    s.tiles[5][6] = WATER;
    s.tiles[6][5] = STONE;
    (s.x[2], s.y[2], s.x[3], s.y[3], s.x[4], s.y[4]) = (5, 5, 6, 6, 0, 3);
    put(&mut p, s);
    let mut acts = vec![(0, 0); 16];
    acts[2] = (2, 0); // into water
    acts[3] = (4, 0); // into stone at (5, 6)
    acts[4] = (4, 0); // off the edge
    step(&mut p, &acts);
    let t = state(&p);
    assert_eq!((t.x[2], t.y[2], t.x[3], t.y[3], t.x[4]), (5, 5, 6, 6, 0));
}

#[test]
fn test_survival_mutual_attacks_both_land_and_kills_are_shared() {
    let mut p = pool(cfg(16, 32, 0), 1, 2);
    let mut s = blank(&mut p);
    // 0 and 1 face each other; 2 is north of 1, low on health.
    (s.x[0], s.y[0], s.x[1], s.y[1], s.x[2], s.y[2]) = (10, 10, 11, 10, 11, 9);
    s.health[2] = ATTACK; // one more hit kills it
    put(&mut p, s);
    let mut acts = vec![(0, 0); 16];
    acts[0] = (0, 4); // attack E (1)
    acts[1] = (0, 6); // attack W (0)
    step(&mut p, &acts);
    let t = state(&p);
    assert_eq!(
        (t.health[0], t.health[1]),
        (MAX_HEALTH - ATTACK, MAX_HEALTH - ATTACK)
    );
    // 1 and a new attacker both hit 2: both are credited with the kill.
    let mut s = t;
    (s.x[3], s.y[3]) = (12, 9);
    put(&mut p, s);
    let mut acts = vec![(0, 0); 16];
    acts[1] = (0, 3); // attack N (2)
    acts[3] = (0, 6); // attack W (2)
    let rewards = step(&mut p, &acts);
    let t = state(&p);
    assert!(!t.alive[2].get());
    assert_eq!((t.kills[1], t.kills[3], t.kills[0]), (1, 1, 0));
    assert_eq!(rewards[2], 0.0, "dead: no survival reward");
    assert_eq!(rewards[1], 1.0, "kills give no reward");
}

#[test]
fn test_survival_forage_drink_and_starvation() {
    let mut p = pool(cfg(16, 32, 16), 1, 3);
    let mut s = blank(&mut p);
    s.tiles[10][10] = FOREST;
    s.tiles[20][21] = WATER;
    (s.x[0], s.y[0], s.x[1], s.y[1]) = (10, 10, 20, 20);
    s.food[0] = 50;
    s.water[1] = 50;
    (s.food[2], s.water[2]) = (0, 40);
    put(&mut p, s);
    let mut acts = vec![(0, 0); 16];
    acts[0] = (0, 1); // forage
    acts[1] = (0, 2); // drink
    step(&mut p, &acts);
    let t = state(&p);
    assert_eq!(t.food[0], 50 + 30 - 1);
    assert_eq!(t.regrow[10][10], 32, "foraged: regrowing");
    assert_eq!(t.water[1], 50 + 30 - 1);
    assert_eq!(t.health[2], MAX_HEALTH - 5, "no food: starving");
    // A regrowing forest cannot be foraged again.
    step(&mut p, &acts);
    assert_eq!(state(&p).food[0], 79 - 1);
}

#[test]
fn test_survival_death_respawn_and_new_generation() {
    let mut p = pool(cfg(16, 32, 3), 1, 4);
    let mut s = blank(&mut p);
    (s.x[0], s.y[0], s.x[1], s.y[1]) = (10, 10, 11, 10);
    s.health[1] = ATTACK;
    put(&mut p, s);
    let generation = |p: &EnvPool<Survival, NoMasks>| p.header(0).unwrap().agents[1].generation;
    let before = generation(&p);
    let mut acts = vec![(0, 0); 16];
    acts[0] = (0, 4);
    step(&mut p, &acts);
    assert!(!state(&p).alive[1].get());
    acts[0] = (0, 0);
    for k in 1..=3 {
        step(&mut p, &acts);
        assert_eq!(
            state(&p).alive[1].get(),
            k == 3,
            "respawns after 3 dead steps"
        );
    }
    let t = state(&p);
    assert_eq!((t.health[1], t.food[1], t.water[1]), (MAX_HEALTH, 100, 100));
    assert_eq!(t.tiles[t.y[1] as usize][t.x[1] as usize], GRASS);
    assert!(generation(&p) > before, "a new agent");
}

#[test]
fn test_survival_truncates_at_steps_and_terminates_when_all_die_without_respawn() {
    let c = SurvivalConfig {
        steps: 5,
        ..cfg(16, 32, 16)
    };
    let mut p = pool(c, 1, 6);
    let acts = vec![(0, 0); 16];
    for k in 1..=5 {
        let mut a = vec![0i32; MAX_AGENTS * 2];
        for (i, &(m, x)) in acts.iter().enumerate() {
            (a[2 * i], a[2 * i + 1]) = (m, x);
        }
        let out = p
            .step(ActionInput {
                discrete: &a,
                continuous: &[],
            })
            .unwrap();
        assert_eq!(out.episode_truncated()[0].get(), k == 5);
        assert!(!out.episode_terminated()[0].get());
    }
    // No respawn and no food or water: everyone starves (100 steps of
    // stock, then 20 of 5 damage), and the episode terminates.
    let mut p = pool(cfg(16, 32, 0), 1, 7);
    let mut s = state(&p);
    for a in 0..16 {
        (s.food[a], s.water[a], s.health[a]) = (0, 0, 10);
    }
    put(&mut p, s);
    step(&mut p, &acts);
    let mut a = vec![0i32; MAX_AGENTS * 2];
    a.fill(0);
    let out = p
        .step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
    assert!(out.episode_terminated()[0].get());
}

#[test]
fn test_survival_validate_refuses_broken_states() {
    let c = cfg(16, 32, 16);
    let p = pool(c.clone(), 1, 8);
    let good = state(&p);
    assert!(Survival::validate(&c, &good).is_ok());
    let broken: [fn(&mut SurvivalState); 6] = [
        |s| s.health[0] = MAX_HEALTH + 1,
        |s| (s.x[1], s.y[1]) = (s.x[0], s.y[0]),
        |s| s.tiles[40][40] = WATER,
        |s| s.regrow[0][0] = 5,
        |s| s.alive[20] = tessera_rl_core::Bool8::TRUE,
        |s| s.health[3] = 0,
    ];
    for (k, f) in broken.iter().enumerate() {
        let mut s = good;
        f(&mut s);
        if k == 3 {
            // A timer is only refused off forest.
            s.tiles[0][0] = GRASS;
        }
        assert!(Survival::validate(&c, &s).is_err(), "case {k}");
    }
}

/// The permutation `i -> (i * 5 + 3) % n` of `n` agent slots (n odd or a
/// power of two: 5 is coprime with both).
fn perm(n: usize) -> Vec<usize> {
    (0..n).map(|i| (i * 5 + 3) % n).collect()
}

/// `s` with agent `i` moved to slot `q[i]`.
fn permuted(s: &SurvivalState, q: &[usize]) -> SurvivalState {
    let mut t = *s;
    for (i, &j) in q.iter().enumerate() {
        t.x[j] = s.x[i];
        t.y[j] = s.y[i];
        t.health[j] = s.health[i];
        t.food[j] = s.food[i];
        t.water[j] = s.water[i];
        t.dead_for[j] = s.dead_for[i];
        t.age[j] = s.age[i];
        t.kills[j] = s.kills[i];
        t.alive[j] = s.alive[i];
    }
    t
}

/// Runs up to `steps` steps (the episode, at most) of an instance and its
/// slot-permuted copy with the same actions per agent. Returns the first step where they differ after
/// relabelling, and the first step where an agent respawned in the
/// original, if any.
fn permutation_divergence(
    c: SurvivalConfig,
    seed: u64,
    steps: usize,
) -> (Option<usize>, Option<usize>) {
    let n = c.agents as usize;
    let q = perm(n);
    let mut a = pool(c.clone(), 1, seed);
    let mut b = pool(c, 1, seed);
    let sa = state(&a);
    // b holds a's agents in permuted slots; the header's agent rows follow.
    let mut header = *a.snapshot(0).unwrap().header();
    let rows = header.agents;
    for (i, &j) in q.iter().enumerate() {
        header.agents[j] = rows[i];
    }
    b.import(
        &[0],
        &[DecodedRecord::from_parts(header, permuted(&sa, &q)).unwrap()],
    )
    .unwrap();
    let mut lcg = seed;
    let mut first_respawn = None;
    for t in 0..steps {
        let was_alive: Vec<bool> = (0..n).map(|i| state(&a).alive[i].get()).collect();
        let mut acts_a = vec![0i32; MAX_AGENTS * 2];
        let mut acts_b = vec![0i32; MAX_AGENTS * 2];
        for i in 0..n {
            lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let m = i32::try_from((lcg >> 33) % 5).unwrap();
            let x = i32::try_from((lcg >> 40) % 7).unwrap();
            (acts_a[2 * i], acts_a[2 * i + 1]) = (m, x);
            (acts_b[2 * q[i]], acts_b[2 * q[i] + 1]) = (m, x);
        }
        let out = a
            .step(ActionInput {
                discrete: &acts_a,
                continuous: &[],
            })
            .unwrap();
        let ra = out.rewards().to_vec();
        let ended = out.episode_terminated()[0].get() || out.episode_truncated()[0].get();
        let rb = b
            .step(ActionInput {
                discrete: &acts_b,
                continuous: &[],
            })
            .unwrap()
            .rewards()
            .to_vec();
        let (sa, sb) = (state(&a), state(&b));
        if first_respawn.is_none() && (0..n).any(|i| !was_alive[i] && sa.alive[i].get()) {
            first_respawn = Some(t);
        }
        let same_rewards = (0..n).all(|i| ra[i].to_bits() == rb[q[i]].to_bits());
        if permuted(&sa, &q) != sb || !same_rewards {
            return (Some(t), first_respawn);
        }
        // The next step resets the instance: new agents in slot order, so
        // the permutation ends with the episode.
        if ended {
            break;
        }
    }
    (None, first_respawn)
}

#[test]
fn test_survival_slot_permutation_without_respawn_is_a_relabelling() {
    // Small and crowded, so contests, swaps and attacks all happen.
    let c = SurvivalConfig {
        steps: 4096,
        ..cfg(64, 32, 0)
    };
    let (divergence, respawned) = permutation_divergence(c, 11, 300);
    assert_eq!(divergence, None);
    assert_eq!(respawned, None);
}

#[test]
fn test_survival_slot_permutation_with_respawn_holds_until_the_first_respawn() {
    // With respawn on, only T5 (respawn draws and placement in slot order)
    // may differ: the runs agree until an agent respawns.
    let c = SurvivalConfig {
        steps: 4096,
        ..cfg(64, 32, 1)
    };
    let (divergence, first_respawn) = permutation_divergence(c, 12, 300);
    assert!(
        first_respawn.is_some(),
        "no agent respawned within 300 steps"
    );
    let first = first_respawn.unwrap();
    if let Some(d) = divergence {
        assert!(
            d >= first,
            "diverged at {d}, before the first respawn at {first}"
        );
    }
}

#[test]
fn test_survival_damage_from_two_attackers_is_summed() {
    let mut p = pool(cfg(16, 32, 16), 1, 13);
    let mut s = blank(&mut p);
    // 0 at (10, 10) is hit by 1 from the west and 2 from the east.
    (s.x[0], s.y[0], s.x[1], s.y[1], s.x[2], s.y[2]) = (10, 10, 9, 10, 11, 10);
    put(&mut p, s);
    let mut acts = vec![(0, 0); 16];
    acts[1] = (0, 4); // attack E
    acts[2] = (0, 6); // attack W
    step(&mut p, &acts);
    assert_eq!(state(&p).health[0], MAX_HEALTH - 2 * ATTACK);
}

fn masked(c: SurvivalConfig, seed: u64) -> EnvPool<Survival, Masked> {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(seed));
    let mut p = EnvPool::with_masks(c, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

/// Instance 0's mask row of agent `a`: move (5) then act (7).
fn mask_row(p: &EnvPool<Survival, Masked>, a: usize) -> Vec<bool> {
    let m = p.outputs().unwrap().masks();
    m[a * 12..a * 12 + 12].iter().map(|f| f.get()).collect()
}

#[test]
fn test_survival_masks_on_a_hand_built_state() {
    let mut p = masked(cfg(16, 32, 16), 14);
    let mut s = p.state(0).unwrap();
    for row in s.tiles.iter_mut().take(32) {
        row[..32].fill(GRASS);
    }
    for row in &mut s.regrow {
        row.fill(0);
    }
    for a in 0..16 {
        (s.x[a], s.y[a]) = (i16::try_from(a).unwrap(), 31);
    }
    // 0 at (10, 10) on ready forest; water to the north; stone to the
    // west; 1 to the east; the edge is far.
    s.tiles[10][10] = FOREST;
    s.tiles[9][10] = WATER;
    s.tiles[10][9] = STONE;
    (s.x[0], s.y[0], s.x[1], s.y[1]) = (10, 10, 11, 10);
    let header = *p.snapshot(0).unwrap().header();
    p.import(&[0], &[DecodedRecord::from_parts(header, s).unwrap()]).unwrap();
    let row = mask_row(&p, 0);
    // Moves: stay, N water, E occupied, S grass, W stone.
    assert_eq!(&row[..5], &[true, false, false, true, false]);
    // Acts: none, forage (ready forest), drink (water north), attack N, E
    // (agent 1), S, W.
    assert_eq!(&row[5..], &[true, true, true, false, true, false, false]);
    // A regrowing forest masks forage.
    let mut acts = vec![0i32; MAX_AGENTS * 2];
    acts[1] = 1;
    p.step(ActionInput { discrete: &acts, continuous: &[] }).unwrap();
    assert!(!mask_row(&p, 0)[6], "foraged: not ready");
    // The edge masks a move off the map: agent 15 at (15, 31) cannot go S.
    assert!(!mask_row(&p, 15)[3]);
}

#[test]
fn test_survival_observation_window_own_and_others() {
    let mut p = masked(cfg(16, 32, 16), 15);
    let mut s = p.state(0).unwrap();
    for row in s.tiles.iter_mut().take(32) {
        row[..32].fill(GRASS);
    }
    for row in &mut s.regrow {
        row.fill(0);
    }
    for a in 0..16 {
        (s.x[a], s.y[a]) = (i16::try_from(a).unwrap(), 31);
    }
    // 0 at (1, 1): the window reaches off the map. 2 is 1 east, 3 is 2
    // north, both visible; 4 is out of view.
    (s.x[0], s.y[0]) = (1, 1);
    (s.x[2], s.y[2], s.health[2]) = (2, 1, 40);
    (s.x[3], s.y[3], s.health[3]) = (1, 3, 70);
    (s.x[4], s.y[4]) = (20, 20);
    s.tiles[1][2] = FOREST;
    let header = *p.snapshot(0).unwrap().header();
    p.import(&[0], &[DecodedRecord::from_parts(header, s).unwrap()]).unwrap();
    let out = p.outputs().unwrap();
    // Leaf 0: window [4, 11, 11] per agent slot (view 5).
    let w = 11;
    let window = &out.observation(0).unwrap()[..4 * w * w];
    let at = |c: usize, dy: i32, dx: i32| window[c * w * w + (dy + 5) as usize * w + (dx + 5) as usize];
    assert_eq!(at(0, -5, -5), 255, "off the map");
    assert_eq!(at(0, 0, 1), FOREST);
    assert_eq!(at(1, 0, 1), 1, "ready forest");
    assert_eq!((at(2, 0, 1), at(3, 0, 1)), (1, 40), "agent 2 and its health");
    assert_eq!((at(2, 2, 0), at(3, 2, 0)), (1, 70), "agent 3");
    // Leaf 1: own.
    let own: Vec<i32> = out.observation(1).unwrap()[..24]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_ne_bytes(*c))
        .collect();
    assert_eq!(own, vec![100, 100, 100, 1, 1, 0]);
    // Leaves 2 and 3: the set's items ([16, 3] i32) and mask ([16] bool).
    let items: Vec<i32> = out.observation(2).unwrap()[..16 * 3 * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_ne_bytes(*c))
        .collect();
    let present: Vec<bool> = out.observation(3).unwrap()[..16].iter().map(|&b| b != 0).collect();
    assert_eq!(&present[..3], &[true, true, false]);
    assert_eq!(&items[..6], &[1, 0, 40, 0, 2, 70], "nearest first");
}

#[test]
fn test_survival_legal_actions_from_the_masks_always_step() {
    // Under `debug-checks` an illegal action is an error: drawing only
    // legal ones must never fail, through deaths and respawns.
    let mut p = masked(cfg(64, 32, 2), 16);
    let mut lcg = 16u64;
    for _ in 0..200 {
        let masks: Vec<bool> = p.outputs().unwrap().masks().iter().map(|f| f.get()).collect();
        let mut acts = vec![0i32; MAX_AGENTS * 2];
        for a in 0..MAX_AGENTS {
            let row = &masks[a * 12..a * 12 + 12];
            for (head, range) in [(0, 0..5), (1, 5..12)] {
                let legal: Vec<usize> = range.clone().filter(|&i| row[i]).collect();
                lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let pick = legal[(lcg >> 33) as usize % legal.len()];
                acts[2 * a + head] = i32::try_from(pick - range.start).unwrap();
            }
        }
        p.step(ActionInput { discrete: &acts, continuous: &[] }).unwrap();
        assert!(Survival::validate(&cfg(64, 32, 2), &p.state(0).unwrap()).is_ok());
    }
}
