// SPDX-License-Identifier: MIT OR Apache-2.0

//! Vrp-v1 rules: instance generation, the exact rounded distance, masks,
//! loads and refills, rewards as travelled distance, termination within
//! `2 * customers` steps, observations, validation and the baseline.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::{Environment, Validate};
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;
use tessera_rl_ops::vrp::{DEPOT, MAX_NODES, NODE_FEATURES, baseline, round_sqrt};
use tessera_rl_ops::{Vrp, VrpConfig, VrpState};

fn cfg(customers: u32, capacity: u32) -> VrpConfig {
    VrpConfig {
        customers,
        vehicles: 1,
        grid: 50,
        max_demand: 9,
        capacity,
    }
}

fn pool(c: VrpConfig, n: u32, seed: u64) -> EnvPool<Vrp, Masked> {
    let options = PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed));
    let mut p = EnvPool::with_masks(c, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn state(p: &EnvPool<Vrp, Masked>, i: u32) -> VrpState {
    *p.state(i).unwrap()
}

/// One step of one instance; the reward and whether it ended.
fn step(p: &mut EnvPool<Vrp, Masked>, action: usize) -> (f32, bool) {
    let a = [i32::try_from(action).unwrap(); 8];
    let out = p
        .step(ActionInput {
            discrete: &a[..8],
            continuous: &[],
        })
        .unwrap();
    (
        out.rewards().first().copied().unwrap(),
        out.episode_terminated().first().unwrap().get(),
    )
}

fn mask(p: &EnvPool<Vrp, Masked>) -> Vec<bool> {
    p.outputs()
        .unwrap()
        .masks()
        .iter()
        .take(usize::from(state(p, 0).customers) + 1)
        .map(|f| f.get())
        .collect()
}

#[test]
fn test_vrp_round_sqrt_is_exact_rounding() {
    for v in 0u64..20_000 {
        #[expect(clippy::cast_precision_loss, reason = "small exact values")]
        let f = (v as f64).sqrt();
        // No value is a tie: (r + 0.5)^2 is never an integer.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "small"
        )]
        let expected = f.round() as u64;
        assert_eq!(round_sqrt(v), expected, "{v}");
    }
    assert_eq!(round_sqrt(u64::from(u32::MAX)), 65_536);
    assert_eq!(round_sqrt(2 * 999 * 999), 1_413);
}

#[test]
fn test_vrp_reset_places_nodes_and_demands_in_range() {
    let c = cfg(30, 30);
    let p = pool(c.clone(), 4, 3);
    for i in 0..4 {
        let s = state(&p, i);
        assert_eq!((s.customers, s.served, s.at[0], s.load[0]), (30, 0, 0, 30));
        assert!(s.x[..31].iter().chain(&s.y[..31]).all(|&v| v < 50));
        assert_eq!(s.demand[DEPOT], 0);
        assert!(s.demand[1..31].iter().all(|&d| (1..=9).contains(&d)));
        assert!(s.demand[31..].iter().all(|&d| d == 0));
        assert!(Vrp::validate(&c, &s).is_ok());
    }
    assert_ne!(state(&p, 0).x, state(&p, 1).x);
}

#[test]
fn test_vrp_masks_loads_refills_and_rewards() {
    // Capacity 9 is the largest demand: one customer fits per trip at worst.
    let c = cfg(5, 9);
    let mut p = pool(c, 1, 7);
    let s0 = state(&p, 0);
    let m = mask(&p);
    assert!(
        !m[DEPOT],
        "at the depot with customers left, the depot is masked"
    );
    assert!(
        m[1..].iter().all(|&f| f),
        "a full vehicle fits every customer"
    );
    let (r, ended) = step(&mut p, 2);
    assert!(!ended);
    assert_eq!(
        r,
        -(f32::from(u16::try_from(s0.distance(DEPOT, 2)).unwrap()))
    );
    let s1 = state(&p, 0);
    assert_eq!((s1.at[0], s1.served, s1.visited[2]), (2, 1, 1));
    assert_eq!(s1.load[0], 9 - u16::from(s0.demand[2]));
    let m = mask(&p);
    assert!(
        m[DEPOT] && !m[2],
        "the depot is legal away from it; served is masked"
    );
    for (n, &legal) in m.iter().enumerate().skip(1) {
        if n != 2 {
            assert_eq!(legal, u16::from(s1.demand[n]) <= s1.load[0], "node {n}");
        }
    }
    step(&mut p, DEPOT);
    let s2 = state(&p, 0);
    assert_eq!((s2.at[0], s2.load[0]), (0, 9), "the depot refills");
    assert_eq!(s2.travelled[0], 2 * s0.distance(DEPOT, 2));
}

#[test]
fn test_vrp_episode_ends_home_within_two_steps_per_customer() {
    for (customers, capacity, seed) in [(10, 9, 1), (25, 30, 2), (40, 200, 3)] {
        let mut p = pool(cfg(customers, capacity), 1, seed);
        let mut total = 0f32;
        let mut steps = 0;
        loop {
            let s = state(&p, 0);
            let (r, ended) = step(&mut p, baseline::act(&s, 0));
            total += r;
            steps += 1;
            if ended {
                let end = state(&p, 0);
                assert_eq!(end.at[0], 0, "ends at the depot");
                assert_eq!(usize::from(end.served), usize::try_from(customers).unwrap());
                assert_eq!(
                    total,
                    -f32::from(
                        u16::try_from(s.travelled[0] + s.distance(s.node_of(0), DEPOT)).unwrap()
                    )
                );
                break;
            }
            assert!(steps <= 2 * customers, "more than 2 * customers steps");
        }
    }
}

#[test]
fn test_vrp_any_legal_policy_ends_within_two_steps_per_customer() {
    // The highest legal action each step: wanders as far as it can.
    let mut p = pool(cfg(30, 9), 1, 9);
    let mut ended = false;
    for _ in 0..60 {
        let m = mask(&p);
        let a = m.iter().rposition(|&f| f).unwrap();
        if step(&mut p, a).1 {
            ended = true;
            break;
        }
    }
    assert!(ended, "the episode did not end within 2 * customers steps");
}

#[test]
fn test_vrp_observation_shapes_and_values() {
    let c = cfg(6, 30);
    let text = format!("{:?}", Vrp::observation_space(&c).unwrap());
    assert!(text.contains("[7, 4]") && text.contains("[3]"), "{text}");
    let mut p = pool(c, 1, 4);
    step(&mut p, 3);
    let s = state(&p, 0);
    let out = p.outputs().unwrap();
    let ints = |b: &[u8]| -> Vec<i32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_ne_bytes(*c))
            .collect()
    };
    // Each leaf holds every agent slot's row; vehicle 0 is the first.
    let nodes = ints(out.observation(0).unwrap());
    let vehicle = ints(out.observation(1).unwrap());
    assert_eq!(nodes.len(), 8 * 7 * NODE_FEATURES);
    assert_eq!(vehicle.len(), 8 * 3);
    let (nodes, vehicle) = (&nodes[..7 * NODE_FEATURES], vehicle[..3].to_vec());
    let row = |n: usize| &nodes[n * NODE_FEATURES..(n + 1) * NODE_FEATURES];
    assert_eq!(row(3), &[i32::from(s.x[3]), i32::from(s.y[3]), 0, 1]);
    assert_eq!(row(1)[2], i32::from(s.demand[1]));
    assert_eq!(row(DEPOT)[2..], [0, 0]);
    let travelled = i32::try_from(s.travelled[0]).unwrap();
    assert_eq!(vehicle, vec![3, i32::from(s.load[0]), travelled]);
}

#[test]
fn test_vrp_validate_refuses_broken_states() {
    let c = cfg(5, 30);
    let mut p = pool(c.clone(), 1, 6);
    step(&mut p, 1);
    let good = state(&p, 0);
    assert!(Vrp::validate(&c, &good).is_ok());
    assert!(Vrp::validate(&c, &VrpState::default()).is_ok());
    let broken: [fn(&mut VrpState); 8] = [
        |s| s.x[2] = 50,
        |s| s.demand[DEPOT] = 1,
        |s| s.demand[3] = 10,
        |s| s.visited[2] = 1,
        |s| s.at[0] = 2,
        |s| s.load[0] = 31,
        |s| s.x[MAX_NODES - 1] = 1,
        |s| s.done = 1,
    ];
    for (k, f) in broken.iter().enumerate() {
        let mut s = good;
        f(&mut s);
        assert!(Vrp::validate(&c, &s).is_err(), "case {k}");
    }
    assert!(Vrp::validate(&cfg(6, 30), &good).is_err(), "another shape");
}

#[test]
fn test_vrp_config_refuses_capacity_below_max_demand() {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    assert!(EnvPool::<Vrp, Masked>::with_masks(cfg(5, 8), options).is_err());
    assert!(EnvPool::<Vrp, Masked>::with_masks(cfg(5, 9), options).is_ok());
}

#[test]
fn test_vrp_baseline_takes_the_nearest_feasible_customer() {
    let p = pool(cfg(12, 30), 1, 8);
    let s = state(&p, 0);
    let pick = baseline::act(&s, 0);
    let best = (1..=12).map(|n| s.distance(DEPOT, n)).min().unwrap();
    assert_eq!(s.distance(DEPOT, pick), best);
    assert!(
        (1..pick).all(|n| s.distance(DEPOT, n) > best),
        "ties take the lowest node"
    );
}

#[test]
fn test_vrp_largest_instance_masks_are_501_wide() {
    let c = VrpConfig {
        customers: 500,
        vehicles: 1,
        grid: 1000,
        max_demand: 255,
        capacity: 10_000,
    };
    let mut p = pool(c, 1, 5);
    assert_eq!(p.outputs().unwrap().masks().len(), 8 * 501);
    let mut steps = 0;
    loop {
        let a = baseline::act(&state(&p, 0), 0);
        if step(&mut p, a).1 {
            break;
        }
        steps += 1;
        assert!(steps <= 1_000);
    }
}

#[test]
fn test_vrp_illegal_action_is_replaced_by_the_lowest_legal_one() {
    let mut p = pool(cfg(5, 30), 1, 12);
    let s0 = state(&p, 0);
    // At reset the depot is illegal: the engine's policy replaces it with
    // the no-op, the lowest legal action, customer 1.
    let (r, _) = step(&mut p, DEPOT);
    let s1 = state(&p, 0);
    assert_eq!((s1.at[0], s1.served), (1, 1));
    assert_eq!(r, -f32::from(u16::try_from(s0.distance(DEPOT, 1)).unwrap()));
}
