// SPDX-License-Identifier: MIT OR Apache-2.0

//! The fogged observation (`rts-design.md` section 14, ruling 3): planes
//! on a hand-built state for both sides, explored-only cells, the globals
//! and their clamp, the space, the pool's observation equal to `observe`,
//! and the search pool refusing the environment (ADR-0035).

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    reason = "test fixtures"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::ObsWriter;
use tessera_rl_core::space::Space;
use tessera_rl_core::{AgentSlot, Bool8, Environment, Reserved};
use tessera_rl_engine::{AutoReset, EnvPool, IllegalActionPolicy, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed};
use tessera_rl_num::rng::{ObsRng, RngAuthority};
use tessera_rl_rts::env::{CHANNELS, GLOBALS, SIDE_SLOTS};
use tessera_rl_rts::fog::side_bit;
use tessera_rl_rts::units::{Kind, Unit};
use tessera_rl_rts::world::{Grid, Team, Terrain};
use tessera_rl_rts::{Rts, RtsConfig, RtsState};
use tessera_rl_search::{SearchError, SearchOptions, SearchPool};

fn cfg(size: u16) -> RtsConfig {
    RtsConfig {
        size,
        decision_ticks: 4,
        max_units: 4,
        step_limit: 600,
        seat_swap: false,
        opponent: 0,
    }
}

/// `(planes, globals)` of agent `a` (agent 0 plays side Zero).
fn observe(c: &RtsConfig, s: &RtsState, a: u16) -> (Vec<u8>, [u16; 2]) {
    let cells = usize::from(c.size) * usize::from(c.size);
    let mut planes = vec![0u8; usize::try_from(CHANNELS).unwrap() * cells];
    let mut globals = [0u8; 4];
    {
        let mut leaves = [&mut planes[..], &mut globals[..]];
        let mut w = ObsWriter::new(&mut leaves);
        let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
        let mut r = ObsRng::new(&RngAuthority::engine_only(), &identity);
        Rts::observe(c, s, AgentSlot::new(a, 2).unwrap(), &mut r, &mut w);
    }
    let g = [
        u16::from_ne_bytes([globals[0], globals[1]]),
        u16::from_ne_bytes([globals[2], globals[3]]),
    ];
    (planes, g)
}

fn unit(kind: Kind, cell: u16) -> Unit {
    Unit {
        cell,
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

/// Plane `k` at frame position `(x, y)`.
fn at(planes: &[u8], size: u16, k: usize, x: u16, y: u16) -> u8 {
    let cells = usize::from(size) * usize::from(size);
    planes[k * cells + usize::from(y) * usize::from(size) + usize::from(x)]
}

#[test]
fn test_observe_planes_on_a_hand_built_state_for_both_sides() {
    let c = cfg(16);
    let g = Grid::new(16).unwrap();
    let cell = |x, y| g.index(x, y).unwrap();
    let mut s = RtsState::default();
    s.units[0] = unit(Kind::Core, cell(1, 1));
    s.units[1] = Unit {
        carry: 1,
        hp: 1,
        ..unit(Kind::Drone, cell(6, 6))
    };
    s.units[SIDE_SLOTS] = unit(Kind::Core, cell(14, 14));
    // An enemy Striker 2 cells from the Drone: seen by side Zero.
    s.units[SIDE_SLOTS + 1] = Unit {
        hp: 2,
        ..unit(Kind::Striker, cell(8, 6))
    };
    // An enemy Lancer far from every own unit: hidden from side Zero.
    s.units[SIDE_SLOTS + 2] = unit(Kind::Lancer, cell(12, 2));
    s.terrain[usize::from(cell(7, 7))] = Terrain::Wall as u8;
    s.terrain[usize::from(cell(13, 3))] = Terrain::Ore as u8;
    // Side Zero has explored (13, 3) earlier but sees it no more.
    s.explored[usize::from(cell(13, 3))] = side_bit(Team::Zero);
    let (p, _) = observe(&c, &s, 0);
    // Own units.
    assert_eq!(at(&p, 16, 3, 6, 6), Kind::Drone as u8);
    assert_eq!(at(&p, 16, 4, 6, 6), 1);
    assert_eq!(at(&p, 16, 5, 6, 6), 1);
    // The seen enemy: kind and hit points.
    assert_eq!(
        (at(&p, 16, 7, 8, 6), at(&p, 16, 8, 8, 6)),
        (Kind::Striker as u8, 2)
    );
    // The hidden enemy: nothing, on any plane.
    assert_eq!(at(&p, 16, 7, 12, 2), 0);
    assert_eq!(at(&p, 16, 1, 12, 2), 0);
    // Visible: the wall shows, flags set. explored is only what `s` says.
    assert_eq!(at(&p, 16, 1, 7, 7), 1);
    assert_eq!(at(&p, 16, 2, 7, 7), 0);
    assert_eq!(at(&p, 16, 0, 7, 7), 0, "terrain only where explored");
    // Explored but not visible: terrain shows, no unit plane.
    assert_eq!(at(&p, 16, 0, 13, 3), Terrain::Ore as u8);
    assert_eq!((at(&p, 16, 1, 13, 3), at(&p, 16, 2, 13, 3)), (0, 1));
    // Side One, in its frame: the map's (x, y) is its (15 - x, 15 - y).
    let (q, _) = observe(&c, &s, 1);
    assert_eq!(at(&q, 16, 3, 1, 1), Kind::Core as u8);
    assert_eq!(at(&q, 16, 3, 7, 9), Kind::Striker as u8);
    assert_eq!(
        at(&q, 16, 7, 9, 9),
        Kind::Drone as u8,
        "the Striker sees the Drone"
    );
    assert_eq!(at(&q, 16, 7, 14, 14), 0, "side Zero's Core is beyond sight");
    assert_eq!(
        at(&q, 16, 2, 2, 12),
        0,
        "side Zero's explored bit is not shown"
    );
}

#[test]
fn test_observe_globals_clamp_ore_and_scale_decisions() {
    let c = cfg(12);
    let g = Grid::new(12).unwrap();
    let mut s = RtsState::default();
    s.units[0] = unit(Kind::Core, g.index(0, 0).unwrap());
    s.units[SIDE_SLOTS] = unit(Kind::Core, g.index(11, 11).unwrap());
    for (ore, decisions, want) in [
        (0, 0, [0, 0]),
        (65_535, 300, [65_535, 32_767]),
        (65_536, 600, [65_535, 65_535]),
        (u32::MAX, 1, [65_535, 109]),
    ] {
        s.ore = [ore, 7];
        s.decisions = decisions;
        assert_eq!(
            observe(&c, &s, 0).1,
            want,
            "ore {ore} decisions {decisions}"
        );
        assert_eq!(observe(&c, &s, 1).1[0], 7);
    }
}

#[test]
fn test_observe_space_follows_the_config() {
    for size in [8u16, 32] {
        let Space::Dict(leaves) = Rts::observation_space(&cfg(size)).unwrap() else {
            panic!("a dict space");
        };
        let shapes: Vec<_> = leaves
            .iter()
            .map(|(_, s)| match s {
                Space::Box { shape, .. } => shape.clone(),
                _ => panic!("box leaves"),
            })
            .collect();
        assert_eq!(
            shapes,
            vec![
                vec![CHANNELS, u32::from(size), u32::from(size)],
                vec![GLOBALS]
            ]
        );
    }
}

#[test]
fn test_observe_search_pool_refuses_the_fogged_environment() {
    const { assert!(!Rts::PERFECT_INFORMATION) };
    let c = cfg(12);
    let options = PoolOptions {
        auto_reset: AutoReset::NextStep,
        illegal_action: IllegalActionPolicy::Noop,
        ..PoolOptions::new(NonZeroU32::new(1).unwrap(), Seed::new(1))
    };
    let mut p: EnvPool<Rts, Masked> = EnvPool::with_masks(c.clone(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let root = p.snapshot(0).unwrap();
    let search = SearchOptions::new(NonZeroU32::new(64).unwrap(), NonZeroU32::new(4).unwrap());
    let pool = SearchPool::<Rts, Masked>::new(c, search, &root, EnvPool::with_masks);
    assert!(matches!(pool, Err(SearchError::ImperfectInformation)));
}

#[test]
fn test_observe_own_finished_and_producing_planes_on_both_sides() {
    let c = cfg(16);
    let g = Grid::new(16).unwrap();
    let cell = |x, y| g.index(x, y).unwrap();
    let mut s = RtsState::default();
    // Side Zero: a Core producing a Drone, an unfinished Foundry at
    // progress 0 (the case a policy could not tell before).
    s.units[0] = Unit {
        producing: Kind::Drone as u8,
        progress: 1,
        ..unit(Kind::Core, cell(1, 1))
    };
    s.units[1] = Unit {
        finished: Bool8::FALSE,
        hp: 1,
        ..unit(Kind::Foundry, cell(3, 1))
    };
    // Side One mirrors it, plus a visible enemy of side Zero next to an
    // own unit, unfinished and producing: planes 9 and 10 stay 0 there.
    s.units[SIDE_SLOTS] = Unit {
        producing: Kind::Drone as u8,
        progress: 1,
        ..unit(Kind::Core, cell(14, 14))
    };
    s.units[SIDE_SLOTS + 1] = Unit {
        finished: Bool8::FALSE,
        hp: 1,
        ..unit(Kind::Foundry, cell(12, 14))
    };
    s.units[2] = Unit {
        finished: Bool8::FALSE,
        hp: 1,
        ..unit(Kind::Foundry, cell(12, 12))
    };
    s.units[SIDE_SLOTS + 2] = Unit {
        producing: Kind::Striker as u8,
        progress: 1,
        ..unit(Kind::Foundry, cell(4, 4))
    };
    for a in [0u16, 1] {
        let (p, _) = observe(&c, &s, a);
        // In each side's frame its own Core is at (1, 1), its unfinished
        // Foundry at (3, 1).
        assert_eq!(
            (at(&p, 16, 9, 1, 1), at(&p, 16, 10, 1, 1)),
            (1, Kind::Drone as u8),
            "agent {a}"
        );
        assert_eq!(
            (at(&p, 16, 9, 3, 1), at(&p, 16, 10, 3, 1)),
            (0, 0),
            "agent {a}"
        );
        // Every enemy unit's cell, seen or not, is 0 on both planes.
        for (x, y) in [(14, 14), (12, 14), (12, 12), (4, 4)] {
            let (fx, fy) = if a == 0 { (x, y) } else { (15 - x, 15 - y) };
            let enemy = if a == 0 {
                [(14, 14), (12, 14), (4, 4)]
            } else {
                [(1, 1), (3, 1), (12, 12)]
            };
            if enemy.contains(&(x, y)) {
                assert_eq!(
                    (at(&p, 16, 9, fx, fy), at(&p, 16, 10, fx, fy)),
                    (0, 0),
                    "agent {a} ({x}, {y})"
                );
            }
        }
    }
    // Side One's own Foundry at (4, 4) map, producing a Striker, is
    // finished and producing in its planes, at frame (11, 11).
    let (q, _) = observe(&c, &s, 1);
    assert_eq!(
        (at(&q, 16, 9, 11, 11), at(&q, 16, 10, 11, 11)),
        (1, Kind::Striker as u8)
    );
}
