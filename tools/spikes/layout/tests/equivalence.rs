// SPDX-License-Identifier: MIT OR Apache-2.0

//! Correctness before timing: every layout ends bit-identical to L0 (the
//! hash of each header and state), after plain stepping and after the
//! fork-heavy workload.

use tessera_rl_classic::{CartPole, CartPoleConfig, Tag, TagConfig};
use tessera_rl_spike_layout::kernel::{SpikeEnv, Work};
use tessera_rl_spike_layout::layouts::{Blocks, Columns, Combined, Layout, Parallel};
use tessera_rl_spike_layout::workload::{ActionTable, fork_heavy, hashes, run};

const N: usize = 37;

fn outcome<E: SpikeEnv, L: Layout<E>>(cfg: &E::Config, choices: u32) -> Vec<(u64, u64)> {
    let mut layout = L::new(cfg, N, 9);
    let mut work = Work::new::<E>(cfg, N);
    let mut table = ActionTable::new(N * E::MAX_AGENTS, choices, 97);
    run::<E, L>(&mut layout, cfg, &mut work, &mut table, 0, 300, 200);
    fork_heavy::<E, L>(&mut layout, cfg, &mut work, &mut table, 8, 10, 200);
    run::<E, L>(&mut layout, cfg, &mut work, &mut table, 1_000, 50, 200);
    hashes::<E, L>(&layout, N)
}

#[test]
fn test_layouts_cartpole_match_parallel_bit_for_bit() {
    let cfg = CartPoleConfig::default();
    let reference = outcome::<CartPole, Parallel<CartPole>>(&cfg, 2);
    assert_eq!(reference.len(), N);
    assert_eq!(outcome::<CartPole, Combined<CartPole>>(&cfg, 2), reference);
    assert_eq!(outcome::<CartPole, Columns<CartPole>>(&cfg, 2), reference);
    assert_eq!(outcome::<CartPole, Blocks<CartPole>>(&cfg, 2), reference);
}

#[test]
fn test_layouts_tag_match_parallel_bit_for_bit() {
    for cfg in [
        TagConfig::default(),
        TagConfig {
            pursuers: 4,
            evaders: 4,
            slip: 300,
            ..TagConfig::default()
        },
    ] {
        let reference = outcome::<Tag, Parallel<Tag>>(&cfg, 5);
        assert_eq!(reference.len(), N);
        assert_eq!(outcome::<Tag, Combined<Tag>>(&cfg, 5), reference);
        assert_eq!(outcome::<Tag, Columns<Tag>>(&cfg, 5), reference);
        assert_eq!(outcome::<Tag, Blocks<Tag>>(&cfg, 5), reference);
    }
}

#[test]
fn test_lanes_soa_physics_matches_aos() {
    use tessera_rl_classic::CartPoleState;
    use tessera_rl_spike_layout::columns::Columnar;
    use tessera_rl_spike_layout::lanes::{physics_aos, physics_soa};
    let mut aos: Vec<CartPoleState> = (0..64)
        .map(|i| CartPoleState {
            s: [0.001 * f64::from(i), -0.01, 0.02, 0.003 * f64::from(i)],
        })
        .collect();
    let mut soa = CartPoleState::columns(64);
    for (i, s) in aos.iter().enumerate() {
        s.scatter(&mut soa, i);
    }
    let push: Vec<i32> = (0..64).map(|i| i % 2).collect();
    for _ in 0..100 {
        physics_aos(&mut aos, &push);
        physics_soa(&mut soa, &push);
    }
    for (i, s) in aos.iter().enumerate() {
        assert_eq!(
            CartPoleState::gather(&soa, i).s.map(f64::to_bits),
            s.s.map(f64::to_bits)
        );
    }
}

#[test]
fn test_layouts_copy_all_reproduces_every_instance() {
    fn check<E: SpikeEnv, L: Layout<E>>(cfg: &E::Config, choices: u32) {
        let mut layout = L::new(cfg, N, 9);
        let mut work = Work::new::<E>(cfg, N);
        let mut table = ActionTable::new(N * E::MAX_AGENTS, choices, 97);
        run::<E, L>(&mut layout, cfg, &mut work, &mut table, 0, 120, 200);
        let mut copy = L::new(cfg, N, 1);
        layout.copy_all(&mut copy);
        assert_eq!(hashes::<E, L>(&copy, N), hashes::<E, L>(&layout, N));
    }
    let cp = CartPoleConfig::default();
    check::<CartPole, Parallel<CartPole>>(&cp, 2);
    check::<CartPole, Combined<CartPole>>(&cp, 2);
    check::<CartPole, Columns<CartPole>>(&cp, 2);
    check::<CartPole, Blocks<CartPole>>(&cp, 2);
    let tag = TagConfig::default();
    check::<Tag, Parallel<Tag>>(&tag, 5);
    check::<Tag, Combined<Tag>>(&tag, 5);
    check::<Tag, Columns<Tag>>(&tag, 5);
    check::<Tag, Blocks<Tag>>(&tag, 5);
}
