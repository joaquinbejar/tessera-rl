// SPDX-License-Identifier: MIT OR Apache-2.0

//! The stage 2 golden: a seeded legal-play trajectory, folding the state
//! hash, the rewards and the whole mask row of every step, identical inline
//! and with 1 and 3 workers. Pinned on macOS arm64; the Linux runs check it.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::Bool8;
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, IllegalActionPolicy, Masked, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_rts::env::{MASK_PER_UNIT, N_PRODUCTS, N_TYPES, WINDOW_CELLS};
use tessera_rl_rts::{Rts, RtsConfig};

type Pool = EnvPool<Rts, Masked>;

fn cfg(size: u16, max_units: u16) -> RtsConfig {
    RtsConfig {
        size,
        decision_ticks: 4,
        max_units,
        step_limit: 400,
        seat_swap: false,
        opponent: 0,
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

fn fnv(hash: u64, word: u64) -> u64 {
    word.to_le_bytes().iter().fold(hash, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn golden_run(workers: Option<usize>) -> u64 {
    let c = cfg(12, 6);
    let mut p = pool(c, 2, 2026, workers);
    let mut rng = Rng(0x7e55_u64);
    let mut masks = ready(&mut p);
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for _ in 0..96 {
        let a = legal_actions(&mut rng, &masks, 2, 6);
        let out = p.step(input(&a)).unwrap();
        for r in out.rewards() {
            hash = fnv(hash, u64::from(r.to_bits()));
        }
        masks = out.masks().to_vec();
        hash = masks.iter().fold(hash, |h, m| fnv(h, u64::from(m.get())));
        for i in 0..2 {
            hash = fnv(hash, p.state_hash(i).unwrap().get());
        }
    }
    hash
}

/// Pinned on macOS arm64; any change to the rules, the masks or the
/// encoding moves it.
const RTS_GOLDEN: u64 = 0x7dcd_1abc_14aa_2177;

/// The stage 2 fields of one state, folded field by field: units, terrain,
/// ore, decisions and ticks. A field that a later stage adds is not read.
fn fold_dynamics(mut hash: u64, s: &tessera_rl_rts::RtsState) -> u64 {
    for u in &s.units {
        for w in [
            u64::from(u.cell),
            u64::from(u.kind),
            u64::from(u.hp),
            u64::from(u.alive.get()),
            u64::from(u.finished.get()),
            u64::from(u.carry),
            u64::from(u.counter),
            u64::from(u.progress),
            u64::from(u.producing),
            u64::from(u.generation),
        ] {
            hash = fnv(hash, w);
        }
    }
    for &t in &s.terrain {
        hash = fnv(hash, u64::from(t));
    }
    for w in [
        u64::from(s.ore[0]),
        u64::from(s.ore[1]),
        u64::from(s.decisions),
        u64::from(s.ticks),
    ] {
        hash = fnv(hash, w);
    }
    hash
}

/// The golden's game with the state hash replaced by [`fold_dynamics`].
fn dynamics_run() -> u64 {
    let c = cfg(12, 6);
    let mut p = pool(c, 2, 2026, None);
    let mut rng = Rng(0x7e55_u64);
    let mut masks = ready(&mut p);
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for _ in 0..96 {
        let a = legal_actions(&mut rng, &masks, 2, 6);
        let out = p.step(input(&a)).unwrap();
        for r in out.rewards() {
            hash = fnv(hash, u64::from(r.to_bits()));
        }
        masks = out.masks().to_vec();
        hash = masks.iter().fold(hash, |h, m| fnv(h, u64::from(m.get())));
        for i in 0..2 {
            hash = fnv_state(hash, &p, i);
        }
    }
    hash
}

fn fnv_state(hash: u64, p: &Pool, i: u32) -> u64 {
    fold_dynamics(hash, p.state(i).unwrap())
}

/// Pinned on the stage 2 code (`rts-design.md` section 14, ruling 6): fog
/// must not change units, terrain, ore, counters, rewards or masks, so
/// this value never moves in stage 3, while [`RTS_GOLDEN`] moves with the
/// state's new field.
const DYNAMICS_GOLDEN: u64 = 0x1d8e_5670_c0d3_ad6f;

#[test]
fn test_rts_dynamics_are_unchanged_by_fog() {
    let hash = dynamics_run();
    assert_eq!(hash, DYNAMICS_GOLDEN, "{hash:#x}");
}

/// Both observation leaves of every agent, folded over 300 decisions of the
/// golden's seeded legal play, inline and with 3 workers.
fn observation_run(workers: Option<usize>) -> u64 {
    let c = cfg(12, 6);
    let mut p = pool(c, 2, 2026, workers);
    let mut rng = Rng(0x0b5e_u64);
    let mut masks = ready(&mut p);
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for _ in 0..300 {
        let a = legal_actions(&mut rng, &masks, 2, 6);
        let out = p.step(input(&a)).unwrap();
        for leaf in 0..2 {
            for &b in out.observation(leaf).unwrap() {
                hash = fnv(hash, u64::from(b));
            }
        }
        masks = out.masks().to_vec();
    }
    hash
}

/// Pinned on macOS arm64 for `ENV_VERSION` 2: the fogged planes and the
/// globals.
const OBSERVATION_GOLDEN: u64 = 0xe4d2_fa41_d533_cc23;

#[test]
fn test_rts_observation_golden_is_pinned_and_identical_across_workers() {
    let inline = observation_run(None);
    assert_eq!(inline, OBSERVATION_GOLDEN, "{inline:#x}");
    assert_eq!(observation_run(Some(3)), inline);
}

#[test]
fn test_rts_golden_is_pinned_and_identical_across_workers() {
    let inline = golden_run(None);
    assert_eq!(inline, RTS_GOLDEN, "{inline:#x}");
    assert_eq!(golden_run(Some(1)), inline);
    assert_eq!(golden_run(Some(3)), inline);
}
