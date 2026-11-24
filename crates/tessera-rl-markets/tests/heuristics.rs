// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scripted baselines (design section 5): over many steps, every
//! action each one writes is legal under the mask the engine reported for
//! that state, the pool counts no illegal action, and the position, order
//! and cash limits hold. Run per baseline with every agent playing it, and
//! for the default mix through `Heuristic::act`.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures: a failure is the assertion"
)]

mod support;

use support::{Pool, agent_limits, config, input, legal_under, pool};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic, ImperfectInformation, Validate};
use tessera_rl_engine::Reseed;
use tessera_rl_markets::actions::{HEADS, MASK_LEN, kind};
use tessera_rl_markets::limits::MAX_AGENTS;
use tessera_rl_markets::{Baseline, LimitOrderBook, LobConfig, act_info};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

const INSTANCES: u32 = 2;
const STEPS: u32 = 300;

fn rng(step: u32, row: usize) -> StepRng {
    let id = RngIdentity::new(
        Seed::new(u64::from(step)),
        LogicalId::new(u32::try_from(row).unwrap()),
    );
    StepRng::new(&RngAuthority::engine_only(), &id, StreamId::DYNAMICS)
}

/// Plays `STEPS` steps where every alive slot's action comes from
/// `choose`, checking each against its mask row and the limits after
/// every step; returns how many non-noop actions were played.
fn play(
    cfg: &LobConfig,
    seed: u64,
    choose: impl Fn(&LobConfig, &tessera_rl_markets::LobState, AgentSlot, &mut StepRng) -> [i32; HEADS],
) -> usize {
    let mut p: Pool = pool(cfg.clone(), INSTANCES, seed, None);
    let mut masks = p.reset_all(Reseed::Keep).unwrap().masks().to_vec();
    let mut active = 0;
    for t in 0..STEPS {
        let mut acts = vec![0; usize::try_from(INSTANCES).unwrap() * MAX_AGENTS * HEADS];
        for i in 0..usize::try_from(INSTANCES).unwrap() {
            let state = p.state(u32::try_from(i).unwrap()).unwrap();
            for slot in 0..cfg.agent_count() {
                if !state.is_alive(slot) {
                    continue;
                }
                let row = i * MAX_AGENTS + slot;
                let agent = AgentSlot::new(u16::try_from(slot).unwrap(), MAX_AGENTS).unwrap();
                let h = choose(cfg, state, agent, &mut rng(t, row));
                let mask = &masks[row * MASK_LEN..(row + 1) * MASK_LEN];
                assert!(legal_under(mask, h), "step {t}, row {row}: {h:?} is masked");
                if h[0] != kind::NOOP {
                    active += 1;
                }
                acts[row * HEADS..(row + 1) * HEADS].copy_from_slice(&h);
            }
        }
        let out = p.step(input(&acts)).unwrap();
        assert!(
            out.episode_illegal_actions()
                .iter()
                .all(|n| n.to_bits() == 0)
        );
        masks = out.masks().to_vec();
        for i in 0..INSTANCES {
            let s = p.state(i).unwrap();
            assert_eq!(LimitOrderBook::validate(cfg, s), Ok(()));
            agent_limits(cfg, s).unwrap();
            if !cfg.allow_short_cash && s.step < cfg.steps {
                assert!(s.accounts.iter().all(|a| a.cash >= 0));
            }
        }
    }
    active
}

fn cfg() -> LobConfig {
    LobConfig {
        steps: 120,
        max_position: 6,
        initial_cash: 4000,
        ..config(8, 1)
    }
}

#[test]
fn test_heuristics_each_baseline_plays_only_masked_legal_actions() {
    for b in Baseline::ALL {
        for (seed, latency) in [(1u64, 0u32), (2, 2)] {
            let c = LobConfig { latency, ..cfg() };
            let active = play(&c, seed, |cfg, state, agent, rng| {
                let info = LimitOrderBook::info_state(cfg, state, agent);
                let mut h = [0i32; HEADS];
                act_info(cfg, &info, b, rng, &mut ActionWriter::new(&mut h, &mut []));
                h
            });
            // Momentum needs a moving mid, which other baselines make;
            // the maker and the noise trader act on their own.
            if b != Baseline::Momentum {
                assert!(active > 0, "{b:?} never acted");
            }
        }
    }
}

#[test]
fn test_heuristics_default_mix_through_heuristic_act_is_legal() {
    let active = play(&cfg(), 3, |cfg, state, agent, rng| {
        let mut h = [0i32; HEADS];
        LimitOrderBook::act(
            cfg,
            state,
            agent,
            rng,
            &mut ActionWriter::new(&mut h, &mut []),
        );
        h
    });
    assert!(active > 0);
}
