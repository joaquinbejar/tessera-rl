// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Determinize` (ADR-0035, ADR-0046; review r1 of #381, correction 3):
//! on states reached by masked play, every sample for every alive agent
//! passes `Validate`, has the agent's information state exactly, keeps
//! the public book and the observer's own part, and gives every agent at
//! most `MAX_OPEN` resting plus pending limit orders and positions within
//! `max_position` counting open and pending lots. The heuristics read the
//! information state only: on a sample they act as on the original.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "property tests: a failure is the assertion"
)]

mod support;

use proptest::prelude::*;
use support::{Rng, agent_limits, config, input, legal_actions, pool};
use tessera_rl_core::encoding::encode_to_vec;
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Determinize, ImperfectInformation, Validate};
use tessera_rl_engine::Reseed;
use tessera_rl_markets::actions::HEADS;
use tessera_rl_markets::book::Level;
use tessera_rl_markets::limits::MAX_AGENTS;
use tessera_rl_markets::{Baseline, LimitOrderBook, LobConfig, LobState, Side, act_info};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

const INSTANCES: u32 = 2;
const SAMPLES: u64 = 6;

fn rng(seed: u64, stream: u32) -> StepRng {
    let id = RngIdentity::new(Seed::new(seed), LogicalId::new(stream));
    StepRng::new(&RngAuthority::engine_only(), &id, StreamId::DYNAMICS)
}

fn levels(s: &LobState, side: Side) -> Vec<Level> {
    let mut out = [Level::default(); 128];
    let n = s.book.levels(side, &mut out);
    out[..n].to_vec()
}

fn lob_config() -> impl Strategy<Value = LobConfig> {
    (2..=6u32, 0..=3u32, 1..=8u32, 300..=5000i64, 0..=4u32).prop_map(
        |(agents, latency, pos, cash, noise)| LobConfig {
            max_position: pos,
            initial_cash: cash,
            noise,
            steps: 60,
            ..config(agents, latency)
        },
    )
}

/// Every sample of every alive agent of `s`.
fn check_samples(cfg: &LobConfig, s: &LobState, seed: u64) -> Result<(), TestCaseError> {
    for me in 0..cfg.agent_count() {
        if !s.is_alive(me) {
            continue;
        }
        let agent = AgentSlot::new(u16::try_from(me).unwrap(), MAX_AGENTS).unwrap();
        let info = LimitOrderBook::info_state(cfg, s, agent);
        for k in 0..SAMPLES {
            let mut sample = LobState::default();
            LimitOrderBook::sample(cfg, &info, &mut rng(seed ^ k, 7), &mut sample);
            prop_assert_eq!(LimitOrderBook::validate(cfg, &sample), Ok(()));
            let again = LimitOrderBook::info_state(cfg, &sample, agent);
            prop_assert_eq!(encode_to_vec(&again), encode_to_vec(&info));
            if let Err(e) = agent_limits(cfg, &sample) {
                return Err(TestCaseError::fail(e));
            }
            prop_assert_eq!(levels(&sample, Side::Buy), levels(s, Side::Buy));
            prop_assert_eq!(levels(&sample, Side::Sell), levels(s, Side::Sell));
            prop_assert_eq!(sample.accounts[me], s.accounts[me]);
            let noise = i64::from(cfg.noise);
            let band = i64::from(s.signal[me]) - i64::from(sample.fundamental);
            prop_assert!(band.abs() <= noise);
            // Lots are conserved when nobody died.
            if (0..cfg.agent_count()).all(|j| s.is_alive(j)) {
                let total: i64 = sample.accounts.iter().map(|a| i64::from(a.inventory)).sum();
                prop_assert_eq!(total, 0);
            }
            // The baselines read the information state only.
            for b in Baseline::ALL {
                let act = |state: &LobState| {
                    let mut h = [0i32; HEADS];
                    let info = LimitOrderBook::info_state(cfg, state, agent);
                    act_info(
                        cfg,
                        &info,
                        b,
                        &mut rng(seed, 9),
                        &mut ActionWriter::new(&mut h, &mut []),
                    );
                    h
                };
                prop_assert_eq!(act(&sample), act(s));
            }
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn test_determinize_samples_are_valid_consistent_and_within_limits(
        cfg in lob_config(),
        seed in any::<u64>(),
        draws in any::<u64>(),
        steps in 1..50u32,
    ) {
        let mut p = pool(cfg.clone(), INSTANCES, seed, None);
        let mut masks = p.reset_all(Reseed::Keep).unwrap().masks().to_vec();
        let mut r = Rng(draws | 1);
        for _ in 0..steps {
            let a = legal_actions(&mut r, &masks, INSTANCES as usize);
            masks = p.step(input(&a)).unwrap().masks().to_vec();
        }
        for i in 0..INSTANCES {
            let s = *p.state(i).unwrap();
            if let Err(e) = agent_limits(&cfg, &s) {
                return Err(TestCaseError::fail(format!("the played state itself: {e}")));
            }
            check_samples(&cfg, &s, seed ^ u64::from(i))?;
        }
    }
}
