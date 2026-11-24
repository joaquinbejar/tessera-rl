// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Determinize` and the information boundary (ADR-0035, ADR-0046; review
//! r1 of #381, correction 3; PR 3 review). On states reached by masked
//! play:
//!
//! - changing only hidden data (another agent's queued items, the owners
//!   and split of other agents' orders, orders beyond the visible depth,
//!   the fundamental within the observer's band, other accounts and
//!   signals) leaves the observer's information state unchanged;
//! - every sample for every alive agent passes `Validate`, has the
//!   agent's information state exactly, reproduces the visible book (the
//!   best `depth` levels per side) exactly, keeps the observer's own part,
//!   and gives every agent at most `MAX_OPEN` resting plus pending limit
//!   orders and positions within `max_position` counting open and pending
//!   lots;
//! - the heuristics read the information state only: on a sample they act
//!   as on the original.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "property tests: a failure is the assertion"
)]

mod support;

use proptest::prelude::*;
use support::{Rng, agent_limits, config, input, legal_actions, pool};
use tessera_rl_core::Reserved;
use tessera_rl_core::encoding::encode_to_vec;
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, ImperfectInformation, Validate};
use tessera_rl_engine::Reseed;
use tessera_rl_markets::actions::HEADS;
use tessera_rl_markets::book::Level;
use tessera_rl_markets::limits::MAX_AGENTS;
use tessera_rl_markets::{
    Baseline, Book, LimitOrderBook, LobConfig, LobState, Order, Pending, Side, act_info,
    sample_with_report,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

const INSTANCES: u32 = 2;
const SAMPLES: u64 = 6;

fn rng(seed: u64, stream: u32) -> StepRng {
    let id = RngIdentity::new(Seed::new(seed), LogicalId::new(stream));
    StepRng::new(&RngAuthority::engine_only(), &id, StreamId::DYNAMICS)
}

/// The visible levels of `side`: the best `depth`.
fn levels(cfg: &LobConfig, s: &LobState, side: Side) -> Vec<Level> {
    let mut out = vec![Level::default(); usize::try_from(cfg.depth).unwrap()];
    let n = s.book.levels(side, &mut out);
    out.truncate(n);
    out
}

/// `s` with every kind of hidden data changed for observer `me`; `None`
/// when `s` has no other alive agent to change.
fn perturbed(cfg: &LobConfig, s: &LobState, me: usize) -> Option<LobState> {
    let other = (0..cfg.agent_count()).find(|&j| j != me && s.is_alive(j))?;
    let other8 = u8::try_from(other).unwrap();
    let me8 = u8::try_from(me).unwrap();
    let mut t = *s;
    // Other agents' orders: all owned by one other agent (owners and
    // grouping change; prices, sizes and ids stay).
    t.book
        .set_owners(|o| if o.owner == me8 { me8 } else { other8 });
    // Another agent's queued item.
    let ready = t.step + 1;
    let _ = t
        .pending
        .push(Pending::limit(other8, Side::Buy, 1, 1), ready);
    // An order beyond the visible depth: a bid below every visible one,
    // only when the visible levels fill the depth.
    let bids = levels(cfg, s, Side::Buy);
    if bids.len() == usize::try_from(cfg.depth).unwrap()
        && let Some(worst) = bids.last().map(|l| l.price).filter(|&p| p > 1)
    {
        let mut orders: Vec<Order> = t.book.orders().copied().collect();
        orders.push(Order {
            price: worst - 1,
            qty: 1,
            seq: t.book.next_seq(),
            owner: other8,
            side: Side::Buy,
            pad: Reserved::ZERO,
        });
        t.book = Book::from_orders(orders, t.book.next_seq() + 1).unwrap();
    }
    // The fundamental, still within the observer's band.
    let band = i64::from(s.signal[me]) - i64::from(cfg.noise);
    t.fundamental = u32::try_from(band.max(i64::from(cfg.min_tick))).unwrap();
    // Another account, signal and last equity.
    t.accounts[other].cash += 7;
    t.accounts[other].inventory = -t.accounts[other].inventory;
    t.signal[other] += 1;
    t.last_equity[other] -= 3;
    Some(t)
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
/// Returns how many samples needed the last resort.
fn check_samples(cfg: &LobConfig, s: &LobState, seed: u64) -> Result<usize, TestCaseError> {
    let mut last_resort = 0;
    for me in 0..cfg.agent_count() {
        if !s.is_alive(me) {
            continue;
        }
        let agent = AgentSlot::new(u16::try_from(me).unwrap(), MAX_AGENTS).unwrap();
        let info = LimitOrderBook::info_state(cfg, s, agent);
        if let Some(t) = perturbed(cfg, s, me) {
            let hidden = LimitOrderBook::info_state(cfg, &t, agent);
            prop_assert_eq!(encode_to_vec(&hidden), encode_to_vec(&info));
        }
        for k in 0..SAMPLES {
            let mut sample = LobState::default();
            let report = sample_with_report(cfg, &info, &mut rng(seed ^ k, 7), &mut sample);
            last_resort += usize::from(report.last_resort);
            prop_assert_eq!(LimitOrderBook::validate(cfg, &sample), Ok(()));
            let again = LimitOrderBook::info_state(cfg, &sample, agent);
            prop_assert_eq!(encode_to_vec(&again), encode_to_vec(&info));
            if let Err(e) = agent_limits(cfg, &sample) {
                return Err(TestCaseError::fail(e));
            }
            prop_assert_eq!(levels(cfg, &sample, Side::Buy), levels(cfg, s, Side::Buy));
            prop_assert_eq!(levels(cfg, &sample, Side::Sell), levels(cfg, s, Side::Sell));
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
    Ok(last_resort)
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
        let mut last_resort = 0;
        for i in 0..INSTANCES {
            let s = p.state(i).unwrap();
            if let Err(e) = agent_limits(&cfg, &s) {
                return Err(TestCaseError::fail(format!("the played state itself: {e}")));
            }
            last_resort += check_samples(&cfg, &s, seed ^ u64::from(i))?;
        }
        // Review of PR 3, follow-up 1: the last resort is never needed.
        prop_assert_eq!(last_resort, 0);
    }
}
