// SPDX-License-Identifier: MIT OR Apache-2.0

//! Search smoke test (design section 9, row 3). `tessera-rl-search` takes
//! a single `Discrete` action head, and this environment's action is
//! `MultiDiscrete([6, 9, 4, 8])`, so a search pool is refused with
//! `UnsupportedActions` rather than built: before that check, plain
//! search is refused for imperfect information and determinized search
//! is accepted as far as the action check. The observation's `Set` leaves
//! (ADR-0082) would be refused next (`SetObservation`): the search pool
//! checks the action space first, so `UnsupportedActions` is the refusal
//! seen. The determinized root the
//! search would build is checked by hand here, the way its sampler does
//! it: a sample from the sampler's identity passes `Validate` and keeps
//! the searching agent's information state.

#![expect(
    clippy::unwrap_used,
    reason = "test fixtures: a failure is the assertion"
)]

mod support;

use core::num::NonZeroU32;

use support::{Rng, config, input, legal_actions, pool};
use tessera_rl_core::encoding::encode_to_vec;
use tessera_rl_core::{AgentSlot, Determinize, Environment, ImperfectInformation, Validate};
use tessera_rl_engine::{EnvPool, Masked, Reseed};
use tessera_rl_markets::limits::MAX_AGENTS;
use tessera_rl_markets::{LimitOrderBook, LobState};
use tessera_rl_num::identity::{Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_search::{
    Determinization, SearchError, SearchOptions, SearchPool, sampler_identity,
};

type Search = SearchPool<LimitOrderBook, Masked>;

fn options() -> SearchOptions {
    SearchOptions::new(NonZeroU32::new(64).unwrap(), NonZeroU32::new(4).unwrap())
}

#[test]
fn test_search_smoke_refuses_multi_discrete_after_information_checks() {
    let cfg = config(4, 2);
    let mut p = pool(cfg.clone(), 1, 11, None);
    let mut masks = p.reset_all(Reseed::Keep).unwrap().masks().to_vec();
    let mut rng = Rng(0x5ea1);
    for _ in 0..12 {
        let a = legal_actions(&mut rng, &masks, 1);
        masks = p.step(input(&a)).unwrap().masks().to_vec();
    }
    let root = p.snapshot(0).unwrap();
    let agent = AgentSlot::new(1, MAX_AGENTS).unwrap();
    let how = Determinization {
        agent,
        sampler_seed: Seed::new(3),
        sample_index: 0,
    };

    let plain = Search::new(cfg.clone(), options(), &root, EnvPool::with_masks);
    assert!(matches!(plain, Err(SearchError::ImperfectInformation)));
    let det = Search::new_determinized(cfg.clone(), options(), &root, how, EnvPool::with_masks);
    assert!(matches!(det, Err(SearchError::UnsupportedActions)));
    // The refusal that would follow the action check.
    let space = LimitOrderBook::observation_space(&cfg).unwrap();
    assert!(space.contains_set());

    // The sample the determinized root would take.
    let info = LimitOrderBook::info_state(&cfg, root.state(), agent);
    let id = sampler_identity(how.sampler_seed, how.sample_index);
    let mut r = StepRng::new(&RngAuthority::engine_only(), &id, StreamId::DYNAMICS);
    let mut sample = LobState::default();
    LimitOrderBook::sample(&cfg, &info, &mut r, &mut sample);
    assert_eq!(LimitOrderBook::validate(&cfg, &sample), Ok(()));
    assert_eq!(
        encode_to_vec(&LimitOrderBook::info_state(&cfg, &sample, agent)),
        encode_to_vec(&info)
    );
}
