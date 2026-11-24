// SPDX-License-Identifier: MIT OR Apache-2.0

//! The draw table and its rounding rule (ADR-0088 item 5).

use proptest::prelude::*;
use tessera_rl_league::{LeagueError, Matchmaking, PfspWeighting, PolicyId, Table};

#[test]
fn test_table_uniform_halves_the_range() {
    let t = Table::from_weights(&[Some(1.0), Some(1.0)]).unwrap();
    assert_eq!(t.thresholds(), [1u64 << 63, u64::MAX]);
    assert_eq!(t.draw(0), Some(PolicyId(0)));
    assert_eq!(t.draw((1u64 << 63) - 1), Some(PolicyId(0)));
    assert_eq!(t.draw(1u64 << 63), Some(PolicyId(1)));
    assert_eq!(t.draw(u64::MAX), Some(PolicyId(1)));
}

#[test]
fn test_table_zero_weight_and_retired_entries_are_never_drawn() {
    let t = Table::from_weights(&[Some(0.0), None, Some(3.0), Some(1.0), None]).unwrap();
    assert_eq!(t.thresholds(), [0, 0, 3u64 << 62, u64::MAX, u64::MAX]);
    for u in [0, 1, (3u64 << 62) - 1] {
        assert_eq!(t.draw(u), Some(PolicyId(2)));
    }
    for u in [3u64 << 62, u64::MAX] {
        assert_eq!(t.draw(u), Some(PolicyId(3)));
    }
}

#[test]
fn test_table_all_zero_weights_fall_back_to_uniform_over_active_entries() {
    let t = Table::from_weights(&[Some(0.0), None, Some(0.0)]).unwrap();
    assert!(t.uniform_fallback());
    assert_eq!(t.draw(0), Some(PolicyId(0)));
    assert_eq!(t.draw(u64::MAX), Some(PolicyId(2)));
    assert_eq!(
        Table::from_weights(&[None, None]),
        Err(LeagueError::NoActivePolicy)
    );
    assert_eq!(Table::from_weights(&[]), Err(LeagueError::NoActivePolicy));
}

#[test]
fn test_pfsp_weightings_match_their_formulas() {
    assert_eq!(PfspWeighting::Hard.weight(0.25), 0.5625);
    assert_eq!(PfspWeighting::Variance.weight(0.25), 0.1875);
    assert_eq!(PfspWeighting::Linear.weight(0.25), 0.75);
    assert_eq!(Matchmaking::Uniform.weight(0.9, 0.0, 1.0e3), 1.0);
    assert_eq!(
        Matchmaking::Elo { temperature: 100.0 }.weight(0.5, 1300.0, 1300.0),
        1.0
    );
}

proptest! {
    #[test]
    fn test_table_draws_only_positive_active_entries(
        weights in proptest::collection::vec(proptest::option::of(0.0f64..10.0), 1..40),
        u in any::<u64>(),
    ) {
        let Ok(t) = Table::from_weights(&weights) else {
            prop_assert!(weights.iter().all(Option::is_none));
            return Ok(());
        };
        let th = t.thresholds();
        prop_assert!(th.windows(2).all(|w| w[0] <= w[1]));
        let id = t.draw(u).unwrap();
        let w = weights[id.index()];
        prop_assert!(w.is_some());
        if !t.uniform_fallback() {
            prop_assert!(w.unwrap() > 0.0);
        }
    }
}

#[test]
fn test_table_negative_and_nan_weights_weigh_zero_without_fallback() {
    // A negative weight must not shrink the total: entry 1 keeps a third.
    let t = Table::from_weights(&[Some(-1.0), Some(f64::NAN), Some(1.0), Some(2.0)]).unwrap();
    assert!(!t.uniform_fallback());
    // floor((1/3 as f64) * 2^64), the rounding rule's value.
    let third = 6_148_914_691_236_516_864_u64;
    assert_eq!(t.thresholds(), [0, 0, third, u64::MAX]);
    assert_eq!(t.draw(0), Some(PolicyId(2)));
    assert_eq!(t.draw(third), Some(PolicyId(3)));
}
