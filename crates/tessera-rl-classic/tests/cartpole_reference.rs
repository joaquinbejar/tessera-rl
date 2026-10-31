// SPDX-License-Identifier: MIT OR Apache-2.0

//! CartPole against reference vectors from `tools/cartpole_reference.py`
//! (Gymnasium's `CartPole-v1` formulas in Python floats): one-step
//! equivalence within 4 ulps or 1e-12 relative (SPEC 18), and short
//! trajectories with their observed divergence.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test parsing: a failure is the assertion"
)]

use tessera_rl_classic::cartpole::physics::{THETA_THRESHOLD, X_THRESHOLD, is_terminal, step};

const DATA: &str = include_str!("data/cartpole_reference.txt");

fn f(word: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(word, 16).unwrap())
}

fn state(words: &[&str]) -> [f64; 4] {
    [f(words[0]), f(words[1]), f(words[2]), f(words[3])]
}

/// Distance in units in the last place, for finite values.
fn ulps(a: f64, b: f64) -> u64 {
    let key = |v: f64| {
        let bits = v.to_bits().cast_signed();
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    key(a).abs_diff(key(b))
}

fn close(a: f64, b: f64) -> bool {
    ulps(a, b) <= 4 || (a - b).abs() <= 1e-12 * b.abs()
}

fn near_threshold(s: [f64; 4]) -> bool {
    (s[0].abs() - X_THRESHOLD).abs() < 1e-9 || (s[2].abs() - THETA_THRESHOLD).abs() < 1e-9
}

#[test]
fn test_cartpole_one_step_matches_reference_within_tolerance() {
    let mut checked = 0;
    for line in DATA.lines().filter(|l| l.starts_with("step ")) {
        let w: Vec<&str> = line.split(' ').collect();
        let (s, action, expected) = (state(&w[1..5]), w[5] == "1", state(&w[6..10]));
        let got = step(s, action);
        for k in 0..4 {
            assert!(
                close(got[k], expected[k]),
                "component {k} of {line}: {} vs {} ({} ulps)",
                got[k],
                expected[k],
                ulps(got[k], expected[k])
            );
        }
        if !near_threshold(expected) {
            assert_eq!(is_terminal(got), w[10] == "1", "{line}");
        }
        checked += 1;
    }
    assert_eq!(checked, 1000);
}

/// Open-loop trajectories amplify one-ulp differences, so they are not
/// gated at the one-step tolerance (SPEC 18); this bounds the observed
/// relative divergence over 50 steps.
#[test]
fn test_cartpole_short_trajectories_stay_close_to_reference() {
    let mut episodes = 0;
    let mut worst = 0.0_f64;
    let mut s = [0.0; 4];
    for line in DATA.lines() {
        let w: Vec<&str> = line.split(' ').collect();
        match w[0] {
            "episode" => {
                s = state(&w[1..5]);
                episodes += 1;
            }
            "t" => {
                s = step(s, w[1] == "1");
                let expected = state(&w[2..6]);
                for k in 0..4 {
                    let scale = expected[k].abs().max(1e-300);
                    worst = worst.max((s[k] - expected[k]).abs() / scale);
                }
                if !near_threshold(expected) {
                    assert_eq!(is_terminal(s), w[6] == "1", "{line}");
                }
            }
            _ => {}
        }
    }
    assert_eq!(episodes, 20);
    assert!(worst <= 1e-9, "relative divergence {worst:e}");
}
