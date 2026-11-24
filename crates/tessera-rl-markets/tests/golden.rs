// SPDX-License-Identifier: MIT OR Apache-2.0

//! The golden suite of `tessera/LimitOrderBook-v1` (ADR-0023; design
//! section 2): fixed seeds, 2, 8 and 16 agents, latency 0 and 2, actions
//! drawn uniformly from the masks; each run folds the rewards, the episode
//! flags, every mask row, every observation leaf and each instance's state
//! hash of every step. The fold must be identical inline and with 1 and 3
//! workers, and equal to the pinned value of its scenario.

#![expect(
    clippy::unwrap_used,
    reason = "test fixtures: a failure is the assertion"
)]

mod support;

use support::{FNV_START, Rng, config, fnv, input, legal_actions, pool};
use tessera_rl_engine::{MAX_LEAVES, Reseed};

const INSTANCES: u32 = 2;
const STEPS: u32 = 120;

/// `(agents, latency, pinned fold)`. Pinned on the first run of this
/// suite; any change to the rules, the masks, the observation or the
/// state encoding moves them.
const GOLDEN: [(u32, u32, u64); 6] = [
    (2, 0, 9_414_531_740_013_021_448),
    (2, 2, 10_654_068_319_335_762_469),
    (8, 0, 2_322_089_580_107_650_509),
    (8, 2, 4_262_903_442_396_779_717),
    (16, 0, 13_378_443_242_271_823_145),
    (16, 2, 4_615_897_759_708_919_202),
];

fn run(agents: u32, latency: u32, workers: Option<usize>) -> u64 {
    // A short episode, so the run crosses settlement and auto-resets.
    let cfg = tessera_rl_markets::LobConfig {
        steps: 50,
        ..config(agents, latency)
    };
    let mut p = pool(cfg, INSTANCES, 2026, workers);
    let mut rng = Rng(0x5eed_0381);
    let mut masks = p.reset_all(Reseed::Keep).unwrap().masks().to_vec();
    let mut hash = FNV_START;
    let n = usize::try_from(INSTANCES).unwrap();
    for _ in 0..STEPS {
        let a = legal_actions(&mut rng, &masks, n);
        let out = p.step(input(&a)).unwrap();
        for r in out.rewards() {
            hash = fnv(hash, u64::from(r.to_bits()));
        }
        for f in out
            .episode_terminated()
            .iter()
            .chain(out.episode_truncated())
            .chain(out.alive())
        {
            hash = fnv(hash, u64::from(f.get()));
        }
        masks = out.masks().to_vec();
        hash = masks.iter().fold(hash, |h, m| fnv(h, u64::from(m.get())));
        for leaf in 0..MAX_LEAVES {
            let Some(bytes) = out.observation(leaf) else {
                break;
            };
            hash = bytes.iter().fold(hash, |h, b| fnv(h, u64::from(*b)));
        }
        for i in 0..INSTANCES {
            hash = fnv(hash, p.state_hash(i).unwrap().get());
        }
    }
    hash
}

#[test]
fn test_golden_same_fold_inline_and_with_workers() {
    for (agents, latency, _) in GOLDEN {
        let inline = run(agents, latency, None);
        assert_eq!(
            run(agents, latency, Some(1)),
            inline,
            "{agents} agents, latency {latency}"
        );
        assert_eq!(
            run(agents, latency, Some(3)),
            inline,
            "{agents} agents, latency {latency}"
        );
    }
}

#[test]
fn test_golden_folds_match_pinned_values() {
    let got: Vec<(u32, u32, u64)> = GOLDEN
        .iter()
        .map(|&(agents, latency, _)| (agents, latency, run(agents, latency, None)))
        .collect();
    assert_eq!(got, GOLDEN, "pin the values printed here");
}
