// SPDX-License-Identifier: MIT OR Apache-2.0

//! Golden suites (ADR-0009, ADR-0024 layer 2) and cross-thread runs (SPEC
//! 8.5) for Survival-v1, plus the map golden of STAGE1-r1: map digests for
//! fixed seeds and sizes, connectivity pass included, so a change in map
//! generation shows there first. The classic crate's scenario driver needs
//! a one-head `Heuristic` (stage 3 brings Survival's), so this file drives
//! the pool itself with a fixed action sequence and uses the classic
//! fixture check and bless policy. Continuation (SPEC 9.1) and the
//! persisted record bytes are checked here too.

#![expect(clippy::panic, reason = "test setup: a failure is the assertion")]
#![expect(clippy::many_single_char_names, reason = "three pools, a, b and c")]

#[path = "../../tessera-rl-classic/tests/support/mod.rs"]
mod support;

use core::num::NonZeroU32;

use core::num::NonZeroUsize;

use support::check;
use tessera_rl_core::encoding::{checksum, decode_from_slice};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EnvPool, NoMasks, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_world::survival::{ENV_ID, ENV_VERSION, MAX_AGENTS};
use tessera_rl_world::{Survival, SurvivalConfig};

const WORKERS: [Option<usize>; 5] = [None, Some(1), Some(2), Some(7), Some(64)];

/// A golden scenario: its config, auto-reset mode and step count.
struct Scenario {
    name: &'static str,
    cfg: SurvivalConfig,
    auto_reset: AutoReset,
    steps: u32,
}

/// Actions `lcg-v1` for step `t`: per instance and agent slot, a move and
/// an act from a fixed linear congruential sequence keyed by the row.
fn actions(t: u32, instances: usize) -> Vec<i32> {
    (0..instances * MAX_AGENTS)
        .flat_map(|row| {
            let key = (u64::from(t) << 32) ^ u64::try_from(row).unwrap_or(0);
            let x = key
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            [
                i32::try_from((x >> 33) % 5).unwrap_or(0),
                i32::try_from((x >> 45) % 7).unwrap_or(0),
            ]
        })
        .collect()
}

const INSTANCES: u32 = 8;

fn pool(s: &Scenario, workers: Option<usize>) -> EnvPool<Survival, NoMasks> {
    let options = PoolOptions {
        auto_reset: s.auto_reset,
        chunk_size: NonZeroU32::new(2).unwrap_or(NonZeroU32::MIN),
        ..PoolOptions::new(
            NonZeroU32::new(INSTANCES).unwrap_or(NonZeroU32::MIN),
            Seed::new(3_820_001),
        )
    };
    let p =
        EnvPool::<Survival, NoMasks>::new(s.cfg.clone(), options).unwrap_or_else(|e| panic!("{e}"));
    match workers {
        None => p,
        Some(w) => p
            .with_workers(SyncOptions::new(
                NonZeroUsize::new(w).unwrap_or(NonZeroUsize::MIN),
            ))
            .unwrap_or_else(|e| panic!("workers: {e}")),
    }
}

/// One line per step: every instance's record hash, then a digest of the
/// rewards and episode flags.
fn run(s: &Scenario, workers: Option<usize>) -> Vec<String> {
    let mut p = pool(s, workers);
    let mut lines = vec![
        format!("env_id={ENV_ID}"),
        format!("env_version={ENV_VERSION}"),
        "format=1".to_owned(),
        "actions=lcg-v1".to_owned(),
        "outputs=rewards+observations+flags".to_owned(),
        format!("config={:?}", s.cfg),
        format!("auto_reset={:?}", s.auto_reset),
    ];
    p.reset_all(Reseed::Keep)
        .unwrap_or_else(|e| panic!("reset: {e}"));
    for t in 1..=s.steps {
        let acts = actions(t, INSTANCES as usize);
        let out = p
            .step(ActionInput {
                discrete: &acts,
                continuous: &[],
            })
            .unwrap_or_else(|e| panic!("step {t}: {e}"));
        let mut bytes: Vec<u8> = out
            .rewards()
            .iter()
            .flat_map(|r| r.to_bits().to_le_bytes())
            .collect();
        // Every observation leaf (window, own, the set's items and its mask).
        let mut leaf = 0;
        while let Some(obs) = out.observation(leaf) {
            bytes.extend_from_slice(obs);
            leaf += 1;
        }
        bytes.extend(out.episode_terminated().iter().map(|f| u8::from(f.get())));
        bytes.extend(out.episode_truncated().iter().map(|f| u8::from(f.get())));
        let outputs = checksum(&bytes);
        let hashes: Vec<String> = (0..INSTANCES)
            .map(|i| {
                format!(
                    "{:016x}",
                    p.snapshot(i).unwrap_or_else(|e| panic!("{e}")).hash().get()
                )
            })
            .collect();
        lines.push(format!("{t} {outputs:016x} {}", hashes.join(" ")));
    }
    lines
}

fn check_survival(s: &Scenario) {
    for workers in WORKERS {
        check(s.name, &run(s, workers));
    }
}

/// Short episodes and quick respawn, so deaths, respawns and auto-reset all
/// happen within the scenario.
fn small() -> SurvivalConfig {
    SurvivalConfig {
        agents: 16,
        size: 32,
        steps: 64,
        respawn: 2,
        ..SurvivalConfig::default()
    }
}

#[test]
fn test_golden_survival_16_next_step_matches_fixture_on_every_mode() {
    check_survival(&Scenario {
        name: "survival-v1-16-next-step",
        cfg: small(),
        auto_reset: AutoReset::NextStep,
        steps: 200,
    });
}

#[test]
fn test_golden_survival_16_same_step_matches_fixture_on_every_mode() {
    check_survival(&Scenario {
        name: "survival-v1-16-same-step",
        cfg: small(),
        auto_reset: AutoReset::SameStep,
        steps: 200,
    });
}

#[test]
fn test_golden_survival_128_next_step_matches_fixture_on_every_mode() {
    let cfg = SurvivalConfig {
        agents: 128,
        size: 96,
        steps: 64,
        ..SurvivalConfig::default()
    };
    check_survival(&Scenario {
        name: "survival-v1-128-next-step",
        cfg,
        auto_reset: AutoReset::NextStep,
        steps: 100,
    });
}

/// Continuation (SPEC 9.1): a snapshot taken mid-episode, restored into a
/// fresh pool, continues exactly as the original; the same through the
/// persisted bytes of the record.
#[test]
fn test_golden_survival_snapshot_and_bytes_continue_exactly() {
    let s = Scenario {
        name: "unused",
        cfg: small(),
        auto_reset: AutoReset::NextStep,
        steps: 0,
    };
    let mut a = pool(&s, None);
    a.reset_all(Reseed::Keep).unwrap_or_else(|e| panic!("{e}"));
    for t in 1..=30 {
        a.step(ActionInput {
            discrete: &actions(t, INSTANCES as usize),
            continuous: &[],
        })
        .unwrap_or_else(|e| panic!("{e}"));
    }
    let records: Vec<_> = (0..INSTANCES)
        .map(|i| a.snapshot(i).unwrap_or_else(|e| panic!("{e}")))
        .collect();
    let mut b = pool(&s, Some(3));
    b.reset_all(Reseed::Keep).unwrap_or_else(|e| panic!("{e}"));
    for (i, r) in records.iter().enumerate() {
        let id = u32::try_from(i).unwrap_or(0);
        b.restore(id, r).unwrap_or_else(|e| panic!("{e}"));
    }
    // Through bytes: decode and import into a third pool.
    let mut c = pool(&s, None);
    c.reset_all(Reseed::Keep).unwrap_or_else(|e| panic!("{e}"));
    let ids: Vec<u32> = (0..INSTANCES).collect();
    let decoded: Vec<DecodedRecord<Survival>> = records
        .iter()
        .map(|r| decode_from_slice(&r.to_bytes()).unwrap_or_else(|e| panic!("{e}")))
        .collect();
    c.import(&ids, &decoded).unwrap_or_else(|e| panic!("{e}"));
    for t in 31..=90 {
        let acts = actions(t, INSTANCES as usize);
        let outs: Vec<Vec<f32>> = [&mut a, &mut b, &mut c]
            .into_iter()
            .map(|p| {
                p.step(ActionInput {
                    discrete: &acts,
                    continuous: &[],
                })
                .unwrap_or_else(|e| panic!("{e}"))
                .rewards()
                .to_vec()
            })
            .collect();
        assert_eq!(outs[0], outs[1], "restore, step {t}");
        assert_eq!(outs[0], outs[2], "bytes, step {t}");
        for i in 0..INSTANCES {
            let h = |p: &EnvPool<Survival, NoMasks>| {
                p.snapshot(i).unwrap_or_else(|e| panic!("{e}")).hash()
            };
            assert_eq!(h(&a), h(&b), "restore, step {t}, instance {i}");
            assert_eq!(h(&a), h(&c), "bytes, step {t}, instance {i}");
        }
    }
}

#[test]
fn test_golden_survival_map_digests_match_fixture() {
    let mut lines = vec![
        format!("env_id={ENV_ID}"),
        format!("env_version={ENV_VERSION}"),
        "format=1".to_owned(),
    ];
    for size in [32u32, 64, 96] {
        let cfg = SurvivalConfig {
            size,
            ..SurvivalConfig::default()
        };
        for seed in [1u64, 2, 3, 382] {
            let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(seed));
            let mut p = EnvPool::<Survival, NoMasks>::new(cfg.clone(), options)
                .unwrap_or_else(|e| panic!("{e}"));
            p.reset_all(Reseed::Keep).unwrap_or_else(|e| panic!("{e}"));
            let s = p.state(0).unwrap_or_else(|| panic!("no state"));
            let bytes: Vec<u8> = s.tiles.iter().flatten().copied().collect();
            lines.push(format!(
                "size={size} seed={seed} tiles={:016x}",
                checksum(&bytes)
            ));
        }
    }
    check("survival-v1-map", &lines);
}
