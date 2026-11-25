// SPDX-License-Identifier: MIT OR Apache-2.0

//! The engine's ONNX opponent seam (ADR-0088 item 2.3, erratum 2; #384
//! head C) with a Rust stand-in for the model: the engine's actions equal
//! the caller submitting the same function of the same rows; a mixed
//! population is bit identical at every worker count; an out-of-range
//! action poisons; the setter's refusals. The real runtime is tested in
//! `tessera-rl-onnx` (`make onnx-test`).

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_classic::{PaddleDuel, PaddleDuelConfig};
use tessera_rl_engine::league::{OpponentPolicies, OpponentPolicyError, sealed};
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, EnvPool, NoMasks, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_league::trueskill::Rating;
use tessera_rl_league::{
    League, LeagueConfig, Matchmaking, PolicyKind, PoolLeague, Population, elo,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 8;
const A: usize = 2;
const HASH: [u8; 16] = [7; 16];

/// The stand-in model: the action is a byte of the agent's row, mod 3, or
/// `bad` for every row.
struct ByteRule {
    bad: Option<i64>,
}

impl sealed::Sealed for ByteRule {}

impl OpponentPolicies for ByteRule {
    fn serves(&self, content_hash: &[u8; 16]) -> bool {
        *content_hash == HASH
    }

    fn act(
        &mut self,
        _content_hash: &[u8; 16],
        rows: &[u8],
        n: usize,
        actions: &mut [i64],
    ) -> Result<(), OpponentPolicyError> {
        let row = rows.len() / n;
        for (j, a) in actions.iter_mut().enumerate() {
            *a = self
                .bad
                .unwrap_or_else(|| rule(rows.get(j * row..(j + 1) * row).unwrap()));
        }
        Ok(())
    }
}

fn rule(row: &[u8]) -> i64 {
    i64::from(
        row.iter()
            .fold(0u8, |h, &b| h.wrapping_mul(31).wrapping_add(b))
            % 3,
    )
}

fn pool(kinds: &[PolicyKind], workers: Option<usize>) -> EnvPool<PaddleDuel, NoMasks> {
    let mut population = Population::new();
    for &k in kinds {
        population
            .add(k, elo::INITIAL, Rating::initial(), 0)
            .unwrap();
    }
    let config = LeagueConfig {
        matchmaking: Matchmaking::Uniform,
        ..LeagueConfig::default()
    };
    let league = PoolLeague::new(League::new(config, population).unwrap()).unwrap();
    let view = league.view().unwrap();
    let options = PoolOptions {
        auto_reset: AutoReset::NextStep,
        max_episode_steps: NonZeroU32::new(5),
        chunk_size: NonZeroU32::new(2).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(11))
    };
    let cfg = PaddleDuelConfig {
        agents: 2,
        ..PaddleDuelConfig::default()
    };
    let p = EnvPool::<PaddleDuel, NoMasks>::new(cfg, options)
        .unwrap()
        .with_league(&[1], view, Box::new(league))
        .unwrap();
    match workers {
        Some(w) => p
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap(),
        None => p,
    }
}

/// One agent's row: its exported leaves, concatenated.
fn row_of(p: &EnvPool<PaddleDuel, NoMasks>, k: usize) -> Vec<u8> {
    let out = p.outputs().unwrap();
    let mut row = Vec::new();
    for l in 0..tessera_rl_engine::MAX_LEAVES {
        let Some(leaf) = out.observation(l) else {
            break;
        };
        let b = leaf.len() / (N as usize * A);
        row.extend_from_slice(leaf.get(k * b..(k + 1) * b).unwrap());
    }
    row
}

fn trace(p: &EnvPool<PaddleDuel, NoMasks>) -> Vec<u8> {
    let out = p.outputs().unwrap();
    let mut t = row_of(p, 0);
    t.clear();
    for l in 0..tessera_rl_engine::MAX_LEAVES {
        let Some(leaf) = out.observation(l) else {
            break;
        };
        t.extend_from_slice(leaf);
    }
    t.extend(out.rewards().iter().flat_map(|r| r.to_bits().to_le_bytes()));
    t
}

/// The learner's actions at step `t`: a fixed pattern.
fn learner(t: usize, i: usize) -> i32 {
    i32::try_from((t + i) % 3).unwrap()
}

#[test]
fn test_onnx_opponents_equal_the_caller_submitting_the_same_rule() {
    let mut onnx = pool(&[PolicyKind::Onnx { content_hash: HASH }], None)
        .with_onnx_opponents(Box::new(ByteRule { bad: None }))
        .unwrap();
    let mut external = pool(&[PolicyKind::External { tag: 3 }], None);
    onnx.reset_all(Reseed::Keep).unwrap();
    external.reset_all(Reseed::Keep).unwrap();
    let mut nonzero = 0;
    for t in 0..20 {
        assert_eq!(trace(&onnx), trace(&external), "step {t}");
        let mut a = vec![0i32; N as usize * A];
        let mut b = vec![0i32; N as usize * A];
        for i in 0..N as usize {
            a[i * A] = learner(t, i);
            b[i * A] = learner(t, i);
            b[i * A + 1] = i32::try_from(rule(&row_of(&external, i * A + 1))).unwrap();
            nonzero += usize::from(b[i * A + 1] != 0);
        }
        onnx.step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
        external
            .step(ActionInput {
                discrete: &b,
                continuous: &[],
            })
            .unwrap();
    }
    // The rule is not the all-zero default rows.
    assert!(nonzero > 0);
}

#[test]
fn test_onnx_mixed_population_is_bit_identical_at_every_worker_count() {
    let kinds = [
        PolicyKind::Heuristic,
        PolicyKind::External { tag: 3 },
        PolicyKind::Onnx { content_hash: HASH },
    ];
    let run = |workers| {
        let mut p = pool(&kinds, workers)
            .with_onnx_opponents(Box::new(ByteRule { bad: None }))
            .unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        let mut out = Vec::new();
        for t in 0..30 {
            let a: Vec<i32> = (0..N as usize * A).map(|k| learner(t, k)).collect();
            p.step(ActionInput {
                discrete: &a,
                continuous: &[],
            })
            .unwrap();
            out.extend(trace(&p));
            out.extend(
                p.assignment()
                    .into_iter()
                    .flatten()
                    .flat_map(|x| x.to_le_bytes()),
            );
        }
        out
    };
    let reference = run(None);
    for w in [1, 2, 4] {
        assert_eq!(run(Some(w)), reference, "{w} workers");
    }
}

#[test]
fn test_onnx_out_of_range_action_poisons_with_opponent_action() {
    for bad in [3, -1, i64::MAX] {
        let mut p = pool(&[PolicyKind::Onnx { content_hash: HASH }], None)
            .with_onnx_opponents(Box::new(ByteRule { bad: Some(bad) }))
            .unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        let a = vec![0i32; N as usize * A];
        let r = p.step(ActionInput {
            discrete: &a,
            continuous: &[],
        });
        assert_eq!(
            r.err(),
            Some(EngineError::OpponentAction {
                instance: 0,
                slot: 1
            }),
            "{bad}"
        );
        assert!(p.is_poisoned());
    }
}

#[test]
fn test_with_onnx_opponents_refusals() {
    let league = || {
        pool(
            &[PolicyKind::Onnx {
                content_hash: [1; 16],
            }],
            None,
        )
    };
    // Not served.
    assert!(matches!(
        league().with_onnx_opponents(Box::new(ByteRule { bad: None })),
        Err(EngineError::League { .. })
    ));
    // Twice.
    let once = pool(&[PolicyKind::Onnx { content_hash: HASH }], None)
        .with_onnx_opponents(Box::new(ByteRule { bad: None }))
        .unwrap();
    assert!(matches!(
        once.with_onnx_opponents(Box::new(ByteRule { bad: None })),
        Err(EngineError::League { .. })
    ));
    // After the first reset.
    let mut reset = pool(&[PolicyKind::Onnx { content_hash: HASH }], None);
    reset.reset_all(Reseed::Keep).unwrap();
    assert!(matches!(
        reset.with_onnx_opponents(Box::new(ByteRule { bad: None })),
        Err(EngineError::League { .. })
    ));
    // Without a league.
    let options = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1));
    let plain = EnvPool::<PaddleDuel, NoMasks>::new(PaddleDuelConfig::default(), options).unwrap();
    assert!(matches!(
        plain.with_onnx_opponents(Box::new(ByteRule { bad: None })),
        Err(EngineError::League { .. })
    ));
}

const HASH_B: [u8; 16] = [8; 16];

/// Each call's policy and row count.
type Calls = std::sync::Arc<std::sync::Mutex<Vec<([u8; 16], usize)>>>;

/// Two ONNX policies with different rules, counting the rows of each call.
struct TwoRules {
    calls: Calls,
}

impl sealed::Sealed for TwoRules {}

impl OpponentPolicies for TwoRules {
    fn serves(&self, content_hash: &[u8; 16]) -> bool {
        *content_hash == HASH || *content_hash == HASH_B
    }

    fn act(
        &mut self,
        content_hash: &[u8; 16],
        rows: &[u8],
        n: usize,
        actions: &mut [i64],
    ) -> Result<(), OpponentPolicyError> {
        self.calls.lock().unwrap().push((*content_hash, n));
        let row = rows.len() / n;
        for (j, a) in actions.iter_mut().enumerate() {
            let r = rule(rows.get(j * row..(j + 1) * row).unwrap());
            *a = if *content_hash == HASH {
                r
            } else {
                (r + 1) % 3
            };
        }
        Ok(())
    }
}

/// Each ONNX policy plays only the instances assigned to it: two ONNX
/// policies equal the caller submitting each rule by `assignment()`.
#[test]
fn test_onnx_two_policies_each_play_their_assigned_instances() {
    let calls = std::sync::Arc::default();
    let mut onnx = pool(
        &[
            PolicyKind::Onnx { content_hash: HASH },
            PolicyKind::Onnx {
                content_hash: HASH_B,
            },
        ],
        None,
    )
    .with_onnx_opponents(Box::new(TwoRules {
        calls: std::sync::Arc::clone(&calls),
    }))
    .unwrap();
    let mut external = pool(
        &[
            PolicyKind::External { tag: 1 },
            PolicyKind::External { tag: 2 },
        ],
        None,
    );
    onnx.reset_all(Reseed::Keep).unwrap();
    external.reset_all(Reseed::Keep).unwrap();
    let mut both = [false; 2];
    for t in 0..20 {
        assert_eq!(trace(&onnx), trace(&external), "step {t}");
        assert_eq!(onnx.assignment(), external.assignment(), "step {t}");
        let assignment = external.assignment().unwrap().to_vec();
        let mut a = vec![0i32; N as usize * A];
        let mut b = vec![0i32; N as usize * A];
        for i in 0..N as usize {
            a[i * A] = learner(t, i);
            b[i * A] = learner(t, i);
            let r = rule(&row_of(&external, i * A + 1));
            let policy = assignment[i] as usize;
            both[policy] = true;
            b[i * A + 1] = i32::try_from(if policy == 0 { r } else { (r + 1) % 3 }).unwrap();
        }
        for (p, x) in [(&mut onnx, &a), (&mut external, &b)] {
            p.step(ActionInput {
                discrete: x,
                continuous: &[],
            })
            .unwrap();
        }
    }
    assert_eq!(both, [true, true]);
    assert!(!calls.lock().unwrap().is_empty());
}

/// Instances waiting for their reset are not in the batch: with every
/// episode ending together, a reset step makes no call, and every other
/// step one call per policy with every instance's row.
#[test]
fn test_onnx_pending_reset_instances_are_not_batched() {
    let calls = std::sync::Arc::default();
    let mut p = pool(&[PolicyKind::Onnx { content_hash: HASH }], None)
        .with_onnx_opponents(Box::new(TwoRules {
            calls: std::sync::Arc::clone(&calls),
        }))
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    // `pool` truncates at 5 steps with `AutoReset::NextStep`: steps 6 and
    // 12 are reset steps.
    let a = vec![1i32; N as usize * A];
    for _ in 0..12 {
        p.step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
    }
    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 10, "{calls:?}");
    assert!(calls.iter().all(|&(h, n)| h == HASH && n == N as usize));
}
