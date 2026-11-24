// SPDX-License-Identifier: MIT OR Apache-2.0

//! Archive resets (ADR-0075) on CartPole: selection is deterministic,
//! weighted in integers and independent of the thread count; an archive
//! reset keeps the instance's own RNG identity; attaching and restoring are
//! checked; and a golden hash pins the whole path.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    reason = "test helpers: a failure is the assertion; counts below 2^52"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_classic::{CartPole, CartPoleConfig, Grid, GridConfig};
use tessera_rl_core::RecordLayout;
use tessera_rl_core::encoding::StateHasher;
use tessera_rl_engine::{
    ActionInput, Archive, AutoReset, EngineError, EnvPool, NoMasks, PoolContext, PoolOptions,
    Record, Reseed, ResetSource, ResetSourceContext, SyncOptions,
};
use tessera_rl_num::identity::Seed;

type Pool = EnvPool<CartPole>;

fn options(n: u32, seed: u64, auto_reset: AutoReset) -> PoolOptions {
    let mut o = PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed));
    o.auto_reset = auto_reset;
    o
}

fn pool(n: u32, seed: u64, auto_reset: AutoReset, workers: usize) -> Pool {
    let p = Pool::new(CartPoleConfig::default(), options(n, seed, auto_reset)).unwrap();
    if workers == 0 {
        p
    } else {
        p.with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
            .unwrap()
    }
}

/// `count` distinct mid-episode CartPole records: instance 0 of a pool
/// stepped `5 * (k + 1)` times with alternating actions.
fn records(count: usize) -> Vec<Record<CartPole>> {
    let mut out = Vec::new();
    for k in 0..count {
        let mut p = pool(1, 100 + k as u64, AutoReset::Disabled, 0);
        p.reset_all(Reseed::Keep).unwrap();
        for t in 0..5 * (k + 1) {
            let a = [i32::try_from(t % 2).unwrap()];
            p.step(ActionInput {
                discrete: &a,
                continuous: &[],
            })
            .unwrap();
        }
        out.push(p.snapshot(0).unwrap());
    }
    out
}

fn archive(count: usize) -> Archive<CartPole> {
    Archive::new(&CartPoleConfig::default(), &records(count)).unwrap()
}

/// The entry instance `i` now holds, matched by state bits.
fn entry_of(p: &Pool, i: u32, entries: &[Record<CartPole>]) -> Option<usize> {
    let now = p.snapshot(i).unwrap();
    let bits = |r: &Record<CartPole>| r.state().s.map(f64::to_bits);
    entries.iter().position(|e| bits(e) == bits(&now))
}

/// Resets instance 0 `n` times and counts the entries it lands on.
fn counts(p: &mut Pool, entries: &[Record<CartPole>], n: usize) -> Vec<u64> {
    let mut c = vec![0u64; entries.len()];
    for _ in 0..n {
        p.reset(&[0], Reseed::Keep).unwrap();
        c[entry_of(p, 0, entries).unwrap()] += 1;
    }
    c
}

#[test]
fn test_archive_resets_weights_match_frequencies_and_zero_is_never_chosen() {
    // ADR-0075 item 10: weights (1, 0, 3, 4), pool seed 7, one instance,
    // 100,000 archive resets; chi-square over entries 0, 2, 3 (df 2) below
    // the 0.999 quantile, 13.82.
    let entries = records(4);
    let arc = Archive::new(&CartPoleConfig::default(), &entries).unwrap();
    let mut p = pool(1, 7, AutoReset::Disabled, 0);
    p.set_reset_source(ResetSource::Archive {
        archive: arc,
        weights: Some(vec![1, 0, 3, 4]),
    })
    .unwrap();
    let n = 100_000;
    let c = counts(&mut p, &entries, n);
    assert_eq!(c[1], 0, "a weight-0 entry was chosen");
    let expected = [n as f64 / 8.0, 3.0 * n as f64 / 8.0, 4.0 * n as f64 / 8.0];
    let observed = [c[0] as f64, c[2] as f64, c[3] as f64];
    let chi2: f64 = observed
        .iter()
        .zip(expected)
        .map(|(o, e)| (o - e) * (o - e) / e)
        .sum();
    assert!(chi2 < 13.82, "chi-square {chi2} for counts {c:?}");
}

#[test]
fn test_archive_resets_keep_the_instance_identity_and_advance_the_episode() {
    let entries = records(2);
    let mut p = pool(2, 11, AutoReset::Disabled, 0);
    p.reset_all(Reseed::Keep).unwrap();
    let before = *p.snapshot(1).unwrap().header().engine();
    p.set_reset_source(ResetSource::Archive {
        archive: Archive::new(&CartPoleConfig::default(), &entries).unwrap(),
        weights: None,
    })
    .unwrap();
    p.reset_from(&[1], &[1]).unwrap();
    let after = p.snapshot(1).unwrap();
    let id = after.header().engine().identity;
    assert_eq!(id.seed(), before.identity.seed());
    assert_eq!(id.logical_id(), before.identity.logical_id());
    assert_eq!(id.episode_index(), before.identity.episode_index() + 1);
    assert_eq!(id.step_index(), 0);
    assert_eq!(entry_of(&p, 1, &entries), Some(1));
    // The entry's own header (its seed and counters) is not copied.
    assert_ne!(
        entries[1].header().engine().identity.seed(),
        id.seed(),
        "the entry came from another seed"
    );
}

#[test]
fn test_archive_resets_selection_is_independent_of_workers() {
    let entries = records(5);
    let arc = Archive::new(&CartPoleConfig::default(), &entries).unwrap();
    let mut runs = Vec::new();
    for workers in [0, 3] {
        let mut p = pool(16, 3, AutoReset::Disabled, workers);
        p.set_reset_source(ResetSource::Archive {
            archive: arc.clone(),
            weights: Some(vec![2, 1, 0, 5, 1]),
        })
        .unwrap();
        let mut picks = Vec::new();
        for _ in 0..20 {
            p.reset_all(Reseed::Keep).unwrap();
            for i in 0..16 {
                picks.push(entry_of(&p, i, &entries).unwrap());
            }
        }
        runs.push(picks);
    }
    assert_eq!(runs[0], runs[1]);
}

#[test]
fn test_archive_resets_attach_and_reset_from_refusals_mutate_nothing() {
    let arc = archive(3);
    let mut p = pool(2, 1, AutoReset::Disabled, 0);
    p.reset_all(Reseed::Keep).unwrap();
    assert!(matches!(
        p.reset_from(&[0], &[0]),
        Err(EngineError::Archive { .. })
    ));
    assert!(matches!(
        p.set_reset_source(ResetSource::Archive {
            archive: arc.clone(),
            weights: Some(vec![0, 0, 0]),
        }),
        Err(EngineError::Archive { .. })
    ));
    assert!(matches!(
        p.set_reset_source(ResetSource::Archive {
            archive: arc.clone(),
            weights: Some(vec![1, 1]),
        }),
        Err(EngineError::Archive { .. })
    ));
    assert_eq!(
        p.pool_context().unwrap().reset_source,
        ResetSourceContext::Fresh
    );
    p.set_reset_source(ResetSource::Archive {
        archive: arc,
        weights: None,
    })
    .unwrap();
    let before = [p.snapshot(0).unwrap(), p.snapshot(1).unwrap()];
    assert!(matches!(
        p.reset_from(&[0, 1], &[0, 3]),
        Err(EngineError::Archive { .. })
    ));
    assert!(matches!(
        p.reset_from(&[0, 1], &[0]),
        Err(EngineError::RecordCount { .. })
    ));
    assert!(p.reset_from(&[0, 0], &[0, 1]).is_err());
    let after = [p.snapshot(0).unwrap(), p.snapshot(1).unwrap()];
    assert_eq!(before.map(|r| r.hash()), after.map(|r| r.hash()));
}

#[test]
fn test_archive_resets_pool_context_round_trip_needs_the_same_archive() {
    let arc = archive(3);
    let mut p = pool(2, 5, AutoReset::Disabled, 0);
    p.set_reset_source(ResetSource::Archive {
        archive: arc.clone(),
        weights: Some(vec![1, 2, 3]),
    })
    .unwrap();
    let ctx = p.pool_context().unwrap();
    assert_eq!(
        ctx.reset_source,
        ResetSourceContext::Archive {
            digest: arc.digest(),
            weights: vec![1, 2, 3],
        }
    );
    let mut q = pool(2, 5, AutoReset::Disabled, 0);
    assert!(matches!(
        q.restore_pool_context(ctx.clone(), None),
        Err(EngineError::ArchiveMismatch { found: 0, .. })
    ));
    let other = archive(4);
    assert!(matches!(
        q.restore_pool_context(ctx.clone(), Some(&other)),
        Err(EngineError::ArchiveMismatch { .. })
    ));
    assert!(matches!(
        q.restore_pool_context(PoolContext::new(0), Some(&arc)),
        Err(EngineError::Archive { .. })
    ));
    assert_eq!(q.pool_context().unwrap(), PoolContext::new(0));
    q.restore_pool_context(ctx.clone(), Some(&arc)).unwrap();
    assert_eq!(q.pool_context().unwrap(), ctx);
}

/// `restore_pool` (#465) brings back the archive source the snapshot's
/// context names, with its weights, and refuses before any write without
/// that archive or with another.
#[test]
fn test_archive_resets_restore_pool_brings_back_the_archive_source() {
    let arc = archive(3);
    let mut p = pool(2, 5, AutoReset::Disabled, 0);
    p.reset_all(Reseed::Keep).unwrap();
    p.set_reset_source(ResetSource::Archive {
        archive: arc.clone(),
        weights: Some(vec![1, 2, 3]),
    })
    .unwrap();
    let state = p.snapshot_pool().unwrap();
    p.set_reset_source(ResetSource::Fresh).unwrap();
    let hashes = |p: &Pool| (0..2).map(|i| p.state_hash(i).unwrap()).collect::<Vec<_>>();
    let before = hashes(&p);

    assert!(matches!(
        p.restore_pool(&state, None),
        Err(EngineError::ArchiveMismatch { found: 0, .. })
    ));
    let other = archive(4);
    assert!(matches!(
        p.restore_pool(&state, Some(&other)),
        Err(EngineError::ArchiveMismatch { .. })
    ));
    assert_eq!(
        p.pool_context().unwrap().reset_source,
        ResetSourceContext::Fresh
    );
    assert_eq!(hashes(&p), before);

    p.restore_pool(&state, Some(&arc)).unwrap();
    assert_eq!(
        p.pool_context().unwrap().reset_source,
        ResetSourceContext::Archive {
            digest: arc.digest(),
            weights: vec![1, 2, 3],
        }
    );
    assert_eq!(p.pool_context().unwrap(), *state.context());
}

#[test]
fn test_archive_resets_archive_of_another_config_is_refused() {
    // CartPole has no config fields: Grid's size gives another config hash.
    let small = GridConfig {
        size: 5,
        ..GridConfig::default()
    };
    let mut src =
        EnvPool::<Grid, NoMasks>::new(small.clone(), options(1, 0, AutoReset::Disabled)).unwrap();
    src.reset_all(Reseed::Keep).unwrap();
    let foreign = Archive::new(&small, &[src.snapshot(0).unwrap()]).unwrap();
    let mut p =
        EnvPool::<Grid, NoMasks>::new(GridConfig::default(), options(1, 0, AutoReset::Disabled))
            .unwrap();
    assert!(matches!(
        p.set_reset_source(ResetSource::Archive {
            archive: foreign,
            weights: None,
        }),
        Err(EngineError::ArchiveMismatch { .. })
    ));
    assert_eq!(
        p.pool_context().unwrap().reset_source,
        ResetSourceContext::Fresh
    );
}

#[test]
fn test_archive_digest_depends_on_entries_and_order() {
    let r = records(3);
    let cfg = CartPoleConfig::default();
    let a = Archive::new(&cfg, &r).unwrap();
    let b = Archive::new(&cfg, &[r[0], r[2], r[1]]).unwrap();
    assert_ne!(a.digest(), b.digest());
    assert_eq!(a.digest(), Archive::new(&cfg, &r).unwrap().digest());
    assert!(matches!(
        Archive::<CartPole>::new(&cfg, &[]),
        Err(EngineError::Archive { .. })
    ));
}

/// The golden run (ADR-0075 item 10): CartPole, a three-entry archive with
/// weights (1, 0, 3), `same_step` auto-reset, 4 instances, 1,000 steps of
/// fixed actions; the state hashes at the end and the chosen entries of
/// every reset, hashed.
#[test]
fn test_archive_resets_golden_same_step_run() {
    let entries = records(3);
    let arc = Archive::new(&CartPoleConfig::default(), &entries).unwrap();
    let mut p = pool(4, 42, AutoReset::SameStep, 0);
    p.reset_all(Reseed::Keep).unwrap();
    p.set_reset_source(ResetSource::Archive {
        archive: arc.clone(),
        weights: Some(vec![1, 0, 3]),
    })
    .unwrap();
    let mut picks = StateHasher::new();
    let mut resets = 0u32;
    for t in 0..1_000u32 {
        let a: Vec<i32> = (0..4)
            .map(|i| i32::try_from((t + i) % 2).unwrap())
            .collect();
        let before: Vec<u32> = (0..4)
            .map(|i| {
                p.snapshot(i)
                    .unwrap()
                    .header()
                    .engine()
                    .identity
                    .episode_index()
            })
            .collect();
        p.step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
        for i in 0..4u32 {
            let now = p.snapshot(i).unwrap();
            if now.header().engine().identity.episode_index() != before[i as usize] {
                let e = entry_of(&p, i, &entries).unwrap();
                assert_ne!(e, 1, "a weight-0 entry was chosen");
                picks.add(&u32::try_from(e).unwrap());
                resets += 1;
            }
        }
    }
    let mut end = StateHasher::new();
    for i in 0..4 {
        end.add(&p.state_hash(i).unwrap().get());
    }
    let golden = (
        resets,
        picks.finish().get(),
        end.finish().get(),
        arc.digest(),
    );
    assert_eq!(golden, GOLDEN, "golden (resets, picks, end, digest)");
}

/// Pinned from the first run of this implementation (ADR-0075 item 10), on
/// Linux x86-64; CartPole is cross-platform, so every platform must match.
const GOLDEN: (u32, u64, u64, u64) = (
    221,
    17_159_053_969_331_542_853,
    16_136_120_751_482_678_990,
    13_652_088_480_962_496_469,
);
