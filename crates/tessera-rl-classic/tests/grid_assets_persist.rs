// SPDX-License-Identifier: MIT OR Apache-2.0

//! Asset entries in persisted headers (ADR-0077 point 6), with the Grid
//! `layout` asset: snapshot files are version 1 without assets and version 2
//! with them; a different asset is `AssetMismatch`, before the config hash;
//! replay files carry the entries in version 1.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::{CartPole, CartPoleConfig, Grid, GridConfig, GridLayout};
use tessera_rl_core::Asset;
use tessera_rl_core::encoding::checksum;
use tessera_rl_engine::replay::{Recorder, Replay, ReplayError, ReplayWriterOptions};
use tessera_rl_engine::{
    ASSET_FORMAT_VERSION, ActionInput, DecodeLimits, EnvPool, FORMAT_VERSION, NoMasks,
    PersistError, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;

const ROOMS: &[u8] =
    b"....#...\n....#...\n........\n....#...\n###.####\n....#...\n........\n....#...\n";
const ROOMS_B: &[u8] =
    b"....#...\n....#...\n........\n....#...\n####.###\n....#...\n........\n....#...\n";

fn cfg(layout: &[u8]) -> GridConfig {
    GridConfig {
        size: 8,
        max_steps: 30,
        layout: Asset::<GridLayout>::from_bytes(layout).unwrap(),
    }
}

fn pool(c: GridConfig) -> EnvPool<Grid, NoMasks> {
    let options = PoolOptions::new(NonZeroU32::new(4).unwrap(), Seed::new(9));
    let mut p = EnvPool::new(c, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn version(file: &[u8]) -> u32 {
    u32::from_le_bytes(file.get(8..12).unwrap().try_into().unwrap())
}

/// Rewrites the version field and the trailing checksum.
fn with_version(file: &[u8], v: u32) -> Vec<u8> {
    let mut f = file.to_vec();
    f.get_mut(8..12).unwrap().copy_from_slice(&v.to_le_bytes());
    let body = f.len() - 8;
    let sum = checksum(f.get(..body).unwrap());
    f.get_mut(body..)
        .unwrap()
        .copy_from_slice(&sum.to_le_bytes());
    f
}

#[test]
fn test_grid_assets_snapshot_version_follows_the_assets() {
    // Every Grid config has a `layout` asset, the open default included.
    let open = GridConfig {
        size: 8,
        max_steps: 30,
        ..GridConfig::default()
    };
    let file = pool(open.clone()).encode(&[0, 1]).unwrap();
    assert_eq!(version(&file), ASSET_FORMAT_VERSION);
    let hash = open.layout.hash().0;
    let entry = [&[6u8][..], b"layout", &hash].concat();
    assert!(file.windows(entry.len()).any(|w| w == entry.as_slice()));
    let walled = pool(cfg(ROOMS)).encode(&[0, 1]).unwrap();
    assert_eq!(version(&walled), ASSET_FORMAT_VERSION);
    // An asset-free environment keeps version 1.
    let cartpole = EnvPool::<CartPole, NoMasks>::new(
        CartPoleConfig::default(),
        PoolOptions::new(NonZeroU32::MIN, Seed::new(1)),
    )
    .unwrap();
    assert_eq!(version(&cartpole.encode(&[0]).unwrap()), FORMAT_VERSION);
}

#[test]
fn test_grid_assets_snapshot_round_trips_and_imports() {
    let mut a = pool(cfg(ROOMS));
    a.step(ActionInput {
        discrete: &[1, 1, 1, 1],
        continuous: &[],
    })
    .unwrap();
    let file = a.encode(&[0, 1, 2, 3]).unwrap();
    let mut b = pool(cfg(ROOMS));
    let records = b.decode(&file, DecodeLimits::default()).unwrap();
    b.import(&[0, 1, 2, 3], &records).unwrap();
    for i in 0..4 {
        assert_eq!(a.state(i).unwrap(), b.state(i).unwrap());
    }
}

#[test]
fn test_grid_walled_reachable_odd_parity_state_round_trips() {
    // #721: on ROOMS the cell (4, 0) is a wall, so four moves right end at
    // (3, 0) after 4 steps: distance 3, odd parity, a reachable state that
    // the validator refused before the fix.
    let mut a = pool(cfg(ROOMS));
    for _ in 0..4 {
        a.step(ActionInput {
            discrete: &[2, 2, 2, 2],
            continuous: &[],
        })
        .unwrap();
    }
    let ids: Vec<u32> = (0..4)
        .filter(|&i| {
            let s = a.state(i).unwrap();
            s.cell() == (3, 0) && s.steps == 4 && !s.reached_goal.get()
        })
        .collect();
    assert!(!ids.is_empty(), "no instance reached the odd-parity state");
    // Through the bytes: encode, decode (which validates) and import.
    let file = a.encode(&ids).unwrap();
    let mut b = pool(cfg(ROOMS));
    let records = b.decode(&file, DecodeLimits::default()).unwrap();
    b.import(&ids, &records).unwrap();
    // In memory: snapshot and restore.
    let mut c = pool(cfg(ROOMS));
    for &i in &ids {
        let record = a.snapshot(i).unwrap();
        c.restore(i, &record).unwrap();
    }
    for &i in &ids {
        assert_eq!(a.state(i).unwrap(), b.state(i).unwrap());
        assert_eq!(a.state(i).unwrap(), c.state(i).unwrap());
    }
}

#[test]
fn test_grid_assets_other_layout_is_an_asset_mismatch() {
    let file = pool(cfg(ROOMS)).encode(&[0]).unwrap();
    let ours = Asset::<GridLayout>::from_bytes(ROOMS_B).unwrap().hash().0;
    let theirs = Asset::<GridLayout>::from_bytes(ROOMS).unwrap().hash().0;
    let err = pool(cfg(ROOMS_B))
        .decode(&file, DecodeLimits::default())
        .unwrap_err();
    match err {
        PersistError::AssetMismatch {
            name,
            expected,
            found,
        } => {
            assert_eq!(name, "layout");
            assert_eq!(expected, Some(ours));
            assert_eq!(found, Some(theirs));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn test_grid_assets_open_file_into_walled_pool_is_an_asset_mismatch() {
    let open = GridConfig {
        size: 8,
        max_steps: 30,
        ..GridConfig::default()
    };
    let file = pool(open).encode(&[0]).unwrap();
    let err = pool(cfg(ROOMS))
        .decode(&file, DecodeLimits::default())
        .unwrap_err();
    assert!(
        matches!(err, PersistError::AssetMismatch { ref name, .. } if name == "layout"),
        "{err:?}"
    );
}

#[test]
fn test_grid_assets_version_and_count_must_agree() {
    let file = pool(cfg(ROOMS)).encode(&[0]).unwrap();
    // Version 1 with asset entries: refused as before ADR-0077.
    let v1 = with_version(&file, FORMAT_VERSION);
    assert!(matches!(
        pool(cfg(ROOMS)).decode(&v1, DecodeLimits::default()),
        Err(PersistError::Incompatible { field: "assets" })
    ));
    // Version 3: unknown.
    let v3 = with_version(&file, 3);
    assert!(matches!(
        pool(cfg(ROOMS)).decode(&v3, DecodeLimits::default()),
        Err(PersistError::UnsupportedFormat { found: 3 })
    ));
}

#[test]
fn test_grid_assets_replay_carries_entries_and_checks_them() {
    let mut p = pool(cfg(ROOMS));
    let options = ReplayWriterOptions {
        keyframe_every: 0,
        checkpoint_every: 2,
        outputs_hash: false,
    };
    let mut rec = Recorder::new(&mut p, Vec::new(), options).unwrap();
    rec.reset_all(Reseed::Keep).unwrap();
    for _ in 0..4 {
        rec.step(ActionInput {
            discrete: &[1, 2, 1, 2],
            continuous: &[],
        })
        .unwrap();
    }
    let bytes = rec.finish().unwrap();
    assert_eq!(version(&bytes), FORMAT_VERSION);
    let replay = Replay::open(&bytes, DecodeLimits::default()).unwrap();
    replay.run(&mut pool(cfg(ROOMS))).unwrap();
    let err = replay.run(&mut pool(cfg(ROOMS_B))).unwrap_err();
    assert!(
        matches!(
            err,
            ReplayError::Persist(PersistError::AssetMismatch { .. })
        ),
        "{err:?}"
    );
}

/// `file` with its single asset entry removed: count 0, `version`.
fn without_entry(file: &[u8], version_field: u32) -> Vec<u8> {
    let at = file.windows(7).position(|w| w == b"\x06layout").unwrap();
    let count_at = at - 2;
    let mut f = [
        file.get(..count_at).unwrap(),
        &0u16.to_le_bytes(),
        file.get(at + 1 + 6 + 16..).unwrap(),
    ]
    .concat();
    f = with_version(&f, version_field);
    f
}

#[test]
fn test_grid_assets_entry_missing_from_the_file_is_an_asset_mismatch() {
    let c = cfg(ROOMS);
    let file = without_entry(&pool(c.clone()).encode(&[0]).unwrap(), FORMAT_VERSION);
    let err = pool(c.clone())
        .decode(&file, DecodeLimits::default())
        .unwrap_err();
    match err {
        PersistError::AssetMismatch {
            name,
            expected,
            found,
        } => {
            assert_eq!(name, "layout");
            assert_eq!(expected, Some(c.layout.hash().0));
            assert_eq!(found, None);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn test_grid_assets_version_2_without_entries_is_invalid() {
    let file = without_entry(
        &pool(cfg(ROOMS)).encode(&[0]).unwrap(),
        ASSET_FORMAT_VERSION,
    );
    assert!(matches!(
        pool(cfg(ROOMS)).decode(&file, DecodeLimits::default()),
        Err(PersistError::InvalidField {
            field: "asset_count"
        })
    ));
}

#[test]
fn test_grid_assets_count_over_64_is_invalid_before_entries() {
    let file = pool(cfg(ROOMS)).encode(&[0]).unwrap();
    let at = file.windows(7).position(|w| w == b"\x06layout").unwrap() - 2;
    let mut f = file.clone();
    f.get_mut(at..at + 2)
        .unwrap()
        .copy_from_slice(&65u16.to_le_bytes());
    let f = with_version(&f, ASSET_FORMAT_VERSION);
    assert!(matches!(
        pool(cfg(ROOMS)).decode(&f, DecodeLimits::default()),
        Err(PersistError::InvalidField {
            field: "asset_count"
        })
    ));
    // 64 entries are read (then the file runs short of them).
    let mut f = file;
    f.get_mut(at..at + 2)
        .unwrap()
        .copy_from_slice(&64u16.to_le_bytes());
    let f = with_version(&f, ASSET_FORMAT_VERSION);
    assert!(!matches!(
        pool(cfg(ROOMS)).decode(&f, DecodeLimits::default()),
        Err(PersistError::InvalidField {
            field: "asset_count"
        })
    ));
}
