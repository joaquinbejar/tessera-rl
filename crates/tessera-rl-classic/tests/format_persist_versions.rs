// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persisted record files, version 1 (`FORMAT_VERSION`) and version 2
//! (`ASSET_FORMAT_VERSION`, a config with assets, ADR-0077), against their
//! committed fixtures: `tests/golden/persist-cartpole.bin` (version 1) and
//! `tests/golden/persist-grid.bin` (version 2, the Grid `layout` asset).
//! `tests/persist.rs` reads both back and pins their writer; this file
//! checks their versions and that a newer or another-major version is
//! refused with the typed error naming it.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::{CartPole, CartPoleConfig, Grid, GridConfig};
use tessera_rl_core::encoding::checksum;
use tessera_rl_engine::{
    ASSET_FORMAT_VERSION, DecodeLimits, EnvPool, FORMAT_VERSION, NoMasks, PersistError, PoolOptions,
};
use tessera_rl_num::identity::Seed;

const V1: &[u8] = include_bytes!("golden/persist-cartpole.bin");
const V2: &[u8] = include_bytes!("golden/persist-grid.bin");

fn version(file: &[u8]) -> u32 {
    u32::from_le_bytes(file[8..12].try_into().unwrap())
}

/// `file` with its version field set to `v` and the trailing checksum
/// rewritten, so only the version is wrong.
fn with_version(file: &[u8], v: u32) -> Vec<u8> {
    let mut f = file.to_vec();
    f[8..12].copy_from_slice(&v.to_le_bytes());
    let body = f.len() - 8;
    let sum = checksum(&f[..body]);
    f[body..].copy_from_slice(&sum.to_le_bytes());
    f
}

fn options() -> PoolOptions {
    PoolOptions::new(NonZeroU32::MIN, Seed::new(1))
}

#[test]
fn test_format_persist_fixtures_have_versions_1_and_2() {
    assert_eq!((FORMAT_VERSION, ASSET_FORMAT_VERSION), (1, 2));
    assert_eq!(version(V1), 1);
    assert_eq!(version(V2), 2);
}

#[test]
fn test_format_persist_newer_and_other_major_versions_are_refused() {
    let cartpole = EnvPool::<CartPole, NoMasks>::new(CartPoleConfig::default(), options()).unwrap();
    let grid = EnvPool::<Grid, NoMasks>::new(GridConfig::default(), options()).unwrap();
    for found in [3, 1 << 16] {
        assert!(matches!(
            cartpole.decode(&with_version(V1, found), DecodeLimits::default()),
            Err(PersistError::UnsupportedFormat { found: f }) if f == found
        ));
        assert!(matches!(
            grid.decode(&with_version(V2, found), DecodeLimits::default()),
            Err(PersistError::UnsupportedFormat { found: f }) if f == found
        ));
    }
    let message = PersistError::UnsupportedFormat { found: 3 }.to_string();
    assert!(
        message.contains('3') && message.contains("version"),
        "{message}"
    );
}
