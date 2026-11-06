// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persisted snapshots of a catalog environment (ADR-0011): continuation through a
//! file, and a checked-in fixture that pins the format. Fixtures are written only with
//! `TESSERA_BLESS=1` and only when missing, with `library_version` set to [`LIB`], so
//! a release changes none. Shared by the classic and games crates.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::missing_panics_doc,
    clippy::cast_possible_truncation,
    reason = "test harness: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::config::CanonicalConfig;
use tessera_rl_core::{AdaptiveValues, Persist, RecordLayout};
use tessera_rl_engine::{
    ActionInput, DecodeLimits, EnvPool, MaskProvider, PersistMask, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;

pub const LIB: &str = "fixture";

/// Plays `steps` with a scripted action from `act(step, row, masks)`.
pub fn played<E, M>(
    build: &dyn Fn(PoolOptions) -> EnvPool<E, M>,
    n: u32,
    seed: u64,
    steps: u32,
    width: usize,
) -> EnvPool<E, M>
where
    E: Persist,
    E::Config: CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    M: PersistMask<E> + MaskProvider<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    let mut p = build(PoolOptions::new(
        NonZeroU32::new(n).unwrap(),
        Seed::new(seed),
    ));
    p.reset_all(Reseed::Keep).unwrap();
    for t in 0..steps {
        drive(&mut p, t, width);
    }
    p
}

/// One step: per row, a scripted choice among the legal actions.
pub fn drive<E, M>(p: &mut EnvPool<E, M>, t: u32, width: usize)
where
    E: Persist,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    M: MaskProvider<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    let rows = p.outputs().rewards().len();
    let masks: Vec<bool> = p.outputs().masks().iter().map(|m| m.get()).collect();
    let mut acts = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut pick = (row as u32)
            .wrapping_mul(31)
            .wrapping_add(t.wrapping_mul(7)) as usize
            % width;
        if !masks.is_empty() {
            let legal = &masks[row * width..(row + 1) * width];
            if !legal[pick] {
                pick = legal.iter().position(|&l| l).unwrap_or(0);
            }
        }
        acts.push(i32::try_from(pick).unwrap());
    }
    p.step(ActionInput {
        discrete: &acts,
        continuous: &[],
    })
    .unwrap();
}

/// A file continues identically in a pool of another size.
pub fn check_continuation<E, M>(build: &dyn Fn(PoolOptions) -> EnvPool<E, M>, width: usize)
where
    E: Persist,
    E::Config: CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    M: PersistMask<E> + MaskProvider<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    for seed in 0..8 {
        let mut a = played(build, 1, seed, 6, width);
        let f = a.encode(&[0]).unwrap();
        let mut b = played(build, 3, seed + 100, 2, width);
        let records = b.decode(&f, DecodeLimits::default()).unwrap();
        b.import(&[2], &records).unwrap();
        assert_eq!(b.state_hash(2).unwrap(), a.state_hash(0).unwrap());
        let mut c = build(PoolOptions::new(NonZeroU32::MIN, Seed::new(seed + 200)));
        c.reset_all(Reseed::Keep).unwrap();
        c.import_all(&records).unwrap();
        for t in 0..12 {
            drive(&mut a, t, width);
            drive(&mut c, t, width);
            assert_eq!(c.state_hash(0).unwrap(), a.state_hash(0).unwrap());
            assert_eq!(c.outputs().rewards(), a.outputs().rewards());
        }
    }
}

/// The fixture decodes and re-encodes to its own bytes, and the same seeded
/// play yields those bytes on every platform.
pub fn check_fixture<E, M>(
    name: &str,
    dir: &str,
    build: &dyn Fn(PoolOptions) -> EnvPool<E, M>,
    width: usize,
) where
    E: Persist,
    E::Config: CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    M: PersistMask<E> + MaskProvider<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    let path = std::path::Path::new(dir)
        .join("tests/golden")
        .join(format!("{name}.bin"));
    let p = played(build, 2, 7, 5, width);
    let fresh = p.encode_with(&[0, 1], LIB, None).unwrap();
    if std::env::var_os("TESSERA_BLESS").is_some() && !path.exists() {
        std::fs::write(&path, &fresh).unwrap();
    }
    let stored = std::fs::read(&path).unwrap();
    assert_eq!(
        fresh, stored,
        "{name}: a format change needs a version bump"
    );
    let mut q = build(PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(1)));
    q.reset_all(Reseed::Keep).unwrap();
    let records = q.decode(&stored, DecodeLimits::default()).unwrap();
    q.import_all(&records).unwrap();
    assert_eq!(q.encode_with(&[0, 1], LIB, None).unwrap(), stored);
}
