// SPDX-License-Identifier: MIT OR Apache-2.0

//! The shared output storage (phase 1b B3a): one allocation, read-only
//! regions that match the typed accessors, a write generation incremented
//! by every call that writes outputs, and memory that outlives the pool.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod common;

use core::num::NonZeroU32;

use common::{CountConfig, Counter};
use tessera_rl_core::encoding::decode_from_slice;
use tessera_rl_engine::{
    ActionInput, DecodedRecord, EnvPool, Masked, OutputDtype, OutputField, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;

type Pool = EnvPool<Counter, Masked>;

fn pool(n: u32) -> Pool {
    let options = PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(3));
    EnvPool::with_masks(CountConfig::default(), options).unwrap()
}

fn step(p: &mut Pool) {
    let actions = vec![1; p.outputs().unwrap().rewards().len()];
    p.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap();
}

fn bytes_of(p: &Pool, field: OutputField) -> Vec<u8> {
    let storage = p.outputs().unwrap().storage();
    let region = storage.region(field).unwrap();
    let size = match region.dtype {
        OutputDtype::F32 | OutputDtype::U32 => 4,
        _ => 1,
    };
    // SAFETY: the region is `len * size` initialized bytes of a storage
    // kept alive by `storage`, and no engine call runs meanwhile.
    #[expect(unsafe_code, reason = "reading a borrowed output region")]
    unsafe { core::slice::from_raw_parts(region.ptr, region.len * size) }.to_vec()
}

#[test]
fn test_output_storage_regions_match_the_typed_accessors() {
    let mut p = pool(4);
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p);
    let rewards: Vec<u8> = p
        .outputs()
        .unwrap()
        .rewards()
        .iter()
        .flat_map(|r| r.to_le_bytes())
        .collect();
    assert_eq!(bytes_of(&p, OutputField::Rewards), rewards);
    assert_eq!(
        bytes_of(&p, OutputField::Observation(0)),
        p.outputs().unwrap().observation(0).unwrap()
    );
    let alive: Vec<u8> = p
        .outputs()
        .unwrap()
        .alive()
        .iter()
        .map(|b| u8::from(b.get()))
        .collect();
    assert_eq!(bytes_of(&p, OutputField::Alive), alive);
}

#[test]
fn test_output_storage_every_writing_call_increments_generation() {
    let mut p = pool(4);
    let storage = p.outputs().unwrap().storage();
    let mut last = storage.generation();
    let mut check = |what: &str, p: &Pool| {
        let g = p.outputs().unwrap().storage().generation();
        assert!(g > last, "{what} did not start a new generation");
        last = g;
    };
    p.reset_all(Reseed::Keep).unwrap();
    check("reset_all", &p);
    p.reset(&[1], Reseed::Keep).unwrap();
    check("reset", &p);
    step(&mut p);
    check("step", &p);
    let snap = p.snapshot(0).unwrap();
    p.restore(1, &snap).unwrap();
    check("restore", &p);
    let all: Vec<_> = (0..4).map(|i| p.snapshot(i).unwrap()).collect();
    p.restore_all(&all).unwrap();
    check("restore_all", &p);
    let decoded: DecodedRecord<Counter> = decode_from_slice(&snap.to_bytes()).unwrap();
    p.import(&[2], &[decoded]).unwrap();
    check("import", &p);
    let decoded: Vec<DecodedRecord<Counter>> = all
        .iter()
        .map(|r| decode_from_slice(&r.to_bytes()).unwrap())
        .collect();
    p.import_all(&decoded).unwrap();
    check("import_all", &p);
    p.fork(0, &[3]).unwrap();
    check("fork", &p);
    p.fork_branched(0, &[1, 2], None).unwrap();
    check("fork_branched", &p);
}

#[test]
fn test_output_storage_rejected_call_keeps_generation() {
    let mut p = pool(2);
    p.reset_all(Reseed::Keep).unwrap();
    let g = p.outputs().unwrap().storage().generation();
    let wrong = [0; 1];
    assert!(
        p.step(ActionInput {
            discrete: &wrong,
            continuous: &[],
        })
        .is_err()
    );
    assert_eq!(p.outputs().unwrap().storage().generation(), g);
}

#[test]
fn test_output_storage_outlives_the_pool() {
    let mut p = pool(3);
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p);
    let before = bytes_of(&p, OutputField::Rewards);
    let storage = p.outputs().unwrap().storage();
    drop(p);
    let region = storage.region(OutputField::Rewards).unwrap();
    // SAFETY: `storage` keeps the allocation alive after the pool is gone;
    // nothing writes it any more.
    #[expect(unsafe_code, reason = "reading a borrowed output region")]
    let after = unsafe { core::slice::from_raw_parts(region.ptr, region.len * 4) };
    assert_eq!(after, before.as_slice());
}
