// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persisted snapshots: arbitrary bytes decoded as a Tag `DecodedRecord`
//! and imported. Decoding and import return typed errors; an accepted
//! record must round-trip and the pool must keep stepping (ADR-0010).

#![no_main]

use core::num::NonZeroU32;

use libfuzzer_sys::fuzz_target;
use tessera_rl_classic::{Tag, TagConfig};
use tessera_rl_core::encoding::{decode_from_slice, encode_to_vec};
use tessera_rl_engine::{ActionInput, DecodedRecord, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

fuzz_target!(|data: &[u8]| {
    let Ok(record) = decode_from_slice::<DecodedRecord<Tag>>(data) else {
        return;
    };
    assert_eq!(encode_to_vec(&record), data, "decoded bytes are not canonical");
    let options = PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(3));
    let mut p = EnvPool::<Tag, Masked>::with_masks(TagConfig::default(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    if p.import(&[1], &[record]).is_ok() {
        assert_eq!(p.snapshot(1).unwrap().to_bytes(), data);
        let acts = [0; 16];
        let _ = p.step(ActionInput {
            discrete: &acts,
            continuous: &[],
        });
    }
});
