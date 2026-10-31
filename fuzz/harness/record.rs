// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `record` harness, shared by the fuzz target and the stable replay
//! test of `fuzz/regressions/record/`: decode arbitrary bytes as a Tag
//! `DecodedRecord`, import it and step. Decoding and import return typed
//! errors; an accepted record must round-trip and the pool must keep
//! stepping. This covers today's import checks, not the header-aware
//! cross rules of ADR-0040 (Proposed).

use core::num::NonZeroU32;

use tessera_rl_classic::{Tag, TagConfig};
use tessera_rl_core::encoding::{decode_from_slice, encode_to_vec};
use tessera_rl_engine::{ActionInput, DecodedRecord, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

/// Runs the harness on `data`; panics on a broken invariant.
pub fn run(data: &[u8]) {
    let Ok(record) = decode_from_slice::<DecodedRecord<Tag>>(data) else {
        return;
    };
    assert_eq!(
        encode_to_vec(&record),
        data,
        "decoded bytes are not canonical"
    );
    let Some(n) = NonZeroU32::new(2) else {
        return;
    };
    let options = PoolOptions::new(n, Seed::new(3));
    let Ok(mut p) = EnvPool::<Tag, Masked>::with_masks(TagConfig::default(), options) else {
        return;
    };
    assert!(p.reset_all(Reseed::Keep).is_ok());
    if p.import(&[1], &[record]).is_ok() {
        let snapshot = p.snapshot(1).map(|r| r.to_bytes());
        assert_eq!(snapshot.ok().as_deref(), Some(data));
        let acts = [0; 16];
        let _ = p.step(ActionInput {
            discrete: &acts,
            continuous: &[],
        });
    }
}
