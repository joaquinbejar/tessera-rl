// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `record` harness, shared by the fuzz target and the stable replay
//! test of `fuzz/regressions/record/`: decode arbitrary bytes as a Tag
//! `DecodedRecord`, import it and step. Decoding and import return typed
//! errors; an accepted record must round-trip, and stepping it must not
//! raise an invalid agent event (ADR-0040: import rejects records whose
//! header and state disagree).

use core::num::NonZeroU32;

use tessera_rl_classic::{Tag, TagConfig};
use tessera_rl_core::encoding::{decode_from_slice, encode_to_vec};
use tessera_rl_engine::{ActionInput, DecodedRecord, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

/// What the harness did with an input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Not a canonical record.
    Undecodable,
    /// Decoded, rejected by import.
    Rejected,
    /// Decoded, imported and stepped.
    Imported,
}

/// Runs the harness on `data`; panics on a broken invariant.
pub fn run(data: &[u8]) -> Outcome {
    let Ok(record) = decode_from_slice::<DecodedRecord<Tag>>(data) else {
        return Outcome::Undecodable;
    };
    assert_eq!(
        encode_to_vec(&record),
        data,
        "decoded bytes are not canonical"
    );
    let Some(n) = NonZeroU32::new(2) else {
        return Outcome::Undecodable;
    };
    let options = PoolOptions::new(n, Seed::new(3));
    let Ok(mut p) = EnvPool::<Tag, Masked>::with_masks(TagConfig::default(), options) else {
        return Outcome::Undecodable;
    };
    assert!(p.reset_all(Reseed::Keep).is_ok());
    if p.import(&[1], &[record]).is_err() {
        return Outcome::Rejected;
    }
    let snapshot = p.snapshot(1).map(|r| r.to_bytes());
    assert_eq!(snapshot.ok().as_deref(), Some(data));
    let events = |p: &EnvPool<Tag, Masked>| {
        p.header(1).map(|h| {
            h.agents
                .iter()
                .map(|r| u32::from(r.invalid_agent_events))
                .sum::<u32>()
        })
    };
    let before = events(&p);
    let acts = [0; 16];
    if p.step(ActionInput {
        discrete: &acts,
        continuous: &[],
    })
    .is_ok()
    {
        let ended = p
            .outputs()
            .unwrap()
            .episode_terminated()
            .get(1)
            .is_some_and(|f| f.get())
            || p.outputs()
                .unwrap()
                .episode_truncated()
                .get(1)
                .is_some_and(|f| f.get());
        if !ended && before.is_some() {
            assert_eq!(
                events(&p),
                before,
                "an accepted record raised an invalid agent event"
            );
        }
    }
    Outcome::Imported
}
