// SPDX-License-Identifier: MIT OR Apache-2.0

//! The persisted-file harness (ADR-0011): arbitrary bytes, resealed with a valid
//! checksum so that the input reaches the checks behind it, go through `decode`,
//! then `import_all` and a step. Every refusal is a typed error; an accepted file
//! re-encodes to its own payload.

use tessera_rl_core::config::CanonicalConfig;
use tessera_rl_core::encoding::{checksum, encode_to_vec};
use tessera_rl_core::{AdaptiveValues, Persist, RecordLayout};
use tessera_rl_engine::{ActionInput, DecodeLimits, EnvPool, MaskProvider, PersistMask};

/// `data` with its last eight bytes replaced by the checksum of the rest.
#[must_use]
pub fn resealed(data: &[u8]) -> Vec<u8> {
    let mut bytes = data.to_vec();
    if let Some(n) = bytes.len().checked_sub(8) {
        let sum = checksum(&bytes[..n]);
        bytes[n..].copy_from_slice(&sum.to_le_bytes());
    }
    bytes
}

/// Runs one input against `pool`.
pub fn run<E, M>(mut pool: EnvPool<E, M>, data: &[u8], build_id: Option<&str>)
where
    E: Persist,
    E::Config: CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    M: PersistMask<E> + MaskProvider<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    let limits = DecodeLimits {
        max_instances: 8,
        max_decoded_bytes: 1 << 20,
    };
    for bytes in [data.to_vec(), resealed(data)] {
        let Ok(records) = pool.decode_with(&bytes, limits, build_id) else {
            continue;
        };
        let start = bytes.len() - 8 - records.iter().map(encode_to_vec).map(|v| v.len()).sum::<usize>();
        let payload: Vec<u8> = records.iter().flat_map(encode_to_vec).collect();
        assert_eq!(payload, bytes[start..bytes.len() - 8], "decoded records are not canonical");
        let ids: Vec<u32> = (0..u32::try_from(records.len()).unwrap_or(0)).collect();
        if pool.import(&ids, &records).is_ok() {
            let rows = pool.outputs().unwrap().rewards().len();
            let _ = pool.step(ActionInput { discrete: &vec![0; rows], continuous: &[] });
        }
    }
}
