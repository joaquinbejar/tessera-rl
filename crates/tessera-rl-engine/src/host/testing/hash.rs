// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-call hashes of a host pool's outputs, one per output field, for the
//! checks of #351. FNV-1a 64: stable across platforms and releases, so a
//! crate can commit the hashes as goldens.

use tessera_rl_core::Bool8;

use crate::outputs::Outputs;

/// FNV-1a 64 offset basis.
const BASIS: u64 = 0xCBF2_9CE4_8422_2325;
/// FNV-1a 64 prime.
const PRIME: u64 = 0x0100_0000_01B3;

/// An FNV-1a 64 hasher.
#[derive(Debug, Clone, Copy)]
struct Fnv(u64);

impl Fnv {
    const fn new() -> Self {
        Self(BASIS)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            // FNV is defined modulo 2^64.
            self.0 = self.0.wrapping_mul(PRIME);
        }
    }

    fn word(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
}

/// Which outputs a hash covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cover {
    /// Every output.
    All,
    /// Every output but the slot generations, which count spawns over the
    /// pool's life rather than the episode's.
    Episode,
}

/// The hash of each output field after one call, in a fixed order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FieldHashes(Vec<(&'static str, u64)>);

impl FieldHashes {
    /// The hashes of `o`'s fields.
    pub(crate) fn of(o: &Outputs, cover: Cover) -> Self {
        let mut out = Vec::with_capacity(24);
        out.push(("observations", leaves(|k| o.observation(k))));
        out.push((
            "terminal observations",
            leaves(|k| o.terminal_observation(k)),
        ));
        out.push(("rewards", floats(o.rewards())));
        out.push(("agent terminated flags", flags(o.agent_terminated())));
        out.push(("agent truncated flags", flags(o.agent_truncated())));
        out.push(("alive flags (spawns and deaths)", flags(o.alive())));
        out.push(("acting flags", flags(o.acting())));
        if cover == Cover::All {
            out.push(("slot generations", words(o.generations())));
        }
        out.push(("episode terminated flags", flags(o.episode_terminated())));
        out.push(("episode truncated flags", flags(o.episode_truncated())));
        out.push(("aborted flags", flags(o.aborted())));
        out.push(("abort codes", words(o.abort_code())));
        out.push(("pending reset flags", flags(o.pending_reset())));
        out.push(("episode returns", floats(o.episode_return())));
        out.push(("episode lengths", floats(o.episode_length())));
        out.push(("illegal action counts", floats(o.episode_illegal_actions())));
        out.push((
            "invalid agent event counts",
            floats(o.episode_invalid_agent_events()),
        ));
        out.push(("masks", flags(o.masks())));
        out.push(("refused flags", flags(o.refused())));
        out.push(("refuse codes", words(o.refuse_code())));
        Self(out)
    }

    /// One hash of every field.
    #[must_use]
    pub(crate) fn total(&self) -> u64 {
        let mut h = Fnv::new();
        for (_, v) in &self.0 {
            h.word(*v);
        }
        h.0
    }

    /// The first field whose hash differs, if any.
    #[must_use]
    pub(crate) fn first_difference(&self, other: &Self) -> Option<&'static str> {
        if self.0.len() != other.0.len() {
            return Some("output fields");
        }
        self.0
            .iter()
            .zip(&other.0)
            .find(|(a, b)| a.1 != b.1)
            .map(|(a, _)| a.0)
    }
}

/// Every observation leaf `get(0..)` returns, each with its length.
fn leaves<'a>(get: impl Fn(usize) -> Option<&'a [u8]>) -> u64 {
    let mut h = Fnv::new();
    let mut k = 0;
    while let Some(leaf) = get(k) {
        h.word(u64::try_from(leaf.len()).unwrap_or(u64::MAX));
        h.bytes(leaf);
        k += 1;
    }
    h.0
}

/// Bit patterns: a NaN payload or a signed zero changes the hash.
fn floats(v: &[f32]) -> u64 {
    let mut h = Fnv::new();
    h.word(u64::try_from(v.len()).unwrap_or(u64::MAX));
    for x in v {
        h.bytes(&x.to_bits().to_le_bytes());
    }
    h.0
}

fn flags(v: &[Bool8]) -> u64 {
    let mut h = Fnv::new();
    h.word(u64::try_from(v.len()).unwrap_or(u64::MAX));
    for x in v {
        h.bytes(&[u8::from(x.get())]);
    }
    h.0
}

fn words(v: &[u32]) -> u64 {
    let mut h = Fnv::new();
    h.word(u64::try_from(v.len()).unwrap_or(u64::MAX));
    for x in v {
        h.bytes(&x.to_le_bytes());
    }
    h.0
}

#[cfg(test)]
mod tests {
    use super::Fnv;

    #[test]
    fn test_fnv_known_vectors_match_the_reference() {
        let mut h = Fnv::new();
        h.bytes(b"");
        assert_eq!(h.0, 0xCBF2_9CE4_8422_2325);
        let mut h = Fnv::new();
        h.bytes(b"a");
        assert_eq!(h.0, 0xAF63_DC4C_8601_EC8C);
    }
}
