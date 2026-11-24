// SPDX-License-Identifier: MIT OR Apache-2.0

//! The population of opponent policies (ADR-0088 item 1).

use crate::LeagueError;
use crate::trueskill;

/// A policy's dense id: its insertion index in the population.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PolicyId(pub u32);

impl PolicyId {
    /// The id as an index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// How the engine computes a policy's actions (ADR-0088 item 2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolicyKind {
    /// A trainer-side checkpoint, named by the trainer.
    External {
        /// The trainer's tag for the checkpoint.
        tag: u64,
    },
    /// The environment's heuristic (ADR-0079 section 4).
    Heuristic,
    /// An ONNX model, identified by its content hash (ADR-0057).
    Onnx {
        /// The model's content hash.
        content_hash: [u8; 16],
    },
}

/// One entry of the population.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Policy {
    /// How its actions are computed.
    pub kind: PolicyKind,
    /// Its Elo rating.
    pub elo: f64,
    /// Its `TrueSkill` rating.
    pub trueskill: trueskill::Rating,
    /// Role tags, a bit set the league configuration defines.
    pub roles: u32,
    /// Retired entries keep their id and are never matched.
    pub retired: bool,
}

/// The population: entries in insertion order, never removed.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Population {
    entries: Vec<Policy>,
}

impl Population {
    /// An empty population.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Adds a policy with the given initial ratings and returns its id.
    ///
    /// # Errors
    ///
    /// [`LeagueError::PopulationFull`] past `u32::MAX` entries.
    pub fn add(
        &mut self,
        kind: PolicyKind,
        elo: f64,
        trueskill: trueskill::Rating,
        roles: u32,
    ) -> Result<PolicyId, LeagueError> {
        let id = u32::try_from(self.entries.len()).map_err(|_| LeagueError::PopulationFull)?;
        if id == u32::MAX {
            return Err(LeagueError::PopulationFull);
        }
        self.entries.push(Policy {
            kind,
            elo,
            trueskill,
            roles,
            retired: false,
        });
        Ok(PolicyId(id))
    }

    /// Retires a policy: it keeps its id and ratings and is never matched.
    ///
    /// # Errors
    ///
    /// [`LeagueError::UnknownPolicy`] for an id outside the population.
    pub fn retire(&mut self, id: PolicyId) -> Result<(), LeagueError> {
        self.get_mut(id)?.retired = true;
        Ok(())
    }

    /// The entry of `id`.
    ///
    /// # Errors
    ///
    /// [`LeagueError::UnknownPolicy`] for an id outside the population.
    pub fn get(&self, id: PolicyId) -> Result<&Policy, LeagueError> {
        self.entries
            .get(id.index())
            .ok_or(LeagueError::UnknownPolicy(id.0))
    }

    pub(crate) fn get_mut(&mut self, id: PolicyId) -> Result<&mut Policy, LeagueError> {
        self.entries
            .get_mut(id.index())
            .ok_or(LeagueError::UnknownPolicy(id.0))
    }

    /// The entries, in id order.
    #[must_use]
    pub fn entries(&self) -> &[Policy] {
        &self.entries
    }

    /// The number of entries, retired ones included.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the population has no entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// A digest of the population's identity: every entry's kind and
    /// retired flag, in id order (FNV-1a 64 over a canonical byte form).
    /// Ratings are not part of it: they change at every barrier.
    #[must_use]
    pub fn digest(&self) -> u64 {
        let mut h = Fnv::new();
        h.bytes(&(self.entries.len() as u64).to_le_bytes());
        for p in &self.entries {
            match p.kind {
                PolicyKind::External { tag } => {
                    h.bytes(&[0]);
                    h.bytes(&tag.to_le_bytes());
                }
                PolicyKind::Heuristic => h.bytes(&[1]),
                PolicyKind::Onnx { content_hash } => {
                    h.bytes(&[2]);
                    h.bytes(&content_hash);
                }
            }
            h.bytes(&p.roles.to_le_bytes());
            h.bytes(&[u8::from(p.retired)]);
        }
        h.0
    }
}

/// FNV-1a 64. Its multiplication wraps by definition of the hash.
struct Fnv(u64);

impl Fnv {
    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}
