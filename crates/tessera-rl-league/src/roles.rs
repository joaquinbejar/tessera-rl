// SPDX-License-Identifier: MIT OR Apache-2.0

//! League roles (ADR-0088 section 6; #384 head E1): each role is one
//! learner, served by one pool or more, with its own matchmaking rule, an
//! optional self-play share and snapshot rules.

use crate::{LeagueError, Matchmaking};

/// A role's self-play entry: a population `External { tag }` entry, drawn
/// with a fixed share of the role's draws, that the trainer serves with the
/// role's current weights. Its results count in its pair counts only, never
/// in the ratings (a game against oneself has an expected score of 0.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelfPlay {
    /// The share of the role's draws, in (0, 1): exactly
    /// `floor(share x 2^64)` of the threshold range.
    pub share: f64,
    /// The trainer's tag for the role's current weights.
    pub tag: u64,
}

/// When a role adds a frozen snapshot of its learner to the population.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SnapshotRule {
    /// After every `n` commits (`n` at least 1).
    EveryRounds(u32),
    /// When the learner's win rate over the games since its last snapshot
    /// (draws counting half) reaches `threshold`, after at least
    /// `min_games` games.
    WinRate {
        /// The win rate, in (0, 1].
        threshold: f64,
        /// The fewest games before it is checked.
        min_games: u64,
    },
}

/// One role of a shared league.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoleConfig {
    /// How the role's opponents are weighted.
    pub matchmaking: Matchmaking,
    /// The role's self-play entry, if any.
    pub self_play: Option<SelfPlay>,
    /// The role's snapshot rule, if any.
    pub snapshot: Option<SnapshotRule>,
    /// The tag of the role's first snapshot; the `k`-th is
    /// `snapshot_tags + k`, and the barrier report lists each one so the
    /// trainer saves that checkpoint under it (ADR-0018).
    pub snapshot_tags: u64,
}

impl RoleConfig {
    /// A role with `matchmaking`, no self-play and no snapshots.
    #[must_use]
    pub const fn new(matchmaking: Matchmaking) -> Self {
        Self {
            matchmaking,
            self_play: None,
            snapshot: None,
            snapshot_tags: 0,
        }
    }

    pub(crate) fn check(&self) -> Result<(), LeagueError> {
        if let Some(s) = self.self_play
            && !(s.share > 0.0 && s.share < 1.0)
        {
            return Err(LeagueError::InvalidConfig(
                "a self-play share must be in (0, 1)",
            ));
        }
        match self.snapshot {
            Some(SnapshotRule::EveryRounds(0)) => {
                Err(LeagueError::InvalidConfig("a snapshot every 0 rounds"))
            }
            Some(SnapshotRule::WinRate { threshold, .. })
                if !(threshold > 0.0 && threshold <= 1.0) =>
            {
                Err(LeagueError::InvalidConfig(
                    "a snapshot win-rate threshold must be in (0, 1]",
                ))
            }
            _ => Ok(()),
        }
    }
}

/// Whether an entry with role bits `roles` may be drawn by role `role`: an
/// entry with no bits serves every role.
#[must_use]
pub(crate) const fn serves(roles: u32, role: usize) -> bool {
    roles == 0 || (role < 32 && (roles >> role) & 1 == 1)
}
