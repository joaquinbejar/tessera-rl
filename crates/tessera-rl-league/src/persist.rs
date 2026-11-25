// SPDX-License-Identifier: MIT OR Apache-2.0

//! League persistence (ADR-0088 item 7.5 and REVIEW-headB-r1 ruling 4;
//! #384 head E2): the canonical bytes `League::save` and
//! `SharedLeague::save` write, version 1, little-endian, floats as their
//! bits.
//!
//! ```text
//! file      := magic:"TESLEAG\x1a" version:u32 kind:u8 body checksum:u64
//! kind      := 0 League | 1 SharedLeague
//! body      := ratings population roles state
//! ratings   := elo_k:f64 beta:f64 tau:f64 draw_margin:f64 rate_population:u8
//! population:= count:u32 entry*count
//! entry     := kind:(0:u8 tag:u64 | 1:u8 | 2:u8 hash:[u8;16]) elo:f64 mu:f64
//!              sigma:f64 roles:u32 retired:u8
//! roles     := count:u32 role*count
//! role      := matchmaking self_play snapshot snapshot_tags:u64 elo:f64
//!              mu:f64 sigma:f64 counts:(u64*3)*population
//!              self_play_id:(0:u8 | 1:u8 id:u32) rounds:u64 snapshots:u64
//!              since:u64*3 view_digest:u64 installed_digest:u64
//! matchmaking := 0:u8 | 1:u8 weighting:u8 | 2:u8 temperature:f64
//! self_play := 0:u8 | 1:u8 share:f64 tag:u64
//! snapshot  := 0:u8 | 1:u8 every:u32 | 2:u8 threshold:f64 min_games:u64
//! state     := frozen:u8 round:u64 members:count:u32 role:u32*count
//! checksum  := XXH3-64 of every byte before it
//! ```
//!
//! The checksum is verified before anything is parsed. Tables and views
//! are not stored: they are recomputed at load inside the canonical FP
//! scope, and each role's recomputed view must reproduce the stored
//! `view_digest` (TR, E2 plan ruling 3), else the load is refused. Every
//! count is checked against the bytes left before anything is allocated.

use tessera_rl_core::encoding::checksum;
use tessera_rl_engine::league::{LeagueView, OpponentKind};

use crate::roles::{RoleConfig, SelfPlay, SnapshotRule};
use crate::trueskill::{Params, Rating};
use crate::{LeagueError, Matchmaking, PairCounts, PfspWeighting, PolicyKind, Population};

/// The file magic.
pub const LEAGUE_MAGIC: [u8; 8] = *b"TESLEAG\x1a";
/// The format version.
pub const LEAGUE_FORMAT_VERSION: u32 = 1;

pub(crate) const KIND_LEAGUE: u8 = 0;
pub(crate) const KIND_SHARED: u8 = 1;

/// One role's saved state.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SavedRole {
    pub(crate) cfg: RoleConfig,
    pub(crate) elo: f64,
    pub(crate) trueskill: Rating,
    pub(crate) counts: Vec<PairCounts>,
    pub(crate) self_play: Option<u32>,
    pub(crate) rounds: u64,
    pub(crate) snapshots: u64,
    pub(crate) since: PairCounts,
    pub(crate) view_digest: u64,
    /// The digest of the view the role's pools hold (a shared league's
    /// pools lag one commit; a `League`'s equal `view_digest`).
    pub(crate) installed_digest: u64,
}

/// A league's saved state.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Saved {
    pub(crate) kind: u8,
    pub(crate) elo_k: f64,
    pub(crate) trueskill: Params,
    pub(crate) rate_population: bool,
    pub(crate) population: Population,
    pub(crate) roles: Vec<SavedRole>,
    pub(crate) frozen: bool,
    pub(crate) round: u64,
    pub(crate) members: Vec<u32>,
}

const fn bad(reason: &'static str) -> LeagueError {
    LeagueError::Persist(reason)
}

/// The digest of a view, as a save stores it: its kinds, thresholds, last
/// drawable entry and population digest.
#[must_use]
pub(crate) fn view_digest(view: &LeagueView) -> u64 {
    let mut b = Vec::new();
    b.extend_from_slice(&view.population_digest().to_le_bytes());
    b.extend_from_slice(&view.last().to_le_bytes());
    for k in view.kinds() {
        match k {
            OpponentKind::External { tag } => {
                b.push(0);
                b.extend_from_slice(&tag.to_le_bytes());
            }
            OpponentKind::Heuristic => b.push(1),
            OpponentKind::Onnx { content_hash } => {
                b.push(2);
                b.extend_from_slice(content_hash);
            }
        }
    }
    for t in view.thresholds() {
        b.extend_from_slice(&t.to_le_bytes());
    }
    checksum(&b)
}

fn f64s(out: &mut Vec<u8>, values: &[f64]) {
    for v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
}

fn counts(out: &mut Vec<u8>, c: &PairCounts) {
    for v in [c.wins, c.draws, c.losses] {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn len_u32(n: usize) -> Result<u32, LeagueError> {
    u32::try_from(n).map_err(|_| bad("a count over u32::MAX"))
}

impl Saved {
    /// The file's bytes.
    pub(crate) fn encode(&self) -> Result<Vec<u8>, LeagueError> {
        let mut out = Vec::new();
        out.extend_from_slice(&LEAGUE_MAGIC);
        out.extend_from_slice(&LEAGUE_FORMAT_VERSION.to_le_bytes());
        out.push(self.kind);
        let p = self.trueskill;
        f64s(&mut out, &[self.elo_k, p.beta, p.tau, p.draw_margin]);
        out.push(u8::from(self.rate_population));
        let entries = self.population.entries();
        out.extend_from_slice(&len_u32(entries.len())?.to_le_bytes());
        for e in entries {
            match e.kind {
                PolicyKind::External { tag } => {
                    out.push(0);
                    out.extend_from_slice(&tag.to_le_bytes());
                }
                PolicyKind::Heuristic => out.push(1),
                PolicyKind::Onnx { content_hash } => {
                    out.push(2);
                    out.extend_from_slice(&content_hash);
                }
            }
            f64s(&mut out, &[e.elo, e.trueskill.mu, e.trueskill.sigma]);
            out.extend_from_slice(&e.roles.to_le_bytes());
            out.push(u8::from(e.retired));
        }
        out.extend_from_slice(&len_u32(self.roles.len())?.to_le_bytes());
        for r in &self.roles {
            match r.cfg.matchmaking {
                Matchmaking::Uniform => out.push(0),
                Matchmaking::Pfsp(w) => out.extend_from_slice(&[
                    1,
                    match w {
                        PfspWeighting::Hard => 0,
                        PfspWeighting::Variance => 1,
                        PfspWeighting::Linear => 2,
                    },
                ]),
                Matchmaking::Elo { temperature } => {
                    out.push(2);
                    f64s(&mut out, &[temperature]);
                }
            }
            match r.cfg.self_play {
                None => out.push(0),
                Some(s) => {
                    out.push(1);
                    f64s(&mut out, &[s.share]);
                    out.extend_from_slice(&s.tag.to_le_bytes());
                }
            }
            match r.cfg.snapshot {
                None => out.push(0),
                Some(SnapshotRule::EveryRounds(n)) => {
                    out.push(1);
                    out.extend_from_slice(&n.to_le_bytes());
                }
                Some(SnapshotRule::WinRate {
                    threshold,
                    min_games,
                }) => {
                    out.push(2);
                    f64s(&mut out, &[threshold]);
                    out.extend_from_slice(&min_games.to_le_bytes());
                }
            }
            out.extend_from_slice(&r.cfg.snapshot_tags.to_le_bytes());
            f64s(&mut out, &[r.elo, r.trueskill.mu, r.trueskill.sigma]);
            if r.counts.len() != entries.len() {
                return Err(bad("a role's counts do not match the population"));
            }
            for c in &r.counts {
                counts(&mut out, c);
            }
            match r.self_play {
                None => out.push(0),
                Some(id) => {
                    out.push(1);
                    out.extend_from_slice(&id.to_le_bytes());
                }
            }
            out.extend_from_slice(&r.rounds.to_le_bytes());
            out.extend_from_slice(&r.snapshots.to_le_bytes());
            counts(&mut out, &r.since);
            out.extend_from_slice(&r.view_digest.to_le_bytes());
            out.extend_from_slice(&r.installed_digest.to_le_bytes());
        }
        out.push(u8::from(self.frozen));
        out.extend_from_slice(&self.round.to_le_bytes());
        out.extend_from_slice(&len_u32(self.members.len())?.to_le_bytes());
        for m in &self.members {
            out.extend_from_slice(&m.to_le_bytes());
        }
        let sum = checksum(&out);
        out.extend_from_slice(&sum.to_le_bytes());
        Ok(out)
    }

    /// Decodes a file of `kind`; the checksum is verified first.
    pub(crate) fn decode(bytes: &[u8], kind: u8) -> Result<Self, LeagueError> {
        let (body, sum) = bytes.split_last_chunk::<8>().ok_or(bad("truncated"))?;
        if checksum(body) != u64::from_le_bytes(*sum) {
            return Err(bad("checksum mismatch"));
        }
        let mut r = Reader { rest: body };
        if r.array::<8>()? != LEAGUE_MAGIC {
            return Err(bad("not a league file (magic)"));
        }
        let found = r.u32()?;
        if found != LEAGUE_FORMAT_VERSION {
            return Err(LeagueError::UnsupportedVersion { found });
        }
        if r.u8()? != kind {
            return Err(bad("another league kind"));
        }
        let elo_k = r.f64()?;
        let trueskill = Params {
            beta: r.f64()?,
            tau: r.f64()?,
            draw_margin: r.f64()?,
        };
        let rate_population = r.flag()?;
        // An entry takes at least 1 + 24 + 5 bytes.
        let n = r.count(30)?;
        let mut population = Population::new();
        for _ in 0..n {
            let kind = match r.u8()? {
                0 => PolicyKind::External { tag: r.u64()? },
                1 => PolicyKind::Heuristic,
                2 => PolicyKind::Onnx {
                    content_hash: r.array::<16>()?,
                },
                _ => return Err(bad("unknown policy kind")),
            };
            let elo = r.f64()?;
            let rating = Rating {
                mu: r.f64()?,
                sigma: r.f64()?,
            };
            let roles = r.u32()?;
            let retired = r.flag()?;
            let id = population.add(kind, elo, rating, roles)?;
            if retired {
                population.retire(id)?;
            }
        }
        // A role takes at least 3 + 8 + 24 + 1 + 16 + 24 + 16 bytes.
        let nr = r.count(92)?;
        let mut roles = Vec::new();
        roles.try_reserve_exact(nr).map_err(|_| bad("allocation"))?;
        for _ in 0..nr {
            roles.push(decode_role(&mut r, n)?);
        }
        let frozen = r.flag()?;
        let round = r.u64()?;
        let nm = r.count(4)?;
        let mut members = Vec::with_capacity(nm);
        for _ in 0..nm {
            members.push(r.u32()?);
        }
        if !r.rest.is_empty() {
            return Err(bad("trailing bytes"));
        }
        Ok(Self {
            kind,
            elo_k,
            trueskill,
            rate_population,
            population,
            roles,
            frozen,
            round,
            members,
        })
    }
}

/// One saved role of a population of `n` entries.
fn decode_role(r: &mut Reader<'_>, n: usize) -> Result<SavedRole, LeagueError> {
    let matchmaking = match r.u8()? {
        0 => Matchmaking::Uniform,
        1 => Matchmaking::Pfsp(match r.u8()? {
            0 => PfspWeighting::Hard,
            1 => PfspWeighting::Variance,
            2 => PfspWeighting::Linear,
            _ => return Err(bad("unknown PFSP weighting")),
        }),
        2 => Matchmaking::Elo {
            temperature: r.f64()?,
        },
        _ => return Err(bad("unknown matchmaking")),
    };
    let self_play = match r.u8()? {
        0 => None,
        1 => Some(SelfPlay {
            share: r.f64()?,
            tag: r.u64()?,
        }),
        _ => return Err(bad("bad self-play flag")),
    };
    let snapshot = match r.u8()? {
        0 => None,
        1 => Some(SnapshotRule::EveryRounds(r.u32()?)),
        2 => Some(SnapshotRule::WinRate {
            threshold: r.f64()?,
            min_games: r.u64()?,
        }),
        _ => return Err(bad("unknown snapshot rule")),
    };
    let cfg = RoleConfig {
        matchmaking,
        self_play,
        snapshot,
        snapshot_tags: r.u64()?,
    };
    let elo = r.f64()?;
    let trueskill = Rating {
        mu: r.f64()?,
        sigma: r.f64()?,
    };
    let mut c = Vec::new();
    if n.checked_mul(24).is_none_or(|b| b > r.rest.len()) {
        return Err(bad("truncated counts"));
    }
    c.try_reserve_exact(n).map_err(|_| bad("allocation"))?;
    for _ in 0..n {
        c.push(r.counts()?);
    }
    let self_play = match r.u8()? {
        0 => None,
        1 => Some(r.u32()?),
        _ => return Err(bad("bad self-play id flag")),
    };
    Ok(SavedRole {
        cfg,
        elo,
        trueskill,
        counts: c,
        self_play,
        rounds: r.u64()?,
        snapshots: r.u64()?,
        since: r.counts()?,
        view_digest: r.u64()?,
        installed_digest: r.u64()?,
    })
}

/// A bounds-checked reader.
struct Reader<'a> {
    rest: &'a [u8],
}

impl Reader<'_> {
    fn array<const N: usize>(&mut self) -> Result<[u8; N], LeagueError> {
        let (head, tail) = self.rest.split_first_chunk::<N>().ok_or(bad("truncated"))?;
        self.rest = tail;
        Ok(*head)
    }

    fn u8(&mut self) -> Result<u8, LeagueError> {
        Ok(u8::from_le_bytes(self.array()?))
    }

    fn flag(&mut self) -> Result<bool, LeagueError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(bad("a flag other than 0 or 1")),
        }
    }

    fn u32(&mut self) -> Result<u32, LeagueError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, LeagueError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn f64(&mut self) -> Result<f64, LeagueError> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn counts(&mut self) -> Result<PairCounts, LeagueError> {
        Ok(PairCounts {
            wins: self.u64()?,
            draws: self.u64()?,
            losses: self.u64()?,
        })
    }

    /// A `u32` count of items of at least `width` bytes each, checked
    /// against the bytes left.
    fn count(&mut self, width: usize) -> Result<usize, LeagueError> {
        let n = usize::try_from(self.u32()?).map_err(|_| bad("count"))?;
        match n.checked_mul(width) {
            Some(b) if b <= self.rest.len() => Ok(n),
            _ => Err(bad("a count larger than the file")),
        }
    }
}
