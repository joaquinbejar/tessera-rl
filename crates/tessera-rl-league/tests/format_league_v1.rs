// SPDX-License-Identifier: MIT OR Apache-2.0

//! The league state file, version 1 (`LEAGUE_FORMAT_VERSION`), against its
//! committed fixture `tests/fixtures/formats/league-v1.bin`: it reads back
//! to the league that wrote it, today's writer writes the same bytes, and a
//! newer or another-major version is refused with the typed error naming
//! it. A changed writer is resolved by a new version that keeps this file
//! and its read-back test, or by a fix; the fixture is regenerated only by
//! `test_format_league_v1_write_fixture` with `TESSERA_BLESS_FORMATS=1`.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test scenarios: a failure is the assertion"
)]

use tessera_rl_core::encoding::checksum;
use tessera_rl_league::trueskill::Rating;
use tessera_rl_league::{
    LEAGUE_FORMAT_VERSION, LEAGUE_MAGIC, League, LeagueConfig, LeagueError, Matchmaking, Outcome,
    PolicyId, PolicyKind, Population, Results, elo,
};

const FIXTURE: &[u8] = include_bytes!("fixtures/formats/league-v1.bin");

/// The league the fixture holds: three policies of every kind, one barrier
/// with four results.
fn league() -> League {
    let mut population = Population::new();
    for k in [
        PolicyKind::Heuristic,
        PolicyKind::External { tag: 7 },
        PolicyKind::Onnx {
            content_hash: [7; 16],
        },
    ] {
        population
            .add(k, elo::INITIAL, Rating::initial(), 0)
            .unwrap();
    }
    let mut league = League::new(
        LeagueConfig {
            matchmaking: Matchmaking::Elo { temperature: 40.0 },
            rate_population: true,
            ..LeagueConfig::default()
        },
        population,
    )
    .unwrap();
    let mut results = Results::new(2, 4).unwrap();
    for (i, id, o) in [
        (0, 0, Outcome::Win),
        (0, 1, Outcome::Loss),
        (1, 2, Outcome::Draw),
        (1, 0, Outcome::Win),
    ] {
        results.record(i, PolicyId(id), o).unwrap();
    }
    league.barrier(&mut results).unwrap();
    league
}

/// `file` with its version field (bytes 8..12) set to `v`, resealed.
fn with_version(file: &[u8], v: u32) -> Vec<u8> {
    let mut f = file.to_vec();
    f[8..12].copy_from_slice(&v.to_le_bytes());
    let body = f.len() - 8;
    let sum = checksum(&f[..body]);
    f[body..].copy_from_slice(&sum.to_le_bytes());
    f
}

#[test]
#[ignore = "writes the fixture; run with TESSERA_BLESS_FORMATS=1"]
fn test_format_league_v1_write_fixture() {
    if std::env::var("TESSERA_BLESS_FORMATS").as_deref() != Ok("1") {
        return;
    }
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/formats/league-v1.bin"
    );
    std::fs::write(path, league().save().unwrap()).unwrap();
}

#[test]
fn test_format_league_v1_fixture_reads_back() {
    assert_eq!(&FIXTURE[..8], &LEAGUE_MAGIC);
    assert_eq!(u32::from_le_bytes(FIXTURE[8..12].try_into().unwrap()), 1);
    assert_eq!(FIXTURE[12], 0, "kind: League");
    let loaded = League::load(FIXTURE).unwrap();
    assert_eq!(loaded, league());
}

#[test]
fn test_format_league_v1_writer_writes_the_fixture() {
    assert_eq!(LEAGUE_FORMAT_VERSION, 1);
    assert_eq!(league().save().unwrap(), FIXTURE);
}

#[test]
fn test_format_league_newer_and_other_major_versions_are_refused() {
    for found in [2, 1 << 16] {
        assert_eq!(
            League::load(&with_version(FIXTURE, found)),
            Err(LeagueError::UnsupportedVersion { found })
        );
    }
    let message = LeagueError::UnsupportedVersion { found: 2 }.to_string();
    assert!(
        message.contains('2') && message.contains("version"),
        "{message}"
    );
}
