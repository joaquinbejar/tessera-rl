// SPDX-License-Identifier: MIT OR Apache-2.0

//! `tessera_rl.League`'s native side (ADR-0088 section 10; #384 head E3):
//! every league is a `SharedLeague` (E1), kept in a table of handles so
//! that several pools made with `make(..., league=)` share it. The Python
//! class holds the handle and drops it when collected.
//!
//! The Python layer passes a league's configuration as one string, one
//! item per line:
//!
//! ```text
//! ratings elo_k rate_population(0|1) [beta tau draw_margin]
//! policy heuristic | policy external <tag> | policy onnx <32 hex digits>
//! role <matchmaking> [self_play <share> <tag>] [snapshot every <n> |
//!      snapshot win_rate <threshold> <min_games>] [tags <base>]
//! ```
//!
//! with `<matchmaking>` one of `uniform`, `pfsp:hard`, `pfsp:variance`,
//! `pfsp:linear`, `elo:<temperature>`. Floats are Python's `repr`, which
//! Rust parses to the same bits.

use std::sync::Mutex;

use tessera_rl_league::trueskill::{Params, Rating};
use tessera_rl_league::{
    Matchmaking, PfspWeighting, PolicyKind, Population, RoleConfig, SelfPlay, SharedConfig,
    SharedLeague, SnapshotRule, elo,
};

use crate::api::Kwargs;
use crate::errors::{Kind, error};
use crate::ffi::capi::{self, CResult, Gil, Mine};

/// The leagues Python holds, by handle; a dropped league leaves `None`.
static LEAGUES: Mutex<Vec<Option<SharedLeague>>> = Mutex::new(Vec::new());

/// The league of `handle`, cloned (it shares its state).
pub(crate) fn league(handle: u64) -> Result<SharedLeague, String> {
    let table = LEAGUES
        .lock()
        .map_err(|_| "the league table was poisoned".to_owned())?;
    usize::try_from(handle)
        .ok()
        .and_then(|h| table.get(h))
        .and_then(Option::as_ref)
        .cloned()
        .ok_or_else(|| format!("no league with handle {handle}"))
}

fn keep(league: SharedLeague) -> Result<u64, String> {
    let mut table = LEAGUES
        .lock()
        .map_err(|_| "the league table was poisoned".to_owned())?;
    table.push(Some(league));
    u64::try_from(table.len() - 1).map_err(|_| "too many leagues".to_owned())
}

fn num<T: core::str::FromStr>(s: Option<&str>, what: &str) -> Result<T, String> {
    s.and_then(|v| v.parse().ok())
        .ok_or_else(|| format!("league config: bad {what}"))
}

fn matchmaking(s: &str) -> Result<Matchmaking, String> {
    Ok(match s {
        "uniform" => Matchmaking::Uniform,
        "pfsp:hard" => Matchmaking::Pfsp(PfspWeighting::Hard),
        "pfsp:variance" => Matchmaking::Pfsp(PfspWeighting::Variance),
        "pfsp:linear" => Matchmaking::Pfsp(PfspWeighting::Linear),
        other => match other.strip_prefix("elo:") {
            Some(t) => Matchmaking::Elo {
                temperature: num(Some(t), "elo temperature")?,
            },
            None => return Err(format!("league config: unknown matchmaking {other:?}")),
        },
    })
}

fn hex16(s: &str) -> Result<[u8; 16], String> {
    let bad = || format!("league config: an onnx content hash is 32 hex digits, not {s:?}");
    if s.len() != 32 {
        return Err(bad());
    }
    let mut out = [0u8; 16];
    for (i, b) in out.iter_mut().enumerate() {
        *b = s
            .get(2 * i..2 * i + 2)
            .and_then(|h| u8::from_str_radix(h, 16).ok())
            .ok_or_else(bad)?;
    }
    Ok(out)
}

/// The league of a configuration string (the module doc's format).
///
/// # Errors
///
/// The message of a malformed item, or of the league's own refusal.
pub(crate) fn parse(spec: &str) -> Result<SharedLeague, String> {
    let mut config = SharedConfig {
        elo_k: 16.0,
        trueskill: Params::default(),
        rate_population: false,
        roles: Vec::new(),
    };
    let mut population = Population::new();
    for line in spec.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let mut w = line.split_whitespace();
        match w.next() {
            Some("ratings") => {
                config.elo_k = num(w.next(), "elo_k")?;
                config.rate_population = num::<u8>(w.next(), "rate_population")? == 1;
                if let Some(beta) = w.next() {
                    config.trueskill = Params {
                        beta: num(Some(beta), "trueskill beta")?,
                        tau: num(w.next(), "trueskill tau")?,
                        draw_margin: num(w.next(), "trueskill draw_margin")?,
                    };
                }
            }
            Some("policy") => {
                let kind = match w.next() {
                    Some("heuristic") => PolicyKind::Heuristic,
                    Some("external") => PolicyKind::External {
                        tag: num(w.next(), "external tag")?,
                    },
                    Some("onnx") => PolicyKind::Onnx {
                        content_hash: hex16(w.next().unwrap_or(""))?,
                    },
                    other => return Err(format!("league config: unknown policy {other:?}")),
                };
                population
                    .add(kind, elo::INITIAL, Rating::initial(), 0)
                    .map_err(|e| e.to_string())?;
            }
            Some("role") => {
                let mut role = RoleConfig::new(matchmaking(w.next().unwrap_or(""))?);
                while let Some(key) = w.next() {
                    match key {
                        "self_play" => {
                            role.self_play = Some(SelfPlay {
                                share: num(w.next(), "self_play share")?,
                                tag: num(w.next(), "self_play tag")?,
                            });
                        }
                        "snapshot" => {
                            role.snapshot = Some(match w.next() {
                                Some("every") => {
                                    SnapshotRule::EveryRounds(num(w.next(), "snapshot every")?)
                                }
                                Some("win_rate") => SnapshotRule::WinRate {
                                    threshold: num(w.next(), "snapshot threshold")?,
                                    min_games: num(w.next(), "snapshot min_games")?,
                                },
                                other => {
                                    return Err(format!(
                                        "league config: unknown snapshot rule {other:?}"
                                    ));
                                }
                            });
                        }
                        "tags" => role.snapshot_tags = num(w.next(), "snapshot tags")?,
                        other => return Err(format!("league config: unknown role item {other:?}")),
                    }
                }
                config.roles.push(role);
            }
            other => return Err(format!("league config: unknown item {other:?}")),
        }
    }
    SharedLeague::new(config, population).map_err(|e| e.to_string())
}

fn handle(g: Gil, kw: &Kwargs<'_>) -> CResult<u64> {
    kw.u64(g, "league")?
        .ok_or_else(|| error(g, Kind::Argument, "missing argument \"league\"".to_owned()))
}

fn config_error(g: Gil, message: String) -> crate::ffi::capi::Raised {
    error(g, Kind::Config, message)
}

/// `league_new(spec=) -> int`.
pub(crate) fn league_new<'py>(g: Gil<'py>, kwargs: Option<capi::Ref<'_>>) -> CResult<Mine<'py>> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&["spec"]])?;
    let spec = kw.string(g, "spec")?.unwrap_or_default();
    let league = parse(&spec).map_err(|e| config_error(g, e))?;
    capi::int_u64(g, keep(league).map_err(|e| config_error(g, e))?)
}

/// `league_load(data=) -> int`.
pub(crate) fn league_load<'py>(g: Gil<'py>, kwargs: Option<capi::Ref<'_>>) -> CResult<Mine<'py>> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&["data"]])?;
    let data = kw
        .get("data")
        .ok_or_else(|| error(g, Kind::Argument, "missing argument \"data\"".to_owned()))?;
    let bytes = capi::bytes_of(g, data, "data")?;
    let league = SharedLeague::load(bytes).map_err(|e| error(g, Kind::Persist, e.to_string()))?;
    capi::int_u64(g, keep(league).map_err(|e| config_error(g, e))?)
}

/// `league_save(league=) -> bytes`.
pub(crate) fn league_save<'py>(g: Gil<'py>, kwargs: Option<capi::Ref<'_>>) -> CResult<Mine<'py>> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&["league"]])?;
    let l = league(handle(g, &kw)?).map_err(|e| error(g, Kind::Argument, e))?;
    let bytes = l
        .save()
        .map_err(|e| error(g, Kind::Persist, e.to_string()))?;
    capi::bytes(g, &bytes)
}

/// `league_export(league=) -> str`.
pub(crate) fn league_export<'py>(g: Gil<'py>, kwargs: Option<capi::Ref<'_>>) -> CResult<Mine<'py>> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&["league"]])?;
    let l = league(handle(g, &kw)?).map_err(|e| error(g, Kind::Argument, e))?;
    let text = l
        .export()
        .map_err(|e| error(g, Kind::Persist, e.to_string()))?;
    capi::string(g, &text)
}

/// `league_drop(league=)`: forgets the handle; pools made with it keep
/// their share of the league.
pub(crate) fn league_drop<'py>(g: Gil<'py>, kwargs: Option<capi::Ref<'_>>) -> CResult<Mine<'py>> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&["league"]])?;
    let h = handle(g, &kw)?;
    if let (Ok(mut table), Ok(i)) = (LEAGUES.lock(), usize::try_from(h))
        && let Some(slot) = table.get_mut(i)
    {
        *slot = None;
    }
    Ok(capi::none(g))
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn test_league_spec_parses_and_refuses() {
        let ok = "ratings 16.0 1 4.166666666666667 0.08333333333333333 0.0\n\
                  policy heuristic\npolicy external 7\n\
                  policy onnx 07070707070707070707070707070707\n\
                  role pfsp:hard self_play 0.25 99 snapshot every 2 tags 500\n\
                  role elo:40.0 snapshot win_rate 0.6 10";
        assert!(parse(ok).is_ok());
        for bad in [
            "",
            "policy heuristic",
            "policy heuristic\nrole nope",
            "policy onnx 07\nrole uniform",
            "policy heuristic\nrole uniform self_play 1.5 1",
            "policy heuristic\nrole uniform tags x",
            "bogus",
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }
}
