// SPDX-License-Identifier: MIT OR Apache-2.0

//! The league's JSON export (ADR-0088 item 7.5; #384 head E2): for metrics
//! only, never read back. RFC 8259: strings escaped, floats in Rust's
//! shortest round-trip form, a non-finite value refused, keys in a fixed
//! order.

use core::fmt::Write;

use crate::persist::Saved;
use crate::{LeagueError, PairCounts, PolicyKind};

/// Appends `s` as an RFC 8259 string.
pub(crate) fn string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn key(out: &mut String, k: &str) {
    string(out, k);
    out.push(':');
}

fn number(out: &mut String, v: f64) -> Result<(), LeagueError> {
    if !v.is_finite() {
        return Err(LeagueError::Persist("a non-finite value in the export"));
    }
    let _ = write!(out, "{v}");
    Ok(())
}

fn counts(out: &mut String, c: &PairCounts) {
    let _ = write!(
        out,
        "{{\"wins\":{},\"draws\":{},\"losses\":{}}}",
        c.wins, c.draws, c.losses
    );
}

fn rating(out: &mut String, mu: f64, sigma: f64) -> Result<(), LeagueError> {
    out.push('{');
    key(out, "mu");
    number(out, mu)?;
    out.push(',');
    key(out, "sigma");
    number(out, sigma)?;
    out.push('}');
    Ok(())
}

/// The export of a saved league.
pub(crate) fn export(saved: &Saved) -> Result<String, LeagueError> {
    let mut out = String::new();
    out.push('{');
    key(&mut out, "round");
    let _ = write!(out, "{}", saved.round);
    out.push(',');
    key(&mut out, "frozen");
    out.push_str(if saved.frozen { "true" } else { "false" });
    out.push(',');
    key(&mut out, "roles");
    out.push('[');
    for (i, r) in saved.roles.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('{');
        key(&mut out, "elo");
        number(&mut out, r.elo)?;
        out.push(',');
        key(&mut out, "trueskill");
        rating(&mut out, r.trueskill.mu, r.trueskill.sigma)?;
        out.push(',');
        key(&mut out, "counts");
        out.push('[');
        for (j, c) in r.counts.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            counts(&mut out, c);
        }
        out.push(']');
        out.push(',');
        key(&mut out, "snapshots");
        let _ = write!(out, "{}", r.snapshots);
        out.push('}');
    }
    out.push(']');
    out.push(',');
    key(&mut out, "population");
    out.push('[');
    for (i, p) in saved.population.entries().iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('{');
        key(&mut out, "id");
        let _ = write!(out, "{i}");
        out.push(',');
        key(&mut out, "kind");
        match p.kind {
            PolicyKind::Heuristic => string(&mut out, "heuristic"),
            PolicyKind::External { tag } => {
                out.push('{');
                key(&mut out, "external");
                let _ = write!(out, "{tag}");
                out.push('}');
            }
            PolicyKind::Onnx { content_hash } => {
                out.push('{');
                key(&mut out, "onnx");
                let mut hex = String::with_capacity(32);
                for b in content_hash {
                    let _ = write!(hex, "{b:02x}");
                }
                string(&mut out, &hex);
                out.push('}');
            }
        }
        out.push(',');
        key(&mut out, "elo");
        number(&mut out, p.elo)?;
        out.push(',');
        key(&mut out, "trueskill");
        rating(&mut out, p.trueskill.mu, p.trueskill.sigma)?;
        out.push(',');
        key(&mut out, "roles");
        let _ = write!(out, "{}", p.roles);
        out.push(',');
        key(&mut out, "retired");
        out.push_str(if p.retired { "true" } else { "false" });
        out.push('}');
    }
    out.push(']');
    out.push('}');
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::string;

    #[test]
    fn test_export_strings_are_rfc8259_escaped() {
        let mut out = String::new();
        string(&mut out, "a\"b\\c\nd\re\tf\u{8}g\u{c}h\u{1}i\u{1f}j/é");
        assert_eq!(out, "\"a\\\"b\\\\c\\nd\\re\\tf\\bg\\fh\\u0001i\\u001fj/é\"");
    }
}
