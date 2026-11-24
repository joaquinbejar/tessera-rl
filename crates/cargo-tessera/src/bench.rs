// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo tessera bench` (#351, SPEC 10.2): runs the crate's criterion
//! bench and prints a summary from criterion's estimates.

use std::fmt::Write as _;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::args::BenchArgs;
use crate::error::ToolError;
use crate::exec;

/// The files `bench` needs.
const NEEDS: &[&str] = &["Cargo.toml", "benches/step.rs"];

/// The rows the scaffold's bench writes: group, function, pool sizes.
const ROWS: &[(&str, &str, &[u32])] = &[
    ("step", "inline", &[1, 1024, 8192]),
    ("step", "workers4", &[1, 1024, 8192]),
    ("snapshot", "save", &[1024]),
    ("snapshot", "restore", &[1024]),
];

/// One summarized estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// Criterion group.
    pub group: &'static str,
    /// Function within the group.
    pub function: &'static str,
    /// Instances.
    pub instances: u32,
    /// Mean time of one iteration (one call over every instance), ns.
    pub mean_ns: f64,
}

/// Runs `cargo bench --bench step` in `dir` (`-- --quick` for `--quick`)
/// and prints the summary table to `out`.
///
/// # Errors
///
/// [`ToolError::NotACrate`], a failing bench, or
/// [`ToolError::NoEstimates`].
pub fn run(args: BenchArgs, dir: &Path, out: &mut impl Write) -> Result<(), ToolError> {
    exec::ensure_crate(dir, NEEDS)?;
    let mut cmd = exec::command(exec::cargo(), dir);
    cmd.args(["bench", "--bench", "step"])
        .args(args.cargo.args());
    if args.quick {
        cmd.args(["--", "--quick"]);
    }
    exec::run(&mut cmd)?;
    let root = criterion_dir(dir);
    let rows = collect(&root);
    if rows.is_empty() {
        return Err(ToolError::NoEstimates { dir: root });
    }
    // Output to a closed stdout is not a bench failure.
    let _ = out.write_all(table(&rows).as_bytes());
    Ok(())
}

/// Where criterion wrote: `$CRITERION_HOME`, else `$CARGO_TARGET_DIR/criterion`,
/// else `target/criterion` (relative paths against `dir`).
fn criterion_dir(dir: &Path) -> PathBuf {
    let from_env = |name| {
        std::env::var_os(name)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let path = from_env("CRITERION_HOME").unwrap_or_else(|| {
        from_env("CARGO_TARGET_DIR")
            .unwrap_or_else(|| PathBuf::from("target"))
            .join("criterion")
    });
    if path.is_absolute() {
        path
    } else {
        dir.join(path)
    }
}

/// Every estimate of [`ROWS`] found under `root`.
fn collect(root: &Path) -> Vec<Row> {
    let mut rows = Vec::new();
    for &(group, function, sizes) in ROWS {
        for &n in sizes {
            let file = root
                .join(group)
                .join(function)
                .join(n.to_string())
                .join("new")
                .join("estimates.json");
            let Some(mean_ns) = std::fs::read_to_string(&file)
                .ok()
                .and_then(|text| mean_ns(&text))
            else {
                continue;
            };
            rows.push(Row {
                group,
                function,
                instances: n,
                mean_ns,
            });
        }
    }
    rows
}

/// The mean's point estimate of a criterion `estimates.json`, in ns.
#[must_use]
pub fn mean_ns(json: &str) -> Option<f64> {
    let after = json.split_once("\"mean\"")?.1;
    let after = after.split_once("\"point_estimate\"")?.1;
    let after = after.trim_start().strip_prefix(':')?.trim_start();
    let end = after
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(after.len());
    after
        .get(..end)?
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v > 0.0)
}

/// The summary table: instance-steps per second for `step`, ns per
/// instance for `snapshot`.
#[must_use]
pub fn table(rows: &[Row]) -> String {
    let mut out = String::from("group     mode       instances    per call (us)    rate\n");
    for r in rows {
        let n = f64::from(r.instances);
        let rate = if r.group == "step" {
            format!("{:.0} instance-steps/s", n * 1e9 / r.mean_ns)
        } else {
            format!("{:.1} ns/instance", r.mean_ns / n)
        };
        // Writing to a String cannot fail.
        let _ = writeln!(
            out,
            "{:<9} {:<10} {:>9} {:>16.2}    {rate}",
            r.group,
            r.function,
            r.instances,
            r.mean_ns / 1e3
        );
    }
    out.push_str("(criterion means on this machine; see target/criterion for the full reports)\n");
    out
}

#[cfg(test)]
mod tests {
    use super::{Row, mean_ns, table};

    const ESTIMATES: &str = r#"{"mean":{"confidence_interval":{"confidence_level":0.95,"lower_bound":1180.5,"upper_bound":1250.25},"point_estimate":1210.75,"standard_error":17.5},"median":{"confidence_interval":{"confidence_level":0.95,"lower_bound":1.0,"upper_bound":2.0},"point_estimate":1200.0,"standard_error":3.0}}"#;

    #[test]
    fn test_mean_ns_reads_the_mean_point_estimate() {
        assert_eq!(mean_ns(ESTIMATES), Some(1210.75));
        assert_eq!(
            mean_ns(r#"{"mean":{"point_estimate": 2.5e3}}"#),
            Some(2500.0)
        );
    }

    #[test]
    fn test_mean_ns_malformed_is_none() {
        for text in [
            "",
            "{}",
            r#"{"mean":{}}"#,
            r#"{"mean":{"point_estimate":"x"}}"#,
            r#"{"mean":{"point_estimate":0}}"#,
        ] {
            assert_eq!(mean_ns(text), None, "{text}");
        }
    }

    #[test]
    fn test_table_rates_per_group() {
        let rows = [
            Row {
                group: "step",
                function: "inline",
                instances: 1024,
                mean_ns: 1024.0,
            },
            Row {
                group: "snapshot",
                function: "save",
                instances: 1024,
                mean_ns: 2048.0,
            },
        ];
        let text = table(&rows);
        assert!(text.contains("1000000000 instance-steps/s"), "{text}");
        assert!(text.contains("2.0 ns/instance"), "{text}");
    }
}
