// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo tessera check` (#351, SPEC 10.2): the crate's Rust checks, then
//! with `--python` its adapter tests.

use std::io::Write;
use std::path::Path;

use crate::args::CheckArgs;
use crate::error::ToolError;
use crate::{exec, python};

/// The files `check` needs.
const NEEDS: &[&str] = &["Cargo.toml", "tests/check.rs"];

/// Runs `cargo test --test check` in `dir` (with `TESSERA_BLESS=1` for
/// `--bless`), then the Python checks with `--python`. Prints one line per
/// check family to `out`.
///
/// # Errors
///
/// [`ToolError::NotACrate`], or the first failing command.
pub fn run(args: CheckArgs, dir: &Path, out: &mut impl Write) -> Result<(), ToolError> {
    exec::ensure_crate(dir, NEEDS)?;
    let mut cmd = exec::command(exec::cargo(), dir);
    cmd.args(["test", "--test", "check"])
        .args(args.cargo.args());
    if args.bless {
        cmd.env("TESSERA_BLESS", "1");
    } else {
        cmd.env_remove("TESSERA_BLESS");
    }
    exec::run(&mut cmd)?;
    let rust = if args.bless {
        "rust checks: ok (tests/golden.json recorded: commit it)"
    } else {
        "rust checks (determinism, continuation, spaces, invariants, goldens): ok"
    };
    // Output to a closed stdout is not a check failure.
    let _ = writeln!(out, "{rust}");
    if args.python {
        python::run(args, dir)?;
        let _ = writeln!(out, "python adapter checks: ok");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::run;
    use crate::args::CheckArgs;
    use crate::error::ToolError;

    #[test]
    fn test_run_outside_a_scaffold_refused() {
        let dir = std::env::temp_dir().join(format!("cargo-tessera-empty-{}", std::process::id()));
        let mut out = Vec::new();
        let err = run(CheckArgs::default(), Path::new(&dir), &mut out).unwrap_err();
        assert!(
            matches!(
                err,
                ToolError::NotACrate {
                    missing: "Cargo.toml",
                    ..
                }
            ),
            "{err}"
        );
    }
}
