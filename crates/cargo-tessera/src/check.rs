// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo tessera check` (#351, #628, SPEC 10.2): the crate's Rust checks,
//! then with `--python` its adapter tests. A `--kind env` crate, marked by
//! `[package.metadata.tessera] kind = "env"` in its Cargo.toml, has no
//! Python package: `--python` is refused there.

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
    let env_kind = is_env_kind(dir)?;
    if env_kind && args.python {
        return Err(ToolError::NoPython {
            dir: dir.to_path_buf(),
        });
    }
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
    } else if env_kind {
        "rust checks (determinism, continuation, snapshots, spaces, validation, goldens): ok"
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

/// Whether the crate in `dir` was scaffolded with `--kind env`: its
/// Cargo.toml has the line `kind = "env"` in `[package.metadata.tessera]`.
fn is_env_kind(dir: &Path) -> Result<bool, ToolError> {
    let path = dir.join("Cargo.toml");
    let text = std::fs::read_to_string(&path).map_err(|source| ToolError::Io {
        action: "reading",
        path: path.clone(),
        source,
    })?;
    Ok(env_marker(&text))
}

/// Whether `toml` has `kind = "env"` in `[package.metadata.tessera]`.
fn env_marker(toml: &str) -> bool {
    let mut inside = false;
    for line in toml.lines().map(str::trim) {
        if line.starts_with('[') {
            inside = line == "[package.metadata.tessera]";
        } else if inside && line.replace(' ', "") == "kind=\"env\"" {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{env_marker, run};
    use crate::args::CheckArgs;
    use crate::error::ToolError;

    #[test]
    fn test_env_marker_only_in_its_table() {
        assert!(env_marker(
            "[package]\nname = \"a\"\n\n[package.metadata.tessera]\n# x\nkind = \"env\"\n"
        ));
        assert!(!env_marker("[package]\nkind = \"env\"\n"));
        assert!(!env_marker("[package.metadata.tessera]\nkind = \"host\"\n"));
        assert!(!env_marker(
            "[package.metadata.tessera]\n[lib]\nkind = \"env\"\n"
        ));
    }

    #[test]
    fn test_run_python_on_an_env_crate_refused() {
        let dir =
            std::env::temp_dir().join(format!("cargo-tessera-envkind-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("tests")).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"a\"\n[package.metadata.tessera]\nkind = \"env\"\n",
        )
        .unwrap();
        std::fs::write(dir.join("tests/check.rs"), "").unwrap();
        let args = CheckArgs {
            python: true,
            ..CheckArgs::default()
        };
        let mut out = Vec::new();
        let err = run(args, &dir, &mut out).unwrap_err();
        assert!(matches!(err, ToolError::NoPython { .. }), "{err}");
        assert!(err.to_string().contains("--kind host"), "{err}");
        std::fs::remove_dir_all(dir).unwrap();
    }

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
