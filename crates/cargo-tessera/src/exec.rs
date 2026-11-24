// SPDX-License-Identifier: MIT OR Apache-2.0

//! Running the programs `check` and `bench` drive (#351).

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::ToolError;

/// The cargo that runs this subcommand (`$CARGO`), or `cargo`.
#[must_use]
pub fn cargo() -> PathBuf {
    std::env::var_os("CARGO").map_or_else(|| PathBuf::from("cargo"), PathBuf::from)
}

/// `name` on the PATH (with `.exe` on Windows), if any.
#[must_use]
pub fn find_program(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    let names: Vec<String> = if cfg!(windows) {
        vec![format!("{name}.exe"), name.to_owned()]
    } else {
        vec![name.to_owned()]
    };
    std::env::split_paths(&path).find_map(|dir| {
        names
            .iter()
            .map(|n| dir.join(n))
            .find(|candidate| candidate.is_file())
    })
}

/// The command line of `cmd`, for messages.
fn describe(cmd: &Command) -> String {
    let mut out = cmd.get_program().to_string_lossy().into_owned();
    for arg in cmd.get_args() {
        out.push(' ');
        out.push_str(&arg.to_string_lossy());
    }
    out
}

/// Runs `cmd` with inherited output; an error unless it exits with success.
///
/// # Errors
///
/// [`ToolError::Spawn`] or [`ToolError::Failed`].
pub fn run(cmd: &mut Command) -> Result<(), ToolError> {
    let status = cmd.status().map_err(|source| ToolError::Spawn {
        program: cmd.get_program().to_string_lossy().into_owned(),
        source,
    })?;
    if status.success() {
        Ok(())
    } else {
        Err(ToolError::Failed {
            command: describe(cmd),
            status: status.to_string(),
        })
    }
}

/// Checks that `dir` holds a scaffolded crate with `files`.
///
/// # Errors
///
/// [`ToolError::NotACrate`] naming the first missing file.
pub fn ensure_crate(dir: &Path, files: &[&'static str]) -> Result<(), ToolError> {
    for &file in files {
        if !dir.join(file).is_file() {
            return Err(ToolError::NotACrate {
                dir: dir.to_path_buf(),
                missing: file,
            });
        }
    }
    Ok(())
}

/// A command running `program` in `dir`.
#[must_use]
pub fn command(program: impl AsRef<OsStr>, dir: &Path) -> Command {
    let mut cmd = Command::new(program);
    cmd.current_dir(dir);
    cmd
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use super::{describe, find_program};

    #[test]
    fn test_find_program_unknown_name_is_none() {
        assert_eq!(find_program("cargo-tessera-no-such-program-351"), None);
    }

    #[test]
    fn test_describe_joins_program_and_args() {
        let mut cmd = Command::new("cargo");
        cmd.args(["test", "--test", "check"]);
        assert_eq!(describe(&cmd), "cargo test --test check");
    }
}
