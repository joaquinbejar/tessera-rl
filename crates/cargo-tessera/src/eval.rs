// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo tessera eval` (ADR-0088 section 9, #384 head E4): a thin wrapper
//! that runs `python -m tessera_rl.eval` with every argument passed through.
//! The interpreter is `$TESSERA_PYTHON`, else `python3`; its output is
//! inherited, and a failure is the tool's failure.

use std::path::PathBuf;
use std::process::Command;

use crate::error::ToolError;
use crate::exec;

/// The interpreter: `$TESSERA_PYTHON`, else `python3`.
#[must_use]
pub fn python() -> PathBuf {
    std::env::var_os("TESSERA_PYTHON").map_or_else(|| PathBuf::from("python3"), PathBuf::from)
}

/// The command `eval` runs for `args`.
#[must_use]
pub fn command(args: &[String]) -> Command {
    let mut cmd = Command::new(python());
    cmd.arg("-m").arg("tessera_rl.eval").args(args);
    cmd
}

/// Runs `python -m tessera_rl.eval args...`.
///
/// # Errors
///
/// [`ToolError::Spawn`] or [`ToolError::Failed`].
pub fn run(args: &[String]) -> Result<(), ToolError> {
    exec::run(&mut command(args))
}

#[cfg(test)]
mod tests {
    use super::command;

    #[test]
    fn test_eval_passes_every_argument_through() {
        let cmd = command(&[
            "--env".to_owned(),
            "tessera/PaddleDuel-v1".to_owned(),
            "--episodes".to_owned(),
            "3".to_owned(),
        ]);
        let args: Vec<_> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            args,
            [
                "-m",
                "tessera_rl.eval",
                "--env",
                "tessera/PaddleDuel-v1",
                "--episodes",
                "3"
            ]
        );
    }
}
