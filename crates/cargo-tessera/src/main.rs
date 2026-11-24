// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo tessera` (#351, SPEC 10.2, ADR-0021): scaffolds a host
//! environment crate (`new`), runs its checks (`check`) and its bench
//! (`bench`). Cargo runs it as `cargo-tessera tessera <args>`.

mod args;
mod bench;
mod check;
mod error;
mod exec;
mod names;
mod new;
mod python;
mod render;

use std::io::Write;
use std::process::ExitCode;

use args::Command;
use error::ToolError;

fn main() -> ExitCode {
    let argv = std::env::args_os().map(|a| a.to_string_lossy().into_owned());
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    match run(argv, &mut out) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let _ = writeln!(std::io::stderr(), "error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Parses `argv` and runs the command in the current directory.
fn run(argv: impl Iterator<Item = String>, out: &mut impl Write) -> Result<(), ToolError> {
    let command = args::parse(argv)?;
    let cwd = || {
        std::env::current_dir().map_err(|source| ToolError::Io {
            action: "reading",
            path: ".".into(),
            source,
        })
    };
    match command {
        Command::Help(topic) => {
            let _ = out.write_all(args::help(topic).as_bytes());
            Ok(())
        }
        Command::Version => {
            let _ = writeln!(out, "cargo-tessera {}", new::TOOL_VERSION);
            Ok(())
        }
        Command::New(a) => new::run(&a, &cwd()?, out),
        Command::Check(a) => check::run(a, &cwd()?, out),
        Command::Bench(a) => bench::run(a, &cwd()?, out),
    }
}
