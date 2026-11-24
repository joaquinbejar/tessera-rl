// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo tessera check --python` (#351): installs the crate's wheel into a
//! virtual environment and runs its adapter tests. The tool downloads
//! nothing itself: `maturin` (or `uv` with `--uv`) and Python come from the
//! PATH.

use std::fs;
use std::path::{Path, PathBuf};

use crate::args::CheckArgs;
use crate::error::ToolError;
use crate::exec;

/// The virtual environment `check` creates when none is active.
const VENV: &str = "target/tessera-check-venv";

/// Installs the wheel and runs `pytest tests/test_adapters.py`.
///
/// The environment is the active one (`VIRTUAL_ENV`), or a new one at
/// `target/tessera-check-venv` made with `python -m venv
/// --system-site-packages` (Python is `$PYTHON`, else `python3`, else
/// `python`), so the first-party `tessera-rl`, `pytest`, `gymnasium` and
/// `pettingzoo` come from the system site-packages. The wheel is built and
/// installed with `maturin develop`, or with `uv pip install --no-deps
/// --reinstall` for `--uv`; `--locked` and `--offline` reach its cargo
/// build either way.
///
/// # Errors
///
/// [`ToolError::MissingProgram`], [`ToolError::Io`], or a failing command.
pub fn run(args: CheckArgs, dir: &Path) -> Result<(), ToolError> {
    exec::ensure_crate(dir, &["pyproject.toml", "tests/test_adapters.py"])?;
    let installer = if args.uv { "uv" } else { "maturin" };
    let installer_path = exec::find_program(installer).ok_or(ToolError::MissingProgram {
        program: installer.to_owned(),
        hint: if args.uv {
            "install uv (https://docs.astral.sh/uv/)"
        } else {
            "install maturin (pip install maturin, or pipx install maturin), or pass --uv"
        },
    })?;
    let venv = match std::env::var_os("VIRTUAL_ENV") {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => create_venv(dir)?,
    };
    let python = venv_python(&venv);
    let cargo_flags = args.cargo.args();
    let mut install = exec::command(&installer_path, dir);
    if args.uv {
        install
            .args(["pip", "install", "--python"])
            .arg(&python)
            .args(["--no-deps", "--reinstall"]);
        if args.cargo.offline {
            install.arg("--offline");
        }
        install.arg(".");
        // maturin's PEP 517 backend passes these to cargo.
        if !cargo_flags.is_empty() {
            install.env("MATURIN_PEP517_ARGS", cargo_flags.join(" "));
        }
    } else {
        install.arg("develop").args(&cargo_flags);
    }
    install.env("VIRTUAL_ENV", &venv);
    exec::run(&mut install)?;
    let mut pytest = exec::command(&python, dir);
    pytest.args(["-m", "pytest", "-q", "tests/test_adapters.py"]);
    pytest.env("VIRTUAL_ENV", &venv);
    exec::run(&mut pytest)
}

/// The interpreter of the virtual environment `venv`.
fn venv_python(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

/// A fresh virtual environment under `dir/target`.
fn create_venv(dir: &Path) -> Result<PathBuf, ToolError> {
    let base = match std::env::var_os("PYTHON") {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => exec::find_program("python3")
            .or_else(|| exec::find_program("python"))
            .ok_or(ToolError::MissingProgram {
                program: "python3".to_owned(),
                hint: "install Python 3.11 or later, or set PYTHON",
            })?,
    };
    let venv = dir.join(VENV);
    if fs::symlink_metadata(&venv).is_ok() {
        fs::remove_dir_all(&venv).map_err(|source| ToolError::Io {
            action: "removing",
            path: venv.clone(),
            source,
        })?;
    }
    let mut cmd = exec::command(&base, dir);
    cmd.args(["-m", "venv", "--system-site-packages"])
        .arg(&venv);
    exec::run(&mut cmd)?;
    Ok(venv)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::venv_python;

    #[test]
    fn test_venv_python_inside_the_venv() {
        let p = venv_python(Path::new("v"));
        assert!(p.starts_with("v"));
        assert!(p.to_string_lossy().contains("python"));
    }
}
