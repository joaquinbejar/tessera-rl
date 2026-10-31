// SPDX-License-Identifier: MIT OR Apache-2.0

//! Link arguments for a Python extension module (for example
//! `-undefined dynamic_lookup` on macOS), so plain `cargo build`, `cargo
//! test` and clippy link the cdylib the way maturin does.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
