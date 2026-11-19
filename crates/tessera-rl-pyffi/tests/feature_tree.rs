// SPDX-License-Identifier: MIT OR Apache-2.0

//! The first-party wheel never enables the host features (#272 step 4,
//! ADR-0062): `tessera-rl-py`'s feature tree, for every feature set its
//! wheels are built with, shows neither `tessera-rl-pyffi`'s `host` nor
//! `tessera-rl-engine`'s, and its manifest line on `tessera-rl-pyffi`
//! names no `host`.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

use std::path::Path;
use std::process::Command;

/// The inverted feature tree of `invert` as `package` (with `features`)
/// enables it: every feature of `invert` and what enables it.
fn tree(package: &str, invert: &str, features: &str) -> String {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let mut args = vec![
        "tree",
        "-q",
        "--offline",
        "-p",
        package,
        "-e",
        "features",
        "-i",
        invert,
    ];
    if !features.is_empty() {
        args.extend(["--features", features]);
    }
    let out = Command::new(cargo).args(&args).output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn test_feature_tree_first_party_wheel_has_no_host_features() {
    for features in ["", "test-hooks", "onnx"] {
        let pyffi = tree("tessera-rl-py", "tessera-rl-pyffi", features);
        assert!(
            pyffi.contains("tessera-rl-pyffi feature \"native-module\""),
            "{pyffi}"
        );
        assert!(
            !pyffi.contains("tessera-rl-pyffi feature \"host\""),
            "{features}: {pyffi}"
        );
        let engine = tree("tessera-rl-py", "tessera-rl-engine", features);
        assert!(
            engine.contains("tessera-rl-engine feature \"default\""),
            "{engine}"
        );
        assert!(
            !engine.contains("tessera-rl-engine feature \"host\""),
            "{features}: {engine}"
        );
    }
}

#[test]
fn test_feature_tree_check_sees_host_where_it_is_enabled() {
    // The control: the same query finds the engine's host feature in a
    // crate that does enable it.
    let engine = tree("tessera-rl-test-host-glue", "tessera-rl-engine", "");
    assert!(
        engine.contains("tessera-rl-engine feature \"host\""),
        "{engine}"
    );
}

#[test]
fn test_feature_tree_manifest_pins_pyffi_without_host() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tessera-rl-py/Cargo.toml");
    let text = std::fs::read_to_string(manifest).unwrap();
    let line = text
        .lines()
        .find(|l| l.starts_with("tessera-rl-pyffi"))
        .unwrap();
    assert_eq!(
        line,
        "tessera-rl-pyffi = { workspace = true, features = [\"native-module\"] }"
    );
}
