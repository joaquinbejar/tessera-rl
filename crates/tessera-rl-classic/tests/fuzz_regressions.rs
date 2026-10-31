// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stable replay of every input in `fuzz/regressions/record/` through the same
//! harness as the `record` fuzz target, so fixed findings stay fixed
//! without nightly or cargo-fuzz.

#[path = "../../../fuzz/harness/record.rs"]
mod harness;

#[test]
fn test_fuzz_regressions_replay_without_panicking() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fuzz/regressions/record");
    let mut replayed = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "bin") {
            harness::run(&std::fs::read(&path).unwrap());
            replayed += 1;
        }
    }
    assert!(replayed >= 1, "no regression inputs in {}", dir.display());
}
