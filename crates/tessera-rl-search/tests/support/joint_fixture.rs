// SPDX-License-Identifier: MIT OR Apache-2.0

//! The joint fixtures shared with the Python cross-check of
//! `tessera_rl.search.joint_log_prior` (`tests/python/test_search_joint.py`,
//! review r1 of #454): per case the head sizes, the parents, the activity
//! table of each head (one bit per parent value pair, the last parent
//! fastest), and the native canonical and legal joint indices. Used by the
//! search tests (test environment) and the order book's tests.

use core::fmt::Write as _;

use tessera_rl_core::space::JointHeads;

/// One case's text.
pub fn joint_case(
    name: &str,
    nvec: &[u32],
    parents: &[[Option<u16>; 2]],
    active: &dyn Fn(usize, [u32; 2]) -> bool,
    joint: &JointHeads,
    legal: &dyn Fn(u32) -> bool,
) -> String {
    let mut s = String::new();
    let _ = write!(s, "case {name}\nnvec");
    for n in nvec {
        let _ = write!(s, " {n}");
    }
    s.push_str("\nparents");
    for p in parents {
        let list: Vec<String> = p.iter().flatten().map(u16::to_string).collect();
        if list.is_empty() {
            s.push_str(" -");
        } else {
            let _ = write!(s, " {}", list.join(","));
        }
    }
    s.push_str("\nactive");
    for (h, p) in parents.iter().enumerate() {
        let size = |q: Option<u16>| q.map_or(1, |q| nvec.get(usize::from(q)).copied().unwrap_or(1));
        let (s1, s2) = (size(p[0]), size(p[1]));
        let bits: Vec<&str> = (0..s1 * s2)
            .map(|pair| {
                let v1 = if p[0].is_some() { pair / s2 } else { 0 };
                let v2 = if p[1].is_some() { pair % s2 } else { 0 };
                if active(h, [v1, v2]) { "1" } else { "0" }
            })
            .collect();
        let _ = write!(s, " {}", bits.join(","));
    }
    s.push_str("\ncanonical");
    for j in (0..joint.joint()).filter(|&j| joint.is_canonical(j)) {
        let _ = write!(s, " {j}");
    }
    s.push_str("\nlegal");
    for j in (0..joint.joint()).filter(|&j| legal(j)) {
        let _ = write!(s, " {j}");
    }
    s.push('\n');
    s
}

/// Compares `text` with the committed fixture `name`
/// (`tests/python/fixtures/joint/`), writing it first when
/// `TESSERA_BLESS_JOINT` is set.
pub fn check_fixture(name: &str, text: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/python/fixtures/joint")
        .join(name);
    if std::env::var_os("TESSERA_BLESS_JOINT").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, text).unwrap();
    }
    let committed = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        committed, text,
        "{name}: the native joints changed; rerun with TESSERA_BLESS_JOINT=1 and review"
    );
}
