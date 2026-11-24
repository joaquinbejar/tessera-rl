// SPDX-License-Identifier: MIT OR Apache-2.0

use core::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use super::{clamp_workers, combine, count_records, physical_cores, sysfs_cores};
use core::num::NonZeroU32;

fn n(v: usize) -> NonZeroUsize {
    NonZeroUsize::new(v).unwrap()
}

/// A scratch sysfs tree with `(cpu, package, core)` rows.
struct Tree(PathBuf);

impl Tree {
    fn new(name: &str, rows: &[(usize, &str, &str)]) -> Self {
        let root =
            std::env::temp_dir().join(format!("tessera-cores-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (cpu, package, core) in rows {
            let topology = root.join(format!("cpu{cpu}")).join("topology");
            std::fs::create_dir_all(&topology).unwrap();
            std::fs::write(topology.join("physical_package_id"), format!("{package}\n")).unwrap();
            std::fs::write(topology.join("core_id"), format!("{core}\n")).unwrap();
        }
        Self(root)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn test_physical_cores_sysfs_smt_siblings_counts_distinct_cores() {
    // Two cores, two SMT threads each.
    let tree = Tree::new(
        "smt",
        &[(0, "0", "0"), (1, "0", "1"), (2, "0", "0"), (3, "0", "1")],
    );
    assert_eq!(sysfs_cores(tree.path(), &[0, 1, 2, 3]), Some(n(2)));
}

#[test]
fn test_physical_cores_sysfs_two_packages_same_core_id_counts_both() {
    let tree = Tree::new("pkg", &[(0, "0", "0"), (1, "1", "0")]);
    assert_eq!(sysfs_cores(tree.path(), &[0, 1]), Some(n(2)));
}

#[test]
fn test_physical_cores_affinity_mask_excludes_unmasked_cores() {
    let tree = Tree::new(
        "mask",
        &[(0, "0", "0"), (1, "0", "1"), (2, "0", "2"), (3, "0", "3")],
    );
    assert_eq!(sysfs_cores(tree.path(), &[1, 3]), Some(n(2)));
}

#[test]
fn test_physical_cores_unreadable_sysfs_returns_none() {
    let tree = Tree::new("missing", &[(0, "0", "0")]);
    assert_eq!(sysfs_cores(tree.path(), &[0, 1]), None);
}

#[test]
fn test_physical_cores_unparsable_id_returns_none() {
    let tree = Tree::new("garbage", &[(0, "x", "0")]);
    assert_eq!(sysfs_cores(tree.path(), &[0]), None);
}

#[test]
fn test_physical_cores_negative_package_id_counts() {
    let tree = Tree::new("neg", &[(0, "-1", "0"), (1, "-1", "1")]);
    assert_eq!(sysfs_cores(tree.path(), &[0, 1]), Some(n(2)));
}

#[test]
fn test_physical_cores_empty_mask_returns_none() {
    let tree = Tree::new("empty", &[(0, "0", "0")]);
    assert_eq!(sysfs_cores(tree.path(), &[]), None);
}

#[test]
fn test_physical_cores_quota_below_cores_uses_available_parallelism() {
    assert_eq!(combine(Some(n(32)), Some(n(2))), n(2));
}

#[test]
fn test_physical_cores_cores_below_parallelism_uses_cores() {
    assert_eq!(combine(Some(n(8)), Some(n(16))), n(8));
}

#[test]
fn test_physical_cores_unreadable_sysfs_falls_back_to_available_parallelism() {
    assert_eq!(combine(None, Some(n(6))), n(6));
}

#[test]
fn test_physical_cores_no_parallelism_uses_cores() {
    assert_eq!(combine(Some(n(4)), None), n(4));
}

#[test]
fn test_physical_cores_nothing_known_is_one() {
    assert_eq!(combine(None, None), NonZeroUsize::MIN);
}

#[test]
fn test_physical_cores_host_at_most_available_parallelism() {
    let count = physical_cores();
    if let Ok(parallelism) = std::thread::available_parallelism() {
        assert!(count <= parallelism, "{count} > {parallelism}");
    }
    assert_eq!(physical_cores(), count, "cached value changed");
}

fn record(kind: i32, size: u32) -> Vec<u8> {
    let mut r = kind.to_ne_bytes().to_vec();
    r.extend_from_slice(&size.to_ne_bytes());
    r.resize(usize::try_from(size).unwrap().max(8), 0);
    r
}

#[test]
fn test_count_records_mixed_relations_counts_matching() {
    let mut list = record(0, 48);
    list.extend(record(3, 80));
    list.extend(record(0, 48));
    assert_eq!(count_records(&list, 0), Some(n(2)));
}

#[test]
fn test_count_records_size_below_header_returns_none() {
    let mut list = record(0, 48);
    list.extend(record(0, 4).get(..8).unwrap());
    assert_eq!(count_records(&list, 0), None);
}

#[test]
fn test_count_records_size_past_end_returns_none() {
    let mut list = record(0, 48);
    list.truncate(40);
    assert_eq!(count_records(&list, 0), None);
}

#[test]
fn test_count_records_truncated_header_returns_none() {
    let mut list = record(0, 48);
    list.extend([0, 0, 0]);
    assert_eq!(count_records(&list, 0), None);
}

#[test]
fn test_count_records_none_matching_returns_none() {
    assert_eq!(count_records(&record(3, 48), 0), None);
    assert_eq!(count_records(&[], 0), None);
}

#[test]
fn test_sync_options_default_uses_physical_cores() {
    let options = crate::SyncOptions::default();
    assert_eq!(options, crate::SyncOptions::new(physical_cores()));
}

#[test]
fn test_pipelined_options_default_uses_physical_cores() {
    let options = crate::PipelinedOptions::default();
    assert_eq!(options.sync, crate::SyncOptions::new(physical_cores()));
}

#[test]
fn test_count_records_size_below_header_with_valid_tail_returns_none() {
    // A 4-byte record would land on a well-formed 8-byte one: still refused.
    let mut list = 0_i32.to_ne_bytes().to_vec();
    list.extend(4_u32.to_ne_bytes());
    list.extend(8_u32.to_ne_bytes());
    assert_eq!(list.len(), 12);
    assert_eq!(count_records(&list, 0), None);
}

fn u(v: u32) -> NonZeroU32 {
    NonZeroU32::new(v).unwrap()
}

#[test]
fn test_default_workers_clamped_to_chunks_and_num_envs() {
    // 1000 instances in chunks of 64: 16 chunks.
    assert_eq!(clamp_workers(n(32), u(1000), u(64)), n(16));
    assert_eq!(clamp_workers(n(8), u(1000), u(64)), n(8));
    // A partial last chunk counts: 65 instances are 2 chunks.
    assert_eq!(clamp_workers(n(8), u(65), u(64)), n(2));
    assert_eq!(clamp_workers(n(8), u(64), u(64)), n(1));
    // Chunks of 1: at most num_envs workers.
    assert_eq!(clamp_workers(n(8), u(3), u(1)), n(3));
}

#[test]
fn test_default_workers_at_most_physical_cores() {
    let w = super::default_workers(u(1 << 20), u(1));
    assert_eq!(w, physical_cores());
}
