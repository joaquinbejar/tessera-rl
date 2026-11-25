// SPDX-License-Identifier: MIT OR Apache-2.0

//! The default worker count: one per physical core the process may use
//! (SPEC 6.1, ADR-0005). Read once, at construction time only.
//!
//! The count is `min(physical cores, available_parallelism())`, so a cgroup
//! CPU quota (which `available_parallelism` honours on Linux) caps it.
//! Physical cores come from sysfs topology within the affinity mask on
//! Linux, `hw.physicalcpu` on macOS and `RelationProcessorCore` records on
//! Windows (all processor groups, ignoring the affinity mask; only the
//! `available_parallelism` cap applies there). All three count efficiency cores of hybrid CPUs as cores. When
//! detection fails, the count is `available_parallelism()` alone.

use core::num::{NonZeroU32, NonZeroUsize};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::OnceLock;

/// Physical cores this process may use, capped by `available_parallelism`;
/// at least 1. Computed once per process.
#[must_use]
pub fn physical_cores() -> NonZeroUsize {
    static COUNT: OnceLock<NonZeroUsize> = OnceLock::new();
    *COUNT.get_or_init(|| {
        let physical = detect();
        let parallelism = std::thread::available_parallelism().ok();
        if physical.is_none() {
            tracing::debug!(
                ?parallelism,
                "physical core count unavailable; using available parallelism"
            );
        }
        combine(physical, parallelism)
    })
}

/// The default worker count of a pool of `num_envs` instances in chunks of
/// `chunk_size`: one per physical core ([`physical_cores`]), at most one per
/// chunk (so never more than `num_envs`, since chunks are non-empty).
#[must_use]
pub fn default_workers(num_envs: NonZeroU32, chunk_size: NonZeroU32) -> NonZeroUsize {
    clamp_workers(physical_cores(), num_envs, chunk_size)
}

/// `cores`, capped by the chunk count of `num_envs` in `chunk_size` chunks.
#[must_use]
pub(crate) fn clamp_workers(
    cores: NonZeroUsize,
    num_envs: NonZeroU32,
    chunk_size: NonZeroU32,
) -> NonZeroUsize {
    let chunks = num_envs.div_ceil(chunk_size);
    match usize::try_from(chunks.get()) {
        Ok(chunks) => cores.min(NonZeroUsize::new(chunks).unwrap_or(NonZeroUsize::MIN)),
        Err(_) => cores,
    }
}

/// `min(physical, parallelism)`; either alone when the other is missing;
/// 1 when both are.
#[must_use]
pub(crate) fn combine(
    physical: Option<NonZeroUsize>,
    parallelism: Option<NonZeroUsize>,
) -> NonZeroUsize {
    match (physical, parallelism) {
        (Some(p), Some(q)) => p.min(q),
        (Some(c), None) | (None, Some(c)) => c,
        (None, None) => NonZeroUsize::MIN,
    }
}

#[cfg(all(target_os = "linux", not(any(loom, miri))))]
fn detect() -> Option<NonZeroUsize> {
    let cpus = crate::sys::available_cores()?;
    sysfs_cores(Path::new("/sys/devices/system/cpu"), &cpus)
}

#[cfg(all(any(target_os = "macos", windows), not(any(loom, miri))))]
fn detect() -> Option<NonZeroUsize> {
    crate::sys::native::physical_cores()
}

#[cfg(any(
    loom,
    miri,
    not(any(target_os = "linux", target_os = "macos", windows))
))]
fn detect() -> Option<NonZeroUsize> {
    None
}

/// Distinct `(physical_package_id, core_id)` pairs of the logical `cpus`
/// under the sysfs cpu directory `root`; `None` if any of them cannot be
/// read or parsed, or `cpus` is empty.
#[cfg_attr(
    not(any(all(target_os = "linux", not(miri)), test)),
    expect(
        dead_code,
        reason = "sysfs exists on Linux only, and Miri's detect() returns None"
    )
)]
#[must_use]
pub(crate) fn sysfs_cores(root: &Path, cpus: &[usize]) -> Option<NonZeroUsize> {
    let mut cores = BTreeSet::new();
    for cpu in cpus {
        let topology = root.join(format!("cpu{cpu}")).join("topology");
        let package = read_id(&topology.join("physical_package_id"))?;
        let core = read_id(&topology.join("core_id"))?;
        cores.insert((package, core));
    }
    NonZeroUsize::new(cores.len())
}

/// Records of `relation` in a Windows
/// `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` list: each record starts with
/// `Relationship: i32` and `Size: u32` (native endian). A truncated or
/// malformed record (size below 8 or past the end) gives `None`.
#[cfg_attr(
    not(any(all(windows, not(miri)), test)),
    expect(
        dead_code,
        reason = "the record list is a Windows format, and Miri's detect() returns None"
    )
)]
#[must_use]
pub(crate) fn count_records(mut rest: &[u8], relation: i32) -> Option<NonZeroUsize> {
    let mut count = 0_usize;
    while !rest.is_empty() {
        let kind = i32::from_ne_bytes(rest.get(0..4)?.try_into().ok()?);
        let size = u32::from_ne_bytes(rest.get(4..8)?.try_into().ok()?);
        let size = usize::try_from(size).ok()?;
        if size < 8 {
            return None;
        }
        if kind == relation {
            count = count.checked_add(1)?;
        }
        rest = rest.get(size..)?;
    }
    NonZeroUsize::new(count)
}

/// One integer sysfs attribute (`physical_package_id` may be -1).
fn read_id(path: &Path) -> Option<i64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests;
