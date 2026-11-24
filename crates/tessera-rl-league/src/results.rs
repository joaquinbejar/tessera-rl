// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-instance result rings (ADR-0088 item 4.1).

use crate::{LeagueError, Outcome, PolicyId};

/// One stored result: the opponent's id and the outcome, 8 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
struct Entry {
    id: u32,
    outcome: u8,
    reserved: [u8; 3],
}

/// Fixed-capacity rings of `(PolicyId, Outcome)`, one per instance.
///
/// The storage is allocated once at construction: recording a result never
/// allocates. A full ring drops its oldest entry and counts the drop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Results {
    capacity: usize,
    entries: Vec<Entry>,
    /// Per instance: the index of the oldest entry and the entry count.
    heads: Vec<(u32, u32)>,
    dropped: u64,
}

impl Results {
    /// Rings for `instances` instances of `capacity` entries each.
    ///
    /// # Errors
    ///
    /// [`LeagueError::InvalidConfig`] for a zero or oversized capacity, and
    /// [`LeagueError::Allocation`] when the storage cannot be allocated.
    pub fn new(instances: usize, capacity: usize) -> Result<Self, LeagueError> {
        if capacity == 0 || u32::try_from(capacity).is_err() {
            return Err(LeagueError::InvalidConfig(
                "result capacity must be 1 to u32::MAX",
            ));
        }
        let total = instances
            .checked_mul(capacity)
            .ok_or(LeagueError::Allocation)?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(total)
            .map_err(|_| LeagueError::Allocation)?;
        entries.resize(total, Entry::default());
        let mut heads = Vec::new();
        heads
            .try_reserve_exact(instances)
            .map_err(|_| LeagueError::Allocation)?;
        heads.resize(instances, (0, 0));
        Ok(Self {
            capacity,
            entries,
            heads,
            dropped: 0,
        })
    }

    /// The number of instances.
    #[must_use]
    pub fn instances(&self) -> usize {
        self.heads.len()
    }

    /// The capacity of each ring.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Results dropped by full rings since the last [`Results::clear`].
    #[must_use]
    pub const fn dropped(&self) -> u64 {
        self.dropped
    }

    /// The stored results of `instance`, oldest first.
    pub fn of(&self, instance: usize) -> impl Iterator<Item = (PolicyId, Outcome)> + '_ {
        let (head, len) = self.heads.get(instance).copied().unwrap_or((0, 0));
        let base = instance.saturating_mul(self.capacity);
        (0..len as usize).filter_map(move |k| {
            let at = base + (head as usize + k) % self.capacity;
            let e = self.entries.get(at)?;
            Some((PolicyId(e.id), Outcome::from_u8(e.outcome)?))
        })
    }

    /// Records a result for `instance`. Never allocates.
    ///
    /// # Errors
    ///
    /// [`LeagueError::UnknownInstance`] for an instance outside the rings.
    pub fn record(
        &mut self,
        instance: usize,
        id: PolicyId,
        outcome: Outcome,
    ) -> Result<(), LeagueError> {
        let instances = self.heads.len();
        let capacity = self.capacity;
        let (head, len) = self
            .heads
            .get_mut(instance)
            .ok_or(LeagueError::UnknownInstance {
                instance,
                instances,
            })?;
        // head < capacity and len <= capacity, both below u32::MAX
        // (checked at construction), so these sums cannot overflow.
        let slot = (*head as usize + *len as usize) % capacity;
        if *len as usize == capacity {
            // Full: overwrite the oldest, which moves the head.
            *head = u32::try_from((*head as usize + 1) % capacity).unwrap_or(0);
            self.dropped = self.dropped.saturating_add(1);
        } else {
            *len += 1;
        }
        // instance * capacity + slot < instances * capacity, which
        // construction checked to fit.
        let at = instance * capacity + slot;
        if let Some(e) = self.entries.get_mut(at) {
            *e = Entry {
                id: id.0,
                outcome: outcome as u8,
                reserved: [0; 3],
            };
        }
        Ok(())
    }

    /// Empties every ring and resets the drop count.
    pub fn clear(&mut self) {
        self.heads.fill((0, 0));
        self.dropped = 0;
    }
}
