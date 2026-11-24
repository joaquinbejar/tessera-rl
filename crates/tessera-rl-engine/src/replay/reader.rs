// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reading replay files: the header, the framing and the trailer are checked
//! at open; event bodies are parsed one at a time (ADR-0072 sections 2, 6
//! and 8).
//!
//! The bytes are untrusted. Nothing is cast, every length is checked against
//! the bytes left before anything is allocated, and the input plus a decoded
//! event never passes `max_decoded_bytes`.

use tessera_rl_core::StateHash;
use tessera_rl_core::encoding::checksum;
use tessera_rl_num::identity::Seed;

use crate::persist::{DecodeLimits, FileHeader, PersistError, Reader, u64_of};

use super::ReplayError;
use super::format::{
    Body, END_BODY_LEN, EventTag, FLAG_FROM_RESET_ONLY, FLAG_OUTPUTS_HASH, FORMAT_VERSION,
    FRAME_LEN, IDS_ALL, IDS_LIST, MAX_CONFIG_BYTES, REPLAY_MAGIC, StateWriteOp,
};

/// The instances an event touches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ids {
    /// Every instance, in id order.
    All,
    /// These ids, in this order.
    List(Vec<u32>),
}

/// The `Reseed` of a recorded reset, owned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordedReseed {
    /// `Reseed::Keep`.
    Keep,
    /// `Reseed::Base`.
    Base(Seed),
    /// `Reseed::List`.
    List(Vec<Seed>),
    /// `Reseed::Each`.
    Each(Vec<Option<Seed>>),
}

impl RecordedReseed {
    /// The pool argument.
    #[must_use]
    pub fn as_reseed(&self) -> crate::pool::Reseed<'_> {
        use crate::pool::Reseed;
        match self {
            Self::Keep => Reseed::Keep,
            Self::Base(seed) => Reseed::Base(*seed),
            Self::List(seeds) => Reseed::List(seeds),
            Self::Each(seeds) => Reseed::Each(seeds),
        }
    }
}

/// One event of a replay file, decoded.
#[derive(Debug, Clone, PartialEq)]
pub enum Event<'a> {
    /// A `reset` (ids) or `reset_all` (`Ids::All`) call.
    Reset {
        /// The reset instances.
        ids: Ids,
        /// The reseed argument.
        reseed: RecordedReseed,
    },
    /// A `step` call.
    Step {
        /// The stepped instances.
        ids: Ids,
        /// The discrete actions as submitted.
        discrete: Vec<i32>,
        /// The continuous actions as submitted, as bits.
        continuous_bits: Vec<u32>,
    },
    /// A state write; a keyframe of `ids` follows.
    StateWrite {
        /// The pool call.
        op: StateWriteOp,
        /// The written instances.
        ids: Ids,
        /// The pool's branch counter after the call.
        branch_counter: u64,
    },
    /// Canonical records of `ids`.
    Keyframe {
        /// The pool's branch counter.
        branch_counter: u64,
        /// The instances.
        ids: Ids,
        /// One canonical record (ADR-0029) per id, `record_len` bytes each;
        /// untrusted.
        records: &'a [u8],
    },
    /// `StateHash`es of `ids`.
    Checkpoint {
        /// The instances.
        ids: Ids,
        /// One hash per id.
        hashes: Vec<StateHash>,
    },
    /// The preceding call poisoned the pool.
    Poisoned,
}

/// The replay extension of the header, after the ADR-0011 fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayHeader<'a> {
    /// Instances in the recorded pool.
    pub num_envs: u32,
    /// Bytes of one canonical record.
    pub record_len: u32,
    /// The canonical config bytes.
    pub config: &'a [u8],
    /// Agents per instance.
    pub agents: u32,
    /// Discrete action heads per agent.
    pub discrete_heads: u32,
    /// Continuous action heads per agent.
    pub continuous_heads: u32,
}

/// An opened replay file: header, framing and trailer checked.
pub struct Replay<'a> {
    pub(crate) file: FileHeader<'a>,
    pub(crate) header: ReplayHeader<'a>,
    events: &'a [u8],
    event_count: u64,
    input_len: u64,
    limits: DecodeLimits,
}

impl core::fmt::Debug for Replay<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Replay")
            .field("header", &self.header)
            .field("event_count", &self.event_count)
            .finish_non_exhaustive()
    }
}

impl<'a> Replay<'a> {
    /// Opens a replay file: checks the magic, the version, the header
    /// checksum and fields, the limits, every event frame and tag, and the
    /// trailer's count and checksum. Event bodies are parsed later, by
    /// [`Replay::events`].
    ///
    /// # Errors
    ///
    /// Returns [`ReplayError::BadMagic`] for another file kind (a snapshot
    /// file included), [`ReplayError::UnsupportedVersion`] for another
    /// version, [`ReplayError::UnknownTag`] or [`ReplayError::ReservedTag`]
    /// for a tag this library does not replay, and another typed error for
    /// anything but a sound file within `limits`.
    pub fn open(bytes: &'a [u8], limits: DecodeLimits) -> Result<Self, ReplayError> {
        let input_len = u64_of(bytes.len());
        if input_len > limits.max_decoded_bytes {
            return Err(ReplayError::TooLarge {
                bytes: input_len,
                limit: limits.max_decoded_bytes,
            });
        }
        let mut r = Reader(bytes);
        if r.array::<8>()? != REPLAY_MAGIC {
            return Err(ReplayError::BadMagic);
        }
        let found = r.u32()?;
        if found != FORMAT_VERSION {
            return Err(ReplayError::UnsupportedVersion { found });
        }
        let mut r = Reader(bytes);
        let file = FileHeader::read(&mut r)?;
        let config_len = usize::try_from(r.u32()?).map_err(|_| PersistError::Truncated)?;
        if config_len > MAX_CONFIG_BYTES {
            return Err(PersistError::FieldTooLong {
                field: "config",
                len: config_len,
                cap: MAX_CONFIG_BYTES,
            }
            .into());
        }
        let config = r.take(config_len)?;
        let agents = r.u32()?;
        let discrete_heads = r.u32()?;
        let continuous_heads = r.u32()?;
        let [flags, host_protocol, notifies, refuses] = r.array::<4>()?;
        let header_len = bytes
            .len()
            .checked_sub(r.0.len())
            .ok_or(PersistError::Truncated)?;
        let stored = r.u64()?;
        let header_bytes = bytes.get(..header_len).ok_or(PersistError::Truncated)?;
        if checksum(header_bytes) != stored {
            return Err(ReplayError::HeaderChecksum);
        }
        file.check_fields()?;
        check_replay_fields(flags, host_protocol, notifies, refuses)?;
        if file.count == 0 {
            return Err(PersistError::InvalidField {
                field: "instance_count",
            }
            .into());
        }
        if file.count > limits.max_instances {
            return Err(PersistError::TooManyInstances {
                count: file.count,
                limit: limits.max_instances,
            }
            .into());
        }
        let header = ReplayHeader {
            num_envs: file.count,
            record_len: file.record_len,
            config,
            agents,
            discrete_heads,
            continuous_heads,
        };
        let (events, event_count) = scan(r.0)?;
        Ok(Self {
            file,
            header,
            events,
            event_count,
            input_len,
            limits,
        })
    }

    /// The replay extension of the header.
    #[must_use]
    pub const fn header(&self) -> &ReplayHeader<'a> {
        &self.header
    }

    /// Events before the trailer.
    #[must_use]
    pub const fn event_count(&self) -> u64 {
        self.event_count
    }

    /// The events in file order, each parsed when reached.
    #[must_use]
    pub const fn events(&self) -> Events<'a> {
        Events {
            rest: self.events,
            index: 0,
            num_envs: self.header.num_envs,
            record_len: self.header.record_len,
            input_len: self.input_len,
            limits: self.limits,
        }
    }
}

fn check_replay_fields(
    flags: u8,
    host_protocol: u8,
    notifies: u8,
    refuses: u8,
) -> Result<(), ReplayError> {
    let invalid = |field: &'static str| -> Result<(), ReplayError> {
        Err(PersistError::InvalidField { field }.into())
    };
    if flags & !(FLAG_FROM_RESET_ONLY | FLAG_OUTPUTS_HASH) != 0 {
        return invalid("flags");
    }
    if notifies > 1 {
        return invalid("notifies_rejection");
    }
    if refuses > 1 {
        return invalid("refuses_agents");
    }
    if flags & FLAG_FROM_RESET_ONLY != 0 {
        return Err(ReplayError::Unsupported {
            feature: "from_reset_only replay files",
        });
    }
    if flags & FLAG_OUTPUTS_HASH != 0 {
        return Err(ReplayError::Unsupported {
            feature: "outputs hashes",
        });
    }
    if host_protocol != 0 || notifies != 0 || refuses != 0 {
        return Err(ReplayError::Unsupported {
            feature: "host pool replays",
        });
    }
    Ok(())
}

/// Checks every frame, tag and the trailer; returns the event bytes before
/// the end event and their count. Allocates nothing.
fn scan(bytes: &[u8]) -> Result<(&[u8], u64), ReplayError> {
    let mut rest = bytes;
    let mut count: u64 = 0;
    loop {
        let (frame, tail) = rest
            .split_at_checked(FRAME_LEN)
            .ok_or(ReplayError::MissingEnd)?;
        let [tag, l0, l1, l2, l3] =
            <[u8; FRAME_LEN]>::try_from(frame).map_err(|_| ReplayError::MissingEnd)?;
        let len = u32::from_le_bytes([l0, l1, l2, l3]);
        let tag = EventTag::from_u8(tag).ok_or(ReplayError::UnknownTag { tag })?;
        if tag.is_reserved() {
            return Err(ReplayError::ReservedTag { tag: tag as u8 });
        }
        if count == 0 && tag != EventTag::Keyframe {
            return Err(ReplayError::Malformed {
                event_index: 0,
                field: "first event (a keyframe of every instance)",
            });
        }
        let body_len = usize::try_from(len).map_err(|_| PersistError::Truncated)?;
        let (body, after) = tail
            .split_at_checked(body_len)
            .ok_or(PersistError::Truncated)?;
        if tag == EventTag::End {
            if len != END_BODY_LEN {
                return Err(ReplayError::Malformed {
                    event_index: count,
                    field: "end event length",
                });
            }
            if !after.is_empty() {
                return Err(ReplayError::TrailingBytes);
            }
            let mut b = Body::new(body, count);
            let recorded = b.u64("event count")?;
            let sum = b.u64("events checksum")?;
            let events_len = bytes
                .len()
                .checked_sub(rest.len())
                .ok_or(PersistError::Truncated)?;
            let events = bytes.get(..events_len).ok_or(PersistError::Truncated)?;
            if checksum(events) != sum {
                return Err(ReplayError::EventChecksum);
            }
            if recorded != count {
                return Err(ReplayError::EventCount {
                    recorded,
                    found: count,
                });
            }
            return Ok((events, count));
        }
        count = count.checked_add(1).ok_or(PersistError::Truncated)?;
        rest = after;
    }
}

/// The events of a [`Replay`], parsed one at a time.
#[derive(Debug, Clone)]
pub struct Events<'a> {
    rest: &'a [u8],
    index: u64,
    num_envs: u32,
    record_len: u32,
    input_len: u64,
    limits: DecodeLimits,
}

impl<'a> Iterator for Events<'a> {
    type Item = Result<(u64, Event<'a>), ReplayError>;

    fn next(&mut self) -> Option<Self::Item> {
        let (frame, tail) = self.rest.split_at_checked(FRAME_LEN)?;
        let [tag, l0, l1, l2, l3] = <[u8; FRAME_LEN]>::try_from(frame).ok()?;
        let len = usize::try_from(u32::from_le_bytes([l0, l1, l2, l3])).ok()?;
        let (body, after) = tail.split_at_checked(len)?;
        self.rest = after;
        let index = self.index;
        self.index = self.index.saturating_add(1);
        let event = self.parse(tag, body, index);
        if event.is_err() {
            // One error ends the iteration.
            self.rest = &[];
        }
        Some(event.map(|e| (index, e)))
    }
}

impl<'a> Events<'a> {
    fn parse(&self, tag: u8, body: &'a [u8], index: u64) -> Result<Event<'a>, ReplayError> {
        // The bytes held: the input plus, at most, a decoded copy of the body
        // widened twice (seeds with flags, records as values).
        let held = u64_of(body.len())
            .checked_mul(2)
            .and_then(|b| b.checked_add(self.input_len))
            .unwrap_or(u64::MAX);
        if held > self.limits.max_decoded_bytes {
            return Err(ReplayError::TooLarge {
                bytes: held,
                limit: self.limits.max_decoded_bytes,
            });
        }
        let mut b = Body::new(body, index);
        let tag = EventTag::from_u8(tag).ok_or(ReplayError::UnknownTag { tag })?;
        let event = match tag {
            EventTag::Reset => {
                let ids = self.ids(&mut b)?;
                let reseed = reseed(&mut b)?;
                Event::Reset { ids, reseed }
            }
            EventTag::Step => {
                let ids = self.ids(&mut b)?;
                let n = b.count(4, "discrete actions")?;
                let mut discrete = alloc(n)?;
                for _ in 0..n {
                    discrete.push(b.u32("discrete actions")?.cast_signed());
                }
                let n = b.count(4, "continuous actions")?;
                let mut continuous_bits = alloc(n)?;
                for _ in 0..n {
                    continuous_bits.push(b.u32("continuous actions")?);
                }
                Event::Step {
                    ids,
                    discrete,
                    continuous_bits,
                }
            }
            EventTag::StateWrite => {
                let op = StateWriteOp::from_u8(b.u8("state write op")?)
                    .ok_or_else(|| b.malformed("state write op"))?;
                let ids = self.ids(&mut b)?;
                let branch_counter = b.u64("branch counter")?;
                Event::StateWrite {
                    op,
                    ids,
                    branch_counter,
                }
            }
            EventTag::Keyframe => {
                let branch_counter = b.u64("branch counter")?;
                let ids = self.ids(&mut b)?;
                let n = self.len_of(&ids)?;
                let len = usize::try_from(self.record_len).map_err(|_| b.malformed("records"))?;
                let bytes = n.checked_mul(len).ok_or_else(|| b.malformed("records"))?;
                let records = b.take(bytes, "records")?;
                Event::Keyframe {
                    branch_counter,
                    ids,
                    records,
                }
            }
            EventTag::Checkpoint => {
                let ids = self.ids(&mut b)?;
                let n = self.len_of(&ids)?;
                if n.checked_mul(8) != Some(b.remaining()) {
                    return Err(b.malformed("checkpoint hashes"));
                }
                let mut hashes = alloc(n)?;
                for _ in 0..n {
                    hashes.push(StateHash::from_raw(b.u64("checkpoint hashes")?));
                }
                Event::Checkpoint { ids, hashes }
            }
            EventTag::Poisoned => Event::Poisoned,
            EventTag::End | EventTag::Rejected | EventTag::Adaptive | EventTag::Recv => {
                return Err(ReplayError::ReservedTag { tag: tag as u8 });
            }
        };
        b.finish()?;
        Ok(event)
    }

    fn ids(&self, b: &mut Body<'_>) -> Result<Ids, ReplayError> {
        match b.u8("ids")? {
            IDS_ALL => Ok(Ids::All),
            IDS_LIST => {
                let n = b.count(4, "ids")?;
                if u64_of(n) > u64::from(self.num_envs) {
                    return Err(b.malformed("ids"));
                }
                let mut ids = alloc(n)?;
                for _ in 0..n {
                    let id = b.u32("ids")?;
                    if id >= self.num_envs {
                        return Err(b.malformed("ids"));
                    }
                    ids.push(id);
                }
                Ok(Ids::List(ids))
            }
            _ => Err(b.malformed("ids")),
        }
    }

    fn len_of(&self, ids: &Ids) -> Result<usize, ReplayError> {
        match ids {
            Ids::All => usize::try_from(self.num_envs).map_err(|_| ReplayError::Allocation),
            Ids::List(ids) => Ok(ids.len()),
        }
    }
}

fn reseed(b: &mut Body<'_>) -> Result<RecordedReseed, ReplayError> {
    Ok(match b.u8("reseed")? {
        0 => RecordedReseed::Keep,
        1 => RecordedReseed::Base(Seed::new(b.u64("reseed")?)),
        2 => {
            let n = b.count(8, "seeds")?;
            let mut seeds = alloc(n)?;
            for _ in 0..n {
                seeds.push(Seed::new(b.u64("seeds")?));
            }
            RecordedReseed::List(seeds)
        }
        3 => {
            let n = b.count(9, "seeds")?;
            let mut seeds = alloc(n)?;
            for _ in 0..n {
                let some = b.u8("seeds")?;
                let seed = b.u64("seeds")?;
                seeds.push(match (some, seed) {
                    (0, 0) => None,
                    (1, seed) => Some(Seed::new(seed)),
                    _ => return Err(b.malformed("seeds")),
                });
            }
            RecordedReseed::Each(seeds)
        }
        _ => return Err(b.malformed("reseed")),
    })
}

fn alloc<T>(n: usize) -> Result<Vec<T>, ReplayError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| ReplayError::Allocation)?;
    Ok(v)
}
