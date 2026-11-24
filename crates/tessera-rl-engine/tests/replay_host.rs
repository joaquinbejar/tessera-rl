// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host pool replay files (ADR-0072, ADR-0069 item 8, ADR-0070 item 9):
//! `from_reset_only` round trips for a plain, a notifying and a refusing
//! host, `Rejected` events only for a notifying host, and the divergence
//! kinds for a tampered file and for a host whose `validate`, `refuse` or
//! `step` changed under the same id, version and header.

#![cfg(all(feature = "host", feature = "replay"))]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::encoding::checksum;
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    ActionReject, AgentSlot, EnvConfig, HostEnvironment, HostError, HostOut, HostSeed, Refusals,
    RefuseCode, SpaceError,
};
use tessera_rl_engine::replay::{
    DivergenceKind, Event, EventTag, HostRecorder, RefusalRow, Replay, ReplayError,
};
use tessera_rl_engine::{
    ActionInput, DecodeLimits, EngineError, HostPool, PersistError, PoolOptions, Reseed,
    SyncOptions,
};
use tessera_rl_num::identity::Seed;

const N: usize = 4;
const SLOTS: usize = 2;
/// The agent is refused by a refusing host.
const REFUSE: i32 = 2;
/// `validate` rejects the whole call.
const REJECT: i32 = 3;
const REJECT_CODE: u32 = 5;
const REFUSE_CODE: u32 = 10;

/// Switches that change a rule while the header stays the same: a host
/// config is not canonical and not in the file.
#[derive(Debug, Clone, Default)]
struct Cfg {
    reject_shift: u32,
    refuse_shift: u32,
    step_bonus: i32,
}

impl EnvConfig for Cfg {}

#[derive(Debug)]
struct Desk<const NOTIFY: bool, const REFUSES: bool> {
    cfg: Cfg,
    score: [i32; SLOTS],
    notices: u32,
    steps: u32,
}

fn slot(s: usize) -> AgentSlot {
    AgentSlot::new(u16::try_from(s).unwrap(), SLOTS).unwrap()
}

impl<const NOTIFY: bool, const REFUSES: bool> HostEnvironment for Desk<NOTIFY, REFUSES> {
    const ENV_ID: &'static str = "test/Desk-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = SLOTS;
    const VALIDATES: bool = true;
    const NOTIFIES_REJECTION: bool = NOTIFY;
    const REFUSES_AGENTS: bool = REFUSES;
    type Config = Cfg;

    fn new(cfg: &Cfg) -> Result<Self, HostError> {
        Ok(Self {
            cfg: cfg.clone(),
            score: [0; SLOTS],
            notices: 0,
            steps: 0,
        })
    }

    fn observation_space(_: &Cfg) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![3],
            dtype: DType::I32,
            bounds: BoxBounds::Int {
                low: vec![-1_000_000],
                high: vec![1_000_000],
            },
        })
    }

    fn action_space(_: &Cfg) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 4 })
    }

    fn reset(&mut self, seed: HostSeed, _: &mut HostOut<'_>) {
        self.score = [i32::try_from(seed.seed() % 7).unwrap(), 0];
        self.notices = 0;
        self.steps = 0;
    }

    fn validate(&self, actions: &Actions<'_, ()>) -> Result<(), ActionReject> {
        if (0..SLOTS).any(|s| actions.discrete(slot(s), 0) == Some(REJECT)) {
            return Err(ActionReject {
                code: REJECT_CODE + self.cfg.reject_shift,
                reason: "the desk rejected the call",
            });
        }
        Ok(())
    }

    fn on_rejected(&mut self, reject: &ActionReject) {
        self.notices += reject.code;
    }

    fn refuse(&self, actions: &Actions<'_, ()>, out: &mut Refusals<'_>) {
        for s in 0..SLOTS {
            if actions.discrete(slot(s), 0) == Some(REFUSE) {
                let code = REFUSE_CODE + u32::try_from(s).unwrap() + self.cfg.refuse_shift;
                out.refuse(slot(s), RefuseCode::new(code).unwrap());
            }
        }
    }

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut HostOut<'_>) {
        self.steps += 1;
        for s in 0..SLOTS {
            let reward = if REFUSES && actions.is_refused(slot(s)) {
                -1
            } else {
                actions.discrete(slot(s), 0).unwrap_or(0) + self.cfg.step_bonus
            };
            self.score[s] += reward;
            out.set_reward(slot(s), f32::from(i16::try_from(reward).unwrap()));
        }
    }

    fn observe(&self, agent: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>) {
        let values = [
            self.score[agent.index()],
            i32::try_from(self.notices).unwrap(),
            i32::try_from(self.steps).unwrap(),
        ];
        if let Some(leaf) = obs.leaf(0) {
            for (bytes, v) in leaf.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                bytes.copy_from_slice(&v.to_le_bytes());
            }
        }
    }
}

type Pool<const NOTIFY: bool, const REFUSES: bool> = HostPool<Desk<NOTIFY, REFUSES>, SLOTS>;

fn pool<const NOTIFY: bool, const REFUSES: bool>(
    cfg: &Cfg,
    workers: usize,
) -> Pool<NOTIFY, REFUSES> {
    let n = NonZeroU32::new(u32::try_from(N).unwrap()).unwrap();
    let options = PoolOptions::new(n, Seed::new(3));
    let p = Pool::<NOTIFY, REFUSES>::new(cfg, options).unwrap();
    if workers == 0 {
        p
    } else {
        p.with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
            .unwrap()
    }
}

/// `fill` everywhere except `special` (instance, slot, action).
fn joint(fill: i32, special: &[(usize, usize, i32)]) -> Vec<i32> {
    let mut a = vec![fill; N * SLOTS];
    for &(i, s, v) in special {
        a[i * SLOTS + s] = v;
    }
    a
}

type Rec<'p, const NOTIFY: bool, const REFUSES: bool> =
    HostRecorder<'p, Desk<NOTIFY, REFUSES>, SLOTS, Vec<u8>>;

fn step<const NOTIFY: bool, const REFUSES: bool>(
    rec: &mut Rec<'_, NOTIFY, REFUSES>,
    actions: &[i32],
) -> Result<(), ReplayError> {
    rec.step(ActionInput {
        discrete: actions,
        continuous: &[],
    })
    .map(|_| ())
}

/// The recorded script: steps, a subset reset, a rejected call (accepted
/// for no host: `validate` always rejects action 3), refusals in instance 0
/// only, with one refused and one unrefused agent.
fn script<const NOTIFY: bool, const REFUSES: bool>(rec: &mut Rec<'_, NOTIFY, REFUSES>) {
    step(rec, &joint(1, &[])).unwrap();
    step(rec, &joint(0, &[(1, 1, 1)])).unwrap();
    rec.reset(&[2, 0], Reseed::Each(&[Some(Seed::new(9)), None]))
        .unwrap();
    let rejected = step(rec, &joint(1, &[(2, 0, REJECT), (3, 1, REJECT)]));
    assert!(matches!(
        rejected,
        Err(ReplayError::Engine(EngineError::ActionRejected {
            instance: 2,
            ..
        }))
    ));
    step(rec, &joint(1, &[(0, 0, REFUSE)])).unwrap();
    rec.reset_all(Reseed::Keep).unwrap();
    step(rec, &joint(0, &[(3, 1, REFUSE)])).unwrap();
}

fn record<const NOTIFY: bool, const REFUSES: bool>(workers: usize) -> Vec<u8> {
    let mut p = pool::<NOTIFY, REFUSES>(&Cfg::default(), workers);
    let mut rec = HostRecorder::new(&mut p, Reseed::Base(Seed::new(1)), Vec::new()).unwrap();
    script(&mut rec);
    rec.finish().unwrap()
}

const LIMITS: DecodeLimits = DecodeLimits {
    max_instances: 1 << 24,
    max_decoded_bytes: 1 << 30,
};

fn events(bytes: &[u8]) -> Vec<Event<'_>> {
    Replay::open(bytes, LIMITS)
        .unwrap()
        .events()
        .map(|e| e.unwrap().1)
        .collect()
}

fn replay<const NOTIFY: bool, const REFUSES: bool>(
    bytes: &[u8],
    cfg: &Cfg,
) -> Result<tessera_rl_engine::ReplaySummary, ReplayError> {
    Replay::open(bytes, LIMITS)
        .unwrap()
        .run_host(&mut pool::<NOTIFY, REFUSES>(cfg, 0))
}

/// Byte offset of the first event of a host file (empty config).
fn header_len(b: &[u8]) -> usize {
    let u16_at = |at: usize| usize::from(u16::from_le_bytes([b[at], b[at + 1]]));
    let mut at = 48;
    for _ in 0..2 {
        at += 2 + u16_at(at);
    }
    at += 2;
    at += 2 + u16_at(at);
    let config = u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize;
    at + 4 + config + 12 + 4 + 8
}

/// The (offset, tag) of every frame, the end event included.
fn frames(b: &[u8]) -> Vec<(usize, u8)> {
    let mut at = header_len(b);
    let mut out = Vec::new();
    while at < b.len() {
        let len = u32::from_le_bytes(b[at + 1..at + 5].try_into().unwrap()) as usize;
        out.push((at, b[at]));
        at += 5 + len;
    }
    out
}

/// Rewrites the end event's count and checksum.
fn reseal(b: &mut [u8]) {
    let fr = frames(b);
    let (end_at, _) = *fr.last().unwrap();
    let start = header_len(b);
    let count = u64::try_from(fr.len() - 1).unwrap();
    let sum = checksum(&b[start..end_at]);
    b[end_at + 5..end_at + 13].copy_from_slice(&count.to_le_bytes());
    b[end_at + 13..end_at + 21].copy_from_slice(&sum.to_le_bytes());
}

/// `b` with one byte of the body of the `nth` event with `tag` flipped at
/// `offset`, resealed.
fn tampered(b: &[u8], tag: EventTag, nth: usize, offset: usize) -> Vec<u8> {
    let at = frames(b)
        .into_iter()
        .filter(|f| f.1 == tag as u8)
        .nth(nth)
        .unwrap()
        .0;
    let mut out = b.to_vec();
    out[at + 5 + offset] ^= 1;
    reseal(&mut out);
    out
}

fn diverged(
    r: Result<tessera_rl_engine::ReplaySummary, ReplayError>,
) -> (DivergenceKind, Option<u32>) {
    match r {
        Err(ReplayError::Diverged { kind, instance, .. }) => (kind, instance),
        other => {
            assert!(other.is_err(), "replay did not diverge");
            (DivergenceKind::CallFailed, None)
        }
    }
}

// Step body: ids (1), count (4), 8 discrete actions (32), count (4).
const STEP_HASH: usize = 41;
const STEP_ROWS: usize = STEP_HASH + 8;
// Rejected body: count (4), 8 actions (32), count (4), instance (4).
const REJECTED_CODE: usize = 44;
/// The reported instance, just before the code.
const REJECTED_INSTANCE: usize = REJECTED_CODE - 4;

#[test]
fn test_replay_host_plain_round_trip_has_no_divergence() {
    let bytes = record::<false, false>(0);
    let summary = replay::<false, false>(&bytes, &Cfg::default()).unwrap();
    // The first reset, four steps and two resets; the rejected call is not
    // logged.
    assert_eq!(summary.calls, 7);
    assert!(!summary.poisoned);
    let ev = events(&bytes);
    assert!(matches!(
        ev[0],
        Event::Reset {
            outputs_hash: Some(_),
            ..
        }
    ));
    assert!(!ev.iter().any(|e| matches!(e, Event::Rejected { .. })));
    // Sync recording and Sync replay give the same file and no divergence.
    let sync = record::<false, false>(2);
    assert_eq!(bytes, sync);
    Replay::open(&sync, LIMITS)
        .unwrap()
        .run_host(&mut pool::<false, false>(&Cfg::default(), 2))
        .unwrap();
}

#[test]
fn test_replay_host_notifying_round_trip_logs_and_reproduces_the_rejection() {
    let bytes = record::<true, false>(0);
    let ev = events(&bytes);
    let rejected: Vec<_> = ev
        .iter()
        .filter_map(|e| match e {
            Event::Rejected {
                instance,
                code,
                discrete,
                ..
            } => Some((*instance, *code, discrete.len())),
            _ => None,
        })
        .collect();
    assert_eq!(rejected, [(2, REJECT_CODE, N * SLOTS)]);
    let summary = replay::<true, false>(&bytes, &Cfg::default()).unwrap();
    assert_eq!(summary.calls, 8);
    assert_eq!(bytes, record::<true, false>(2));
}

#[test]
fn test_replay_host_refusing_round_trip_reproduces_refusal_rows() {
    let bytes = record::<false, true>(0);
    let rows: Vec<Vec<RefusalRow>> = events(&bytes)
        .into_iter()
        .filter_map(|e| match e {
            Event::Step { refusals, .. } => Some(refusals),
            _ => None,
        })
        .collect();
    let row = |instance, slot: u32| RefusalRow {
        instance,
        slot,
        code: REFUSE_CODE + slot,
    };
    // Instance 0 refuses slot 0 and steps slot 1; the others do not refuse.
    assert_eq!(rows[2], [row(0, 0)]);
    assert_eq!(rows[3], [row(3, 1)]);
    assert!(rows[0].is_empty() && rows[1].is_empty());
    replay::<false, true>(&bytes, &Cfg::default()).unwrap();
    assert_eq!(bytes, record::<false, true>(2));
}

#[test]
fn test_replay_host_non_notifying_rejection_is_not_logged() {
    let mut p = pool::<false, false>(&Cfg::default(), 0);
    let mut rec = HostRecorder::new(&mut p, Reseed::Keep, Vec::new()).unwrap();
    assert!(step(&mut rec, &joint(REJECT, &[])).is_err());
    let bytes = rec.finish().unwrap();
    assert_eq!(Replay::open(&bytes, LIMITS).unwrap().event_count(), 1);
    // A Rejected event in a non-notifying file is refused at open.
    let (end, _) = *frames(&bytes).last().unwrap();
    let mut forged = bytes[..end].to_vec();
    forged.extend_from_slice(&[EventTag::Rejected as u8, 0, 0, 0, 0]);
    forged.extend_from_slice(&bytes[end..]);
    reseal(&mut forged);
    assert!(matches!(
        Replay::open(&forged, LIMITS),
        Err(ReplayError::ReservedTag { tag: 3 })
    ));
}

#[test]
fn test_replay_host_tampered_rejection_code_is_a_rejection_mismatch() {
    let bytes = record::<true, false>(0);
    let bad = tampered(&bytes, EventTag::Rejected, 0, REJECTED_CODE);
    assert_eq!(
        diverged(replay::<true, false>(&bad, &Cfg::default())),
        (DivergenceKind::RejectionMismatch, Some(2))
    );
}

#[test]
fn test_replay_host_tampered_rejection_instance_is_a_rejection_mismatch() {
    let bytes = record::<true, false>(0);
    // The reported instance 2 becomes 3; the code is unchanged.
    let bad = tampered(&bytes, EventTag::Rejected, 0, REJECTED_INSTANCE);
    assert_eq!(
        diverged(replay::<true, false>(&bad, &Cfg::default())),
        (DivergenceKind::RejectionMismatch, Some(3))
    );
}

#[test]
fn test_replay_host_tampered_refusal_row_is_a_refusal_mismatch() {
    let bytes = record::<false, true>(0);
    // The code of the only row of the third step: 10 becomes 11.
    let bad = tampered(&bytes, EventTag::Step, 2, STEP_ROWS + 4 + 8);
    assert_eq!(
        diverged(replay::<false, true>(&bad, &Cfg::default())),
        (DivergenceKind::RefusalMismatch, Some(0))
    );
}

#[test]
fn test_replay_host_tampered_outputs_hash_is_an_outputs_mismatch() {
    let bytes = record::<false, false>(0);
    let bad = tampered(&bytes, EventTag::Step, 1, STEP_HASH);
    assert_eq!(
        diverged(replay::<false, false>(&bad, &Cfg::default())).0,
        DivergenceKind::OutputsMismatch
    );
    // The first reset's hash, after the ids marker and the Base reseed.
    let bad = tampered(&bytes, EventTag::Reset, 0, 1 + 1 + 8);
    assert_eq!(
        diverged(replay::<false, false>(&bad, &Cfg::default())).0,
        DivergenceKind::OutputsMismatch
    );
}

#[test]
fn test_replay_host_changed_validate_refuse_or_step_diverges() {
    let notifying = record::<true, false>(0);
    let changed_validate = Cfg {
        reject_shift: 1,
        ..Cfg::default()
    };
    assert_eq!(
        diverged(replay::<true, false>(&notifying, &changed_validate)),
        (DivergenceKind::RejectionMismatch, Some(2))
    );
    let refusing = record::<false, true>(0);
    let changed_refuse = Cfg {
        refuse_shift: 1,
        ..Cfg::default()
    };
    assert_eq!(
        diverged(replay::<false, true>(&refusing, &changed_refuse)),
        (DivergenceKind::RefusalMismatch, Some(0))
    );
    let plain = record::<false, false>(0);
    let changed_step = Cfg {
        step_bonus: 1,
        ..Cfg::default()
    };
    assert_eq!(
        diverged(replay::<false, false>(&plain, &changed_step)).0,
        DivergenceKind::OutputsMismatch
    );
}

#[test]
fn test_replay_host_file_into_another_host_kind_is_incompatible() {
    let plain = record::<false, false>(0);
    assert!(matches!(
        Replay::open(&plain, LIMITS)
            .unwrap()
            .run_host(&mut pool::<true, false>(&Cfg::default(), 0)),
        Err(ReplayError::Persist(PersistError::Incompatible {
            field: "notifies_rejection"
        }))
    ));
    assert!(matches!(
        Replay::open(&plain, LIMITS)
            .unwrap()
            .run_host(&mut pool::<false, true>(&Cfg::default(), 0)),
        Err(ReplayError::Persist(PersistError::Incompatible {
            field: "refuses_agents"
        }))
    ));
}

#[test]
fn test_replay_host_header_flags_and_protocol() {
    for (bytes, protocol, notifies, refuses) in [
        (record::<false, false>(0), 1, false, false),
        (record::<true, false>(0), 1, true, false),
        (record::<false, true>(0), 2, false, true),
    ] {
        let replay = Replay::open(&bytes, LIMITS).unwrap();
        let host = replay.header().host.unwrap();
        assert_eq!(
            (host.host_protocol, host.notifies, host.refuses),
            (protocol, notifies, refuses)
        );
        assert_eq!(replay.header().config, []);
        assert_eq!(replay.header().record_len, 0);
    }
}

/// `b` with the header's tier byte set to `tier` (offset 13: magic, version,
/// payload encoding) and the header checksum rewritten.
fn with_tier(b: &[u8], tier: u8) -> Vec<u8> {
    let mut out = b.to_vec();
    out[13] = tier;
    let end = header_len(&out) - 8;
    let sum = checksum(&out[..end]);
    out[end..end + 8].copy_from_slice(&sum.to_le_bytes());
    out
}

#[test]
fn test_replay_host_header_tier_is_two_and_only_two() {
    let bytes = record::<false, false>(0);
    assert_eq!(bytes[13], 2, "host files carry the host tier (ADR-0073)");
    for tier in [0u8, 1, 3] {
        assert!(
            matches!(
                Replay::open(&with_tier(&bytes, tier), LIMITS),
                Err(ReplayError::Persist(PersistError::InvalidField {
                    field: "tier"
                }))
            ),
            "tier {tier} in a host file"
        );
    }
}
