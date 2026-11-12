// SPDX-License-Identifier: MIT OR Apache-2.0

//! Observation leaves are zero on entry to `observe` (ADR-0058).
//!
//! The output buffers are read-only from outside the engine, so the sentinel
//! goes in through the engine itself: the fixture writes `SENTINEL` into
//! every byte of both leaves while its state is loud, and writes nothing
//! while it is quiet. Each path below starts from rows that hold the
//! sentinel (checked first) and must leave the rows the engine observes
//! all zero wherever the fixture writes nothing: reset, step, auto-reset in
//! both terminal modes (terminal rows included), and the re-observation
//! after restore, import, fork and load. Slot 1 can die and spawn again in
//! the same row position, so a stale row from its earlier life would show.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, InvariantViolation, NoAdaptive, ObsLeaf, Observation,
    Persist, RecordHeader, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodeLimits, DecodedRecord, EnvPool, NoMasks, PoolOptions, Reseed,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

/// Written into every byte of every leaf while the state is loud.
const SENTINEL: u8 = 0xAB;

/// Agent 0's action bits; agent 1's action is ignored.
const QUIET: i32 = 0;
const LOUD: i32 = 1;
const KILL: i32 = 2;
const SPAWN: i32 = 4;
const END: i32 = 8;

const AGENTS: usize = 2;
const LEAF_BYTES: [usize; 2] = [3, 8];

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Blink {
    loud: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct BlinkConfig {
    #[config(default = 0)]
    unused: u32,
}

#[derive(Observation)]
struct BlinkObs {
    #[space(box(low = 0, high = 255, dtype = u8, shape = [3]))]
    bytes: ObsLeaf<u8>,
    #[space(box(low = -1, high = 1, shape = [2]))]
    words: ObsLeaf<i32>,
}

/// Two agents; slot 1 can die and spawn. Writes the sentinel or nothing.
struct BlinkEnv;

const SECOND: AgentSlot = match AgentSlot::new(1, 2) {
    Some(s) => s,
    None => AgentSlot::FIRST,
};

impl Environment for BlinkEnv {
    type State = Blink;
    type Config = BlinkConfig;
    type Obs = BlinkObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = AGENTS;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &BlinkConfig) -> Result<Space, SpaceError> {
        BlinkObs::space(cfg)
    }

    fn action_space(_: &BlinkConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 16 })
    }

    /// Every episode starts quiet.
    fn reset(_: &BlinkConfig, _: &mut StepRng, state: &mut Blink) {
        *state = Blink::default();
    }

    fn step(
        _: &BlinkConfig,
        state: &mut Blink,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let bits = actions.discrete(AgentSlot::FIRST, 0).unwrap_or(0);
        state.loud = u32::from(bits & LOUD != 0);
        if bits & KILL != 0 {
            out.agent_died(SECOND);
        }
        if bits & SPAWN != 0 {
            out.agent_spawned(SECOND);
        }
        if bits & END != 0 {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    /// Fills every leaf with the sentinel while loud; otherwise writes no
    /// byte at all.
    fn observe(
        _: &BlinkConfig,
        state: &Blink,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, BlinkObs>,
    ) {
        if state.loud == 0 {
            return;
        }
        for leaf in 0..LEAF_BYTES.len() {
            if let Some(bytes) = out.leaf(leaf) {
                bytes.fill(SENTINEL);
            }
        }
    }
}

impl Validate for BlinkEnv {
    fn validate(_: &BlinkConfig, state: &Blink) -> Result<(), InvariantViolation> {
        if state.loud > 1 {
            return Err(InvariantViolation {
                code: "loud is 0 or 1",
            });
        }
        Ok(())
    }
}

impl Persist for BlinkEnv {
    const ENV_ID: &'static str = "tessera/Blink-v1";
    const ENV_VERSION: u32 = 1;
}

type Blinks = EnvPool<BlinkEnv, NoMasks>;

const LIMITS: DecodeLimits = DecodeLimits {
    max_instances: 16,
    max_decoded_bytes: 1 << 20,
};

fn pool(num_envs: u32, auto_reset: AutoReset) -> Blinks {
    let options = PoolOptions {
        auto_reset,
        ..PoolOptions::new(NonZeroU32::new(num_envs).unwrap(), Seed::new(5))
    };
    let mut p = Blinks::new(BlinkConfig::default(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

/// One step; `bits[i]` is agent 0's action in instance `i`.
fn step(p: &mut Blinks, bits: &[i32]) {
    let actions: Vec<i32> = bits.iter().flat_map(|&b| [b, QUIET]).collect();
    p.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap();
}

/// What a row must hold: the sentinel or zero, byte for byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Row {
    Zero,
    Sentinel,
}

fn rows(bytes: &[u8], i: usize, slot: usize, leaf: usize) -> &[u8] {
    let n = LEAF_BYTES[leaf];
    let at = (i * AGENTS + slot) * n;
    &bytes[at..at + n]
}

fn check(found: &[u8], want: Row, what: &str) {
    let byte = match want {
        Row::Zero => 0,
        Row::Sentinel => SENTINEL,
    };
    assert!(
        found.iter().all(|&b| b == byte),
        "{what}: expected {want:?}, found {found:?}"
    );
}

/// Asserts instance `i`'s observation rows, slot by slot, on every leaf.
fn observation(p: &Blinks, i: usize, want: [Row; AGENTS], what: &str) {
    let out = p.outputs().unwrap();
    for leaf in 0..LEAF_BYTES.len() {
        let bytes = out.observation(leaf).unwrap();
        for (slot, &w) in want.iter().enumerate() {
            check(
                rows(bytes, i, slot, leaf),
                w,
                &format!("{what}: instance {i} slot {slot} leaf {leaf}"),
            );
        }
    }
}

/// Asserts instance `i`'s terminal observation rows (`SameStep` only).
fn terminal(p: &Blinks, i: usize, want: [Row; AGENTS], what: &str) {
    let out = p.outputs().unwrap();
    for leaf in 0..LEAF_BYTES.len() {
        let bytes = out.terminal_observation(leaf).unwrap();
        for (slot, &w) in want.iter().enumerate() {
            check(
                rows(bytes, i, slot, leaf),
                w,
                &format!("{what}: terminal, instance {i} slot {slot} leaf {leaf}"),
            );
        }
    }
}

fn alive(p: &Blinks, i: usize) -> [bool; AGENTS] {
    let a = p.outputs().unwrap().alive();
    [a[i * AGENTS].get(), a[i * AGENTS + 1].get()]
}

const ZERO: [Row; AGENTS] = [Row::Zero, Row::Zero];
const FULL: [Row; AGENTS] = [Row::Sentinel, Row::Sentinel];

#[test]
fn test_observe_zero_step_and_reset_over_sentinel_rows_are_zero() {
    let mut p = pool(1, AutoReset::Disabled);
    step(&mut p, &[LOUD]);
    observation(&p, 0, FULL, "loud step");
    step(&mut p, &[QUIET]);
    observation(&p, 0, ZERO, "step");

    step(&mut p, &[LOUD]);
    observation(&p, 0, FULL, "loud step");
    p.reset_all(Reseed::Keep).unwrap();
    observation(&p, 0, ZERO, "reset_all");

    step(&mut p, &[LOUD]);
    observation(&p, 0, FULL, "loud step");
    p.reset(&[0], Reseed::Keep).unwrap();
    observation(&p, 0, ZERO, "reset");
}

#[test]
fn test_observe_zero_dead_then_respawned_slot_has_no_stale_row() {
    let mut p = pool(1, AutoReset::Disabled);
    step(&mut p, &[LOUD]);
    observation(&p, 0, FULL, "both alive and loud");

    step(&mut p, &[LOUD | KILL]);
    assert_eq!(alive(&p, 0), [true, false]);
    observation(&p, 0, [Row::Sentinel, Row::Zero], "slot 1 dead");

    step(&mut p, &[LOUD]);
    assert_eq!(alive(&p, 0), [true, false]);
    observation(&p, 0, [Row::Sentinel, Row::Zero], "slot 1 still dead");

    step(&mut p, &[QUIET | SPAWN]);
    assert_eq!(alive(&p, 0), [true, true]);
    observation(&p, 0, ZERO, "slot 1 alive again, quiet");

    // Die while the row holds the sentinel from this life.
    step(&mut p, &[LOUD]);
    observation(&p, 0, FULL, "both alive and loud");
    step(&mut p, &[LOUD | KILL]);
    observation(&p, 0, [Row::Sentinel, Row::Zero], "slot 1 dead again");
}

#[test]
fn test_observe_zero_next_step_auto_reset_over_sentinel_rows_is_zero() {
    let mut p = pool(1, AutoReset::NextStep);
    step(&mut p, &[LOUD | END]);
    observation(&p, 0, FULL, "final observation, loud");
    // The instance resets instead of stepping; its initial state is quiet.
    step(&mut p, &[LOUD]);
    observation(&p, 0, ZERO, "auto-reset");
}

#[test]
fn test_observe_zero_same_step_auto_reset_terminal_rows_are_zero() {
    let mut p = pool(1, AutoReset::SameStep);
    // A loud final observation goes to the terminal rows; the reset's
    // observation, quiet, to the observation rows.
    step(&mut p, &[LOUD | END]);
    terminal(&p, 0, FULL, "loud final observation");
    observation(&p, 0, ZERO, "same-step reset after a loud end");

    // Terminal rows hold the sentinel; a quiet end right after must give
    // zero terminal rows.
    step(&mut p, &[QUIET | END]);
    terminal(&p, 0, ZERO, "quiet final observation");
    observation(&p, 0, ZERO, "same-step reset");

    // Observation rows hold the sentinel; a quiet end must not carry it
    // into the terminal rows.
    step(&mut p, &[LOUD]);
    observation(&p, 0, FULL, "loud step");
    step(&mut p, &[QUIET | END]);
    terminal(&p, 0, ZERO, "quiet final observation over sentinel rows");
    observation(&p, 0, ZERO, "same-step reset");
}

#[test]
fn test_observe_zero_restore_and_import_over_sentinel_rows_are_zero() {
    let mut p = pool(1, AutoReset::Disabled);
    let quiet = p.snapshot(0).unwrap();

    step(&mut p, &[LOUD]);
    observation(&p, 0, FULL, "loud step");
    p.restore(0, &quiet).unwrap();
    observation(&p, 0, ZERO, "restore");

    step(&mut p, &[LOUD]);
    observation(&p, 0, FULL, "loud step");
    p.restore_all(&[quiet]).unwrap();
    observation(&p, 0, ZERO, "restore_all");

    step(&mut p, &[LOUD]);
    observation(&p, 0, FULL, "loud step");
    p.import(&[0], &[DecodedRecord::from(quiet)]).unwrap();
    observation(&p, 0, ZERO, "import");
}

#[test]
fn test_observe_zero_fork_over_sentinel_rows_is_zero() {
    let mut p = pool(2, AutoReset::Disabled);
    step(&mut p, &[QUIET, LOUD]);
    observation(&p, 0, ZERO, "quiet source");
    observation(&p, 1, FULL, "loud destination");
    p.fork(0, &[1]).unwrap();
    observation(&p, 1, ZERO, "fork");
}

#[test]
fn test_observe_zero_load_over_sentinel_rows_is_zero() {
    let mut p = pool(2, AutoReset::Disabled);
    let file = p.encode(&[0, 1]).unwrap();
    step(&mut p, &[LOUD, LOUD]);
    observation(&p, 0, FULL, "loud step");
    observation(&p, 1, FULL, "loud step");
    let records = p.decode(&file, LIMITS).unwrap();
    p.import_all(&records).unwrap();
    observation(&p, 0, ZERO, "load");
    observation(&p, 1, ZERO, "load");
}
