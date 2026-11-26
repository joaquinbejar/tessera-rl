// SPDX-License-Identifier: MIT OR Apache-2.0

//! Survival-v1's agent-lane step against its reference step (ADR-0100
//! decisions 2 and 5, head 3). The oracle is always the reference
//! (`Survival::step` and the `SurvivalState` helpers it uses), never the
//! kernels at `Level::Scalar`, and every check runs at every level this CPU
//! has (`Resolved::available`, which includes `Scalar`).
//!
//! - Per phase: each lane part against the reference's loop for it, on
//!   reachable states (pool trajectories with deaths and respawns, 16, 21,
//!   64 and 128 agents, so groups with tails) and random heads.
//! - Whole step: a pool of `SurvivalLanes` (the agent-lane step) against a
//!   pool of `Survival` with `AgentLanes::Reference`, step by step: every
//!   instance's state, rewards, flags and observation bytes.
//! - Whole step on stressed states: `lanes::step` against `Survival::step`
//!   directly, from reachable states pushed to low food, water and health
//!   and with dead agents due to respawn, so that starvation, kills and T5
//!   respawns all happen; the test asserts that each one did.
//! - Goldens: the same through Survival's golden scenarios, whose committed
//!   per-step output digests both pools reproduce. The record-hash fields
//!   are not compared: they hash the record's environment id, which the
//!   wrapper does not share; the states are compared directly instead.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

#[path = "support/survival_lanes_env.rs"]
mod survival_lanes_env;

use core::num::{NonZeroU32, NonZeroUsize};

use survival_lanes_env::SurvivalLanes;
use tessera_rl_core::encoding::checksum;
use tessera_rl_core::io::Actions;
use tessera_rl_core::{Bool8, Environment, StepOut};
use tessera_rl_engine::{
    ActionInput, AgentLanes, AutoReset, EnvPool, NoMasks, Outputs, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_simd::Resolved;
use tessera_rl_world::survival::{
    ATTACK, DRINK, FORAGE, MAX_AGENTS, MAX_HEALTH, MAX_STOCK, STARVE, lanes, toward,
};
use tessera_rl_world::{Survival, SurvivalConfig, SurvivalState};

/// splitmix64.
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn below(key: u64, n: u64) -> i32 {
    i32::try_from(mix(key) % n).unwrap()
}

/// Random actions for `instances` instances at step `t`: a move and an act
/// per agent row.
fn random_actions(seed: u64, t: u32, instances: usize) -> Vec<i32> {
    (0..instances * MAX_AGENTS)
        .flat_map(|row| {
            let key = seed ^ (u64::from(t) << 40) ^ (u64::try_from(row).unwrap() << 2);
            [below(key, 5), below(key ^ 1, 7)]
        })
        .collect()
}

fn config(agents: u32, size: u32) -> SurvivalConfig {
    SurvivalConfig {
        agents,
        size,
        steps: 64,
        respawn: 2,
        ..SurvivalConfig::default()
    }
}

/// The configs: groups of 8 with and without a tail.
fn configs() -> Vec<SurvivalConfig> {
    vec![
        config(16, 32),
        config(21, 40),
        config(64, 64),
        config(128, 96),
    ]
}

/// Reachable states: instance 0 of a reference pool before each of `steps`
/// steps of random actions.
fn states(cfg: &SurvivalConfig, seed: u64, steps: u32) -> Vec<SurvivalState> {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(seed));
    let mut p = EnvPool::<Survival, NoMasks>::new(cfg.clone(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let mut out = Vec::new();
    for t in 1..=steps {
        out.push(p.state(0).unwrap());
        p.step(ActionInput {
            discrete: &random_actions(seed, t, 1),
            continuous: &[],
        })
        .unwrap();
    }
    out
}

fn every_state() -> Vec<SurvivalState> {
    let mut all = Vec::new();
    for (i, cfg) in configs().iter().enumerate() {
        for seed in [11, 12] {
            all.extend(states(cfg, seed * 100 + u64::try_from(i).unwrap(), 70));
        }
    }
    all
}

fn head(seed: u64, a: usize, n: u64) -> [i32; MAX_AGENTS] {
    core::array::from_fn(|i| below(seed ^ u64::try_from(i * 31 + a).unwrap(), n))
}

#[test]
fn test_survival_lanes_states_cover_deaths_and_tails() {
    let all = every_state();
    assert!(all.len() > 500);
    assert!(
        all.iter()
            .any(|s| (0..s.agent_count()).any(|a| !s.is_alive(a)))
    );
    assert!(all.iter().any(|s| s.agent_count() % 8 != 0));
}

#[test]
fn test_survival_lanes_move_targets_equal_the_reference() {
    for (k, s) in every_state().iter().enumerate() {
        let dirs = head(u64::try_from(k).unwrap(), 0, 5);
        // The reference: `Survival::step`'s first phase-1 loop.
        let mut want = [(0i16, 0i16); MAX_AGENTS];
        for (a, t) in want.iter_mut().enumerate().take(s.agent_count()) {
            let (x, y) = s.pos(a);
            *t = (x, y);
            if !s.is_alive(a) {
                continue;
            }
            let (tx, ty) = toward(x, y, u32::try_from(dirs[a]).unwrap());
            if (tx, ty) != (x, y) && s.free(tx, ty) {
                *t = (tx, ty);
            }
        }
        for level in Resolved::available() {
            assert_eq!(
                lanes::move_targets(level, s, &dirs),
                want,
                "state {k}, {level:?}"
            );
        }
    }
}

#[test]
fn test_survival_lanes_attack_victims_equal_the_reference() {
    for (k, s) in every_state().iter().enumerate() {
        let acts = head(u64::try_from(k).unwrap() ^ 0x55, 1, 7);
        // The reference: `Survival::step`'s phase-2 lookup.
        let want: [i32; MAX_AGENTS] = core::array::from_fn(|a| {
            let act = u32::try_from(acts[a]).unwrap();
            if a >= s.agent_count() || !s.is_alive(a) || !(3..=6).contains(&act) {
                return -1;
            }
            let (x, y) = s.pos(a);
            let (tx, ty) = toward(x, y, act - 2);
            s.agent_at(tx, ty).map_or(-1, |v| i32::try_from(v).unwrap())
        });
        for level in Resolved::available() {
            assert_eq!(
                lanes::attack_victims(level, s, &acts),
                want,
                "state {k}, {level:?}"
            );
        }
    }
}

#[test]
fn test_survival_lanes_upkeep_equals_the_reference() {
    for (k, s) in every_state().iter().enumerate() {
        let seed = u64::try_from(k).unwrap();
        let acts = head(seed ^ 0x77, 1, 7);
        let damage: [i16; MAX_AGENTS] = core::array::from_fn(|i| {
            i16::try_from(below(seed ^ (u64::try_from(i).unwrap() << 8), 5)).unwrap() * ATTACK
        });
        // The reference: `Survival::step`'s phase-3 agent loop on `moved`.
        let mut want = lanes::Upkeep {
            food: s.food,
            water: s.water,
            health: s.health,
            forage: [false; MAX_AGENTS],
        };
        for a in 0..s.agent_count() {
            if !s.is_alive(a) {
                continue;
            }
            let (x, y) = s.pos(a);
            let act = acts[a];
            let mut food = s.food[a];
            let mut water = s.water[a];
            if act == 1 && s.ready_forest(x, y) {
                food = (food + FORAGE).min(MAX_STOCK);
                want.forage[a] = true;
            }
            if act == 2 && s.by_water(x, y) {
                water = (water + DRINK).min(MAX_STOCK);
            }
            food = (food - 1).max(0);
            water = (water - 1).max(0);
            let mut health = s.health[a];
            if food == 0 || water == 0 {
                health -= STARVE;
            }
            health -= damage[a];
            (want.food[a], want.water[a], want.health[a]) =
                (food, water, health.clamp(0, MAX_HEALTH));
        }
        for level in Resolved::available() {
            assert_eq!(
                lanes::upkeep(level, s, &acts, &damage),
                want,
                "state {k}, {level:?}"
            );
        }
    }
}

#[test]
fn test_survival_lanes_deaths_and_ages_equal_the_reference() {
    for (k, s) in every_state().iter().enumerate() {
        let seed = u64::try_from(k).unwrap();
        let health: [i16; MAX_AGENTS] = core::array::from_fn(|i| {
            i16::try_from(below(seed ^ u64::try_from(i).unwrap(), 12)).unwrap() - 2
        });
        let want: [bool; MAX_AGENTS] = core::array::from_fn(|v| s.is_alive(v) && health[v] <= 0);
        // The post-death state: the dying agents marked dead.
        let mut now = *s;
        for (v, &d) in want.iter().enumerate() {
            if d {
                now.alive[v] = tessera_rl_core::Bool8::FALSE;
            }
        }
        let mut aged = now;
        for a in 0..now.agent_count() {
            if now.is_alive(a) {
                aged.age[a] += 1;
            } else if !s.is_alive(a) {
                aged.dead_for[a] += 1;
            }
        }
        for level in Resolved::available() {
            assert_eq!(
                lanes::deaths(level, s, &health),
                want,
                "state {k}, {level:?}"
            );
            let mut got = now;
            lanes::ages(level, s, &mut got);
            assert_eq!(got, aged, "state {k}, {level:?}");
        }
    }
}

/// The outputs of a step: rewards, every observation leaf and the episode
/// flags, as bytes.
fn output_bytes(out: &Outputs) -> Vec<u8> {
    let mut bytes: Vec<u8> = out
        .rewards()
        .iter()
        .flat_map(|r| r.to_bits().to_le_bytes())
        .collect();
    let mut leaf = 0;
    while let Some(obs) = out.observation(leaf) {
        bytes.extend_from_slice(obs);
        leaf += 1;
    }
    bytes.extend(out.episode_terminated().iter().map(|f| u8::from(f.get())));
    bytes.extend(out.episode_truncated().iter().map(|f| u8::from(f.get())));
    bytes
}

fn with_workers<E: Environment>(
    p: EnvPool<E, NoMasks>,
    workers: Option<usize>,
) -> EnvPool<E, NoMasks>
where
    NoMasks: tessera_rl_engine::MaskProvider<E>,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
{
    match workers {
        None => p,
        Some(w) => p
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap(),
    }
}

/// A reference pool and an agent-lane pool at `level`, stepped together.
fn pools(
    cfg: &SurvivalConfig,
    instances: u32,
    auto_reset: AutoReset,
    workers: Option<usize>,
    level: Resolved,
) -> (EnvPool<Survival, NoMasks>, EnvPool<SurvivalLanes, NoMasks>) {
    let base = PoolOptions {
        auto_reset,
        chunk_size: NonZeroU32::new(2).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(instances).unwrap(), Seed::new(3_820_001))
    };
    let reference = PoolOptions {
        agent_lanes: AgentLanes::Reference,
        ..base.clone()
    };
    let lanes = PoolOptions {
        agent_lanes: AgentLanes::Auto,
        simd_level: Some(level.level()),
        ..base
    };
    (
        with_workers(EnvPool::new(cfg.clone(), reference).unwrap(), workers),
        with_workers(EnvPool::new(cfg.clone(), lanes).unwrap(), workers),
    )
}

#[test]
fn test_survival_lanes_whole_step_equals_the_reference_pool() {
    for cfg in [config(16, 32), config(21, 40), config(128, 96)] {
        for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
            for workers in [None, Some(2)] {
                for level in Resolved::available() {
                    let (mut a, mut b) = pools(&cfg, 4, auto_reset, workers, level);
                    a.reset_all(Reseed::Keep).unwrap();
                    b.reset_all(Reseed::Keep).unwrap();
                    for t in 1..=90 {
                        let acts = random_actions(7, t, 4);
                        let input = || ActionInput {
                            discrete: &acts,
                            continuous: &[],
                        };
                        let oa = output_bytes(a.step(input()).unwrap());
                        let ob = output_bytes(b.step(input()).unwrap());
                        let what = format!("{cfg:?} {auto_reset:?} {workers:?} {level:?} step {t}");
                        assert_eq!(oa, ob, "outputs: {what}");
                        for i in 0..4 {
                            assert_eq!(a.state(i), b.state(i), "state {i}: {what}");
                        }
                    }
                }
            }
        }
    }
}

/// The golden scenarios' actions (`lcg-v1`, `tests/golden.rs`).
fn golden_actions(t: u32, instances: usize) -> Vec<i32> {
    (0..instances * MAX_AGENTS)
        .flat_map(|row| {
            let key = (u64::from(t) << 32) ^ u64::try_from(row).unwrap_or(0);
            let x = key
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            [
                i32::try_from((x >> 33) % 5).unwrap_or(0),
                i32::try_from((x >> 45) % 7).unwrap_or(0),
            ]
        })
        .collect()
}

/// The committed per-step output digests of golden fixture `name` (the
/// second field of each step line; the header lines and the record-hash
/// fields are skipped).
fn golden_outputs(name: &str) -> Vec<String> {
    let path = format!("{}/tests/golden/{name}.golden", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| l.starts_with(|c: char| c.is_ascii_digit()))
        .map(|l| l.split(' ').nth(1).unwrap().to_owned())
        .collect()
}

#[test]
fn test_survival_lanes_reproduce_the_golden_outputs() {
    for (name, cfg, auto_reset, steps) in [
        (
            "survival-v1-16-next-step",
            config(16, 32),
            AutoReset::NextStep,
            200,
        ),
        (
            "survival-v1-16-same-step",
            config(16, 32),
            AutoReset::SameStep,
            200,
        ),
        (
            "survival-v1-128-next-step",
            SurvivalConfig {
                agents: 128,
                size: 96,
                steps: 64,
                ..SurvivalConfig::default()
            },
            AutoReset::NextStep,
            100,
        ),
    ] {
        let want = golden_outputs(name);
        assert_eq!(want.len(), usize::try_from(steps).unwrap(), "{name}");
        for level in Resolved::available() {
            let (mut a, mut b) = pools(&cfg, 8, auto_reset, None, level);
            a.reset_all(Reseed::Keep).unwrap();
            b.reset_all(Reseed::Keep).unwrap();
            for t in 1..=steps {
                let acts = golden_actions(t, 8);
                let input = || ActionInput {
                    discrete: &acts,
                    continuous: &[],
                };
                let da = format!("{:016x}", checksum(&output_bytes(a.step(input()).unwrap())));
                let db = format!("{:016x}", checksum(&output_bytes(b.step(input()).unwrap())));
                let k = usize::try_from(t - 1).unwrap();
                assert_eq!(da, want[k], "{name} reference, step {t}");
                assert_eq!(db, want[k], "{name} {level:?}, step {t}");
                for i in 0..8 {
                    assert_eq!(
                        a.state(i),
                        b.state(i),
                        "{name} {level:?} state {i}, step {t}"
                    );
                }
            }
        }
    }
}

/// One instance stepped by `step` from `s` with `discrete`: the state, the
/// rewards, the died, spawned, terminated and truncated rows, and the end.
type Stepped = (
    SurvivalState,
    Vec<f32>,
    Vec<[bool; 4]>,
    Option<tessera_rl_core::EpisodeEnd>,
);

fn one_step(
    cfg: &SurvivalConfig,
    s: &SurvivalState,
    discrete: &[i32],
    seed: u64,
    lanes_at: Option<Resolved>,
) -> Stepped {
    let mut state = *s;
    let identity = RngIdentity::new(Seed::new(seed), LogicalId::new(0));
    let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
    let (mut rewards, mut term, mut trunc, mut died, mut spawned) = (
        vec![0.0f32; MAX_AGENTS],
        vec![Bool8::FALSE; MAX_AGENTS],
        vec![Bool8::FALSE; MAX_AGENTS],
        vec![Bool8::FALSE; MAX_AGENTS],
        vec![Bool8::FALSE; MAX_AGENTS],
    );
    let mut out = StepOut::new(&mut rewards, &mut term, &mut trunc, &mut died, &mut spawned);
    let actions = Actions::<()>::new(discrete, 2, &[], 0);
    match lanes_at {
        None => Survival::step(cfg, &mut state, &actions, &mut rng, &mut out),
        Some(level) => lanes::step(cfg, &mut state, &actions, &mut rng, &mut out, level),
    }
    let end = out.episode_end();
    let rows = died
        .iter()
        .zip(&spawned)
        .zip(&term)
        .zip(&trunc)
        .map(|(((d, s), t), u)| [d.get(), s.get(), t.get(), u.get()])
        .collect();
    (state, rewards, rows, end)
}

/// `s` pushed towards the rare rules: low stocks and health for the live
/// agents, and every fifth live agent dead and due to respawn.
fn stressed(s: &SurvivalState, seed: u64) -> SurvivalState {
    let mut t = *s;
    for a in 0..t.agent_count() {
        if !t.is_alive(a) {
            continue;
        }
        let key = seed ^ (u64::try_from(a).unwrap() << 12);
        if a % 5 == 4 {
            t.alive[a] = Bool8::FALSE;
            t.dead_for[a] = 2;
            continue;
        }
        t.food[a] = i16::try_from(below(key, 3)).unwrap();
        t.water[a] = i16::try_from(below(key ^ 2, 3)).unwrap();
        t.health[a] = 1 + i16::try_from(below(key ^ 3, 40)).unwrap();
    }
    t
}

#[test]
fn test_survival_lanes_whole_step_on_stressed_states_equals_the_reference() {
    // A crowded map (the smallest size, 32), so attacks meet their targets.
    let crowded = SurvivalConfig {
        agents: 128,
        size: 32,
        steps: 64,
        respawn: 2,
        ..SurvivalConfig::default()
    };
    let (mut starved, mut kills, mut respawns) = (false, false, false);
    for (c, cfg) in [config(21, 40), crowded].iter().enumerate() {
        for (k, s0) in states(cfg, 40 + u64::try_from(c).unwrap(), 30)
            .iter()
            .enumerate()
            .step_by(3)
        {
            let seed = u64::try_from(c * 1000 + k).unwrap();
            let mut s = stressed(s0, seed);
            for t in 1..=12 {
                let acts = random_actions(seed, t, 1);
                let want = one_step(cfg, &s, &acts, seed + u64::from(t), None);
                starved |= (0..s.agent_count())
                    .any(|a| s.is_alive(a) && (s.food[a] <= 1 || s.water[a] <= 1));
                kills |= (0..s.agent_count()).any(|a| want.0.kills[a] > s.kills[a]);
                respawns |= want.2.iter().any(|r| r[1]);
                for level in Resolved::available() {
                    let got = one_step(cfg, &s, &acts, seed + u64::from(t), Some(level));
                    assert_eq!(got, want, "config {c}, state {k}, step {t}, {level:?}");
                }
                s = want.0;
            }
        }
    }
    assert!(
        starved && kills && respawns,
        "starved={starved} kills={kills} respawns={respawns}"
    );
}
