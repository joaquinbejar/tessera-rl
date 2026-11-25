// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Survival-v1 scripted baseline (stage 3): it acts from the agent's own
//! view (`InfoHeuristic`), the information state is the observation's
//! window and `own` bytes, every baseline action is legal under the masks,
//! and the baseline keeps more agents alive than a uniform legal policy.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::cast_sign_loss,
    reason = "test scenarios: a failure is the assertion; small action indices"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic, ImperfectInformation};
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_world::survival::MAX_AGENTS;
use tessera_rl_world::{Survival, SurvivalConfig};

fn cfg(respawn: u32) -> SurvivalConfig {
    SurvivalConfig {
        agents: 32,
        size: 48,
        steps: 4096,
        respawn,
        ..SurvivalConfig::default()
    }
}

fn pool(c: SurvivalConfig, seed: u64) -> EnvPool<Survival, Masked> {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(seed));
    let mut p = EnvPool::with_masks(c, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn rng(seed: u64) -> StepRng {
    let id = RngIdentity::new(Seed::new(seed), LogicalId::DEFAULT);
    StepRng::new(&RngAuthority::engine_only(), &id, StreamId::DYNAMICS)
}

/// The baseline's actions for every slot of instance 0.
fn baseline_actions(
    p: &EnvPool<Survival, Masked>,
    c: &SurvivalConfig,
    rng: &mut StepRng,
) -> Vec<i32> {
    let s = p.state(0).unwrap();
    let mut acts = vec![0i32; MAX_AGENTS * 2];
    for a in 0..32 {
        let mut d = [0i32; 2];
        let mut w = ActionWriter::new(&mut d, &mut []);
        Survival::act(
            c,
            &s,
            AgentSlot::new(u16::try_from(a).unwrap(), MAX_AGENTS).unwrap(),
            rng,
            &mut w,
        );
        (acts[2 * a], acts[2 * a + 1]) = (d[0], d[1]);
    }
    acts
}

#[test]
fn test_survival_baseline_actions_are_legal_under_the_masks() {
    let c = cfg(2);
    let mut p = pool(c.clone(), 21);
    let mut r = rng(21);
    for t in 0..300 {
        let acts = baseline_actions(&p, &c, &mut r);
        let masks: Vec<bool> = p
            .outputs()
            .unwrap()
            .masks()
            .iter()
            .map(|f| f.get())
            .collect();
        for a in 0..32 {
            let row = &masks[a * 12..a * 12 + 12];
            // A dead agent does not act: the engine leaves its row empty.
            if !row.iter().any(|&f| f) {
                assert_eq!((acts[2 * a], acts[2 * a + 1]), (0, 0), "a dead agent stays");
                continue;
            }
            let (m, x) = (acts[2 * a] as usize, acts[2 * a + 1] as usize);
            assert!(
                row[m] && row[5 + x],
                "step {t}, agent {a}: ({m}, {x}) not legal"
            );
        }
        p.step(ActionInput {
            discrete: &acts,
            continuous: &[],
        })
        .unwrap();
    }
}

#[test]
fn test_survival_info_state_is_the_observation() {
    let c = cfg(2);
    let mut p = pool(c.clone(), 22);
    let mut r = rng(22);
    for _ in 0..20 {
        let acts = baseline_actions(&p, &c, &mut r);
        p.step(ActionInput {
            discrete: &acts,
            continuous: &[],
        })
        .unwrap();
    }
    let s = p.state(0).unwrap();
    let out = p.outputs().unwrap();
    let w = 4 * 11 * 11;
    let window = out.observation(0).unwrap();
    let own = out.observation(1).unwrap();
    for a in 0..32 {
        let info = Survival::info_state(
            &c,
            &s,
            AgentSlot::new(u16::try_from(a).unwrap(), MAX_AGENTS).unwrap(),
        );
        assert_eq!(
            &info.window[..w],
            &window[a * w..(a + 1) * w],
            "agent {a} window"
        );
        assert!(info.window[w..].iter().all(|&b| b == 0), "padding");
        let own_a: Vec<i32> = own[a * 24..(a + 1) * 24]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_ne_bytes(*b))
            .collect();
        assert_eq!(info.own.to_vec(), own_a, "agent {a} own");
    }
}

/// The mean number of live agents over `steps` steps with no respawn,
/// under the baseline or a uniform legal policy.
fn mean_alive(baseline: bool, seed: u64, steps: usize) -> f64 {
    let c = cfg(0);
    let mut p = pool(c.clone(), seed);
    let mut r = rng(seed);
    let mut lcg = seed;
    let mut total = 0usize;
    for _ in 0..steps {
        let acts = if baseline {
            baseline_actions(&p, &c, &mut r)
        } else {
            let masks: Vec<bool> = p
                .outputs()
                .unwrap()
                .masks()
                .iter()
                .map(|f| f.get())
                .collect();
            let mut acts = vec![0i32; MAX_AGENTS * 2];
            for a in 0..32 {
                let row = &masks[a * 12..a * 12 + 12];
                for (head, range) in [(0, 0..5usize), (1, 5..12)] {
                    let legal: Vec<usize> = range.clone().filter(|&i| row[i]).collect();
                    if legal.is_empty() {
                        continue;
                    }
                    lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    acts[2 * a + head] =
                        i32::try_from(legal[(lcg >> 33) as usize % legal.len()] - range.start)
                            .unwrap();
                }
            }
            acts
        };
        let out = p
            .step(ActionInput {
                discrete: &acts,
                continuous: &[],
            })
            .unwrap();
        if out.episode_terminated()[0].get() {
            break;
        }
        total += (0..32)
            .filter(|&a| p.state(0).unwrap().alive[a].get())
            .count();
    }
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let mean = total as f64 / steps as f64;
    mean
}

#[test]
fn test_survival_baseline_outlives_a_uniform_legal_policy() {
    for seed in [31, 32, 33] {
        let (b, u) = (mean_alive(true, seed, 400), mean_alive(false, seed, 400));
        assert!(
            b > u,
            "seed {seed}: baseline {b:.2} live agents on average, uniform {u:.2}"
        );
    }
}
