// SPDX-License-Identifier: MIT OR Apache-2.0

//! The PPO loop: rollouts from a Tessera pool, GAE, clipped updates.

use core::num::NonZeroU32;

use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use tessera_rl_classic::{CartPole, CartPoleConfig};
use tessera_rl_engine::{ActionInput, AutoReset, EngineError, EnvPool, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

use crate::model::ActorCritic;
use crate::rng::Rng;

type B = Autodiff<NdArray<f32>>;

/// Pool size.
pub const NUM_ENVS: u64 = 16;
/// Steps per environment per rollout.
pub const ROLLOUT: u64 = 128;
const OBS: usize = 4;
const ACTIONS: usize = 2;
const EPOCHS: usize = 4;
const MINIBATCHES: usize = 4;
const GAMMA: f32 = 0.99;
const LAMBDA: f32 = 0.95;
const CLIP: f32 = 0.2;
const LR: f64 = 2.5e-4;
const VALUE_COEF: f32 = 0.5;
const ENTROPY_COEF: f32 = 0.01;

/// A run's settings.
pub struct Config {
    /// Seeds the pool, the network and the sampler.
    pub seed: u64,
    /// Environment steps (all instances).
    pub total_steps: u64,
}

/// A run's outcome.
pub struct Report {
    /// Environment steps taken.
    pub steps: u64,
    /// Episodes ended.
    pub episodes: usize,
    /// Mean return of the last 20 episodes (0 with none).
    pub last_mean: f32,
    /// Whether every loss was finite.
    pub finite: bool,
}

struct Batch {
    obs: Vec<f32>,
    actions: Vec<i64>,
    logp: Vec<f32>,
    values: Vec<f32>,
    rewards: Vec<f32>,
    dones: Vec<f32>,
}

fn observations(pool: &EnvPool<CartPole>) -> Result<Vec<f32>, EngineError> {
    let out = pool.outputs()?;
    let bytes = out.observation(0).unwrap_or_default();
    Ok(bytes
        .chunks_exact(4)
        .map(|c| <[u8; 4]>::try_from(c).map_or(0.0, f32::from_le_bytes))
        .collect())
}

/// Trains and reports.
///
/// # Errors
///
/// An engine error from the pool.
pub fn train(config: &Config) -> Result<Report, EngineError> {
    let device = Default::default();
    B::seed(&device, config.seed);
    let mut rng = Rng::new(config.seed ^ 0x5eed);
    let n = usize::try_from(NUM_ENVS).unwrap_or(16);
    let t_len = usize::try_from(ROLLOUT).unwrap_or(128);
    let options = PoolOptions {
        auto_reset: AutoReset::SameStep,
        ..PoolOptions::new(NonZeroU32::new(16).unwrap_or(NonZeroU32::MIN), Seed::new(config.seed))
    };
    let mut pool = EnvPool::<CartPole>::new(CartPoleConfig::default(), options)?;
    pool.reset_all(Reseed::Keep)?;
    let mut model = ActorCritic::<B>::new(OBS, ACTIONS, &device);
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();
    let mut returns: Vec<f32> = Vec::new();
    let mut steps = 0u64;
    let mut finite = true;
    let mut obs = observations(&pool)?;
    while steps < config.total_steps {
        let mut b = Batch {
            obs: Vec::with_capacity(t_len * n * OBS),
            actions: Vec::with_capacity(t_len * n),
            logp: Vec::with_capacity(t_len * n),
            values: Vec::with_capacity(t_len * n),
            rewards: Vec::with_capacity(t_len * n),
            dones: Vec::with_capacity(t_len * n),
        };
        for _ in 0..t_len {
            let x = Tensor::<B, 2>::from_data(TensorData::new(obs.clone(), [n, OBS]), &device);
            let probs = softmax(model.logits(x.clone()), 1).into_data().to_vec::<f32>().unwrap_or_default();
            let values = model.value(x).into_data().to_vec::<f32>().unwrap_or_default();
            let mut acts = Vec::with_capacity(n);
            for (i, row) in probs.chunks_exact(ACTIONS).enumerate() {
                let u = rng.unit() as f32;
                let a = usize::from(u >= row.first().copied().unwrap_or(0.5));
                b.logp.push(row.get(a).copied().unwrap_or(1.0).max(1e-8).ln());
                b.values.push(values.get(i).copied().unwrap_or(0.0));
                b.actions.push(i64::try_from(a).unwrap_or(0));
                acts.push(i32::try_from(a).unwrap_or(0));
            }
            b.obs.extend_from_slice(&obs);
            pool.step(ActionInput { discrete: &acts, continuous: &[] })?;
            let out = pool.outputs()?;
            b.rewards.extend_from_slice(out.rewards());
            for i in 0..n {
                let ended = out.episode_terminated().get(i).is_some_and(|f| f.get())
                    || out.episode_truncated().get(i).is_some_and(|f| f.get());
                b.dones.push(if ended { 1.0 } else { 0.0 });
                if ended {
                    returns.push(out.episode_return().get(i).copied().unwrap_or(0.0));
                }
            }
            obs = observations(&pool)?;
            steps += NUM_ENVS;
        }
        let last_x = Tensor::<B, 2>::from_data(TensorData::new(obs.clone(), [n, OBS]), &device);
        let last_v = model.value(last_x).into_data().to_vec::<f32>().unwrap_or_default();
        let (adv, ret) = gae(&b, &last_v, n, t_len);
        let (m, ok) = update(model, &mut optim, &b, &adv, &ret, &mut rng, &device);
        model = m;
        finite &= ok;
        let tail = &returns[returns.len().saturating_sub(20)..];
        #[expect(clippy::cast_precision_loss, reason = "at most 20 episodes")]
        let mean = if tail.is_empty() { 0.0 } else { tail.iter().sum::<f32>() / tail.len() as f32 };
        println!("ppo-cartpole: steps {steps} episodes {} last-20 mean {mean:.1}", returns.len());
    }
    let tail = &returns[returns.len().saturating_sub(20)..];
    #[expect(clippy::cast_precision_loss, reason = "at most 20 episodes")]
    let last_mean = if tail.is_empty() { 0.0 } else { tail.iter().sum::<f32>() / tail.len() as f32 };
    Ok(Report { steps, episodes: returns.len(), last_mean, finite })
}

/// Generalized advantage estimation; an ended step does not bootstrap
/// (`SameStep` auto-reset: the next observation starts a new episode).
fn gae(b: &Batch, last_v: &[f32], n: usize, t_len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut adv = vec![0.0f32; t_len * n];
    for i in 0..n {
        let mut next_adv = 0.0f32;
        let mut next_v = last_v.get(i).copied().unwrap_or(0.0);
        for t in (0..t_len).rev() {
            let k = t * n + i;
            let live = 1.0 - b.dones.get(k).copied().unwrap_or(1.0);
            let v = b.values.get(k).copied().unwrap_or(0.0);
            let r = b.rewards.get(k).copied().unwrap_or(0.0);
            let delta = r + GAMMA * next_v * live - v;
            next_adv = delta + GAMMA * LAMBDA * live * next_adv;
            if let Some(a) = adv.get_mut(k) {
                *a = next_adv;
            }
            next_v = v;
        }
    }
    let ret = adv.iter().zip(&b.values).map(|(a, v)| a + v).collect();
    (adv, ret)
}

fn take<T: Copy>(v: &[T], idx: &[usize], width: usize) -> Vec<T> {
    idx.iter()
        .flat_map(|&i| v.get(i * width..(i + 1) * width).unwrap_or_default().iter().copied())
        .collect()
}

fn update<O: Optimizer<ActorCritic<B>, B>>(
    mut model: ActorCritic<B>,
    optim: &mut O,
    b: &Batch,
    adv: &[f32],
    ret: &[f32],
    rng: &mut Rng,
    device: &<B as Backend>::Device,
) -> (ActorCritic<B>, bool) {
    let total = adv.len();
    let size = total / MINIBATCHES;
    let mut finite = true;
    let mut idx: Vec<usize> = (0..total).collect();
    for _ in 0..EPOCHS {
        rng.shuffle(&mut idx);
        for chunk in idx.chunks(size) {
            let m = chunk.len();
            let x = Tensor::<B, 2>::from_data(TensorData::new(take(&b.obs, chunk, OBS), [m, OBS]), device);
            let a = Tensor::<B, 2, Int>::from_data(TensorData::new(take(&b.actions, chunk, 1), [m, 1]), device);
            let old = Tensor::<B, 1>::from_data(TensorData::new(take(&b.logp, chunk, 1), [m]), device);
            let mb_adv = take(adv, chunk, 1);
            #[expect(clippy::cast_precision_loss, reason = "minibatch sizes are small")]
            let mean = mb_adv.iter().sum::<f32>() / m as f32;
            #[expect(clippy::cast_precision_loss, reason = "minibatch sizes are small")]
            let std = (mb_adv.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / m as f32).sqrt() + 1e-8;
            let norm: Vec<f32> = mb_adv.iter().map(|v| (v - mean) / std).collect();
            let adv_t = Tensor::<B, 1>::from_data(TensorData::new(norm, [m]), device);
            let ret_t = Tensor::<B, 1>::from_data(TensorData::new(take(ret, chunk, 1), [m]), device);

            let logits = model.logits(x.clone());
            let logp_all = log_softmax(logits, 1);
            let logp = logp_all.clone().gather(1, a).squeeze(1);
            let ratio = (logp - old).exp();
            let s1 = ratio.clone() * adv_t.clone();
            let s2 = ratio.clamp(1.0 - CLIP, 1.0 + CLIP) * adv_t;
            let policy_loss = s1.min_pair(s2).mean().neg();
            let value_loss = (model.value(x) - ret_t).powf_scalar(2.0).mean();
            let entropy = (logp_all.clone().exp() * logp_all).sum_dim(1).mean().neg();
            let loss = policy_loss + value_loss.mul_scalar(VALUE_COEF) - entropy.mul_scalar(ENTROPY_COEF);
            let l = loss.clone().into_data().to_vec::<f32>().unwrap_or_default();
            finite &= l.iter().all(|v| v.is_finite());
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(LR, model, grads);
        }
    }
    (model, finite)
}
