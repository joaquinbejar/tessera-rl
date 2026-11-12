// SPDX-License-Identifier: MIT OR Apache-2.0

//! A slow, readable reference of the Gumbel root (design note for #69,
//! steps 1 to 9), written from the note alone: plain loops and `Vec`, no
//! code shared with the implementation. It replays one call on a root
//! whose every visit to action `a` backs up the known value `h[a]` (the
//! Ladder game), from given starting root statistics.

use tessera_rl_num::math;

/// The inputs of one call.
pub struct Input<'a> {
    /// Tree priors and legality of the root edges.
    pub priors: &'a [f64],
    pub legal: &'a [bool],
    /// The call's Gumbel variates, one per action.
    pub g: &'a [f64],
    pub m: u32,
    pub n: u32,
    pub c_visit: f64,
    pub c_scale: f64,
    pub v_root: f64,
    /// Root edge visits and values before the call.
    pub visits: Vec<u32>,
    pub values: Vec<f64>,
    /// The value each visit of action `a` backs up to its root edge.
    pub h: &'a [f64],
}

/// What a call produces.
#[derive(Debug, PartialEq)]
pub struct Output {
    pub visits: Vec<u32>,
    pub values: Vec<u64>,
    pub considered: Vec<u32>,
    pub winner: u32,
    pub policy: Vec<u64>,
}

/// The uniform of `k` on the open interval and its Gumbel variate.
pub fn gumbel_variate(word: u64) -> f64 {
    let k = word >> 12;
    let numerator = 2 * k + 1;
    #[expect(clippy::cast_precision_loss, reason = "below 2^53, exact")]
    let u = numerator as f64 / 9_007_199_254_740_992.0;
    -math::ln(-math::ln(u))
}

fn completed_q(x: &Input<'_>, visits: &[u32], values: &[f64]) -> Vec<f64> {
    let mut total = 0u64;
    for &v in visits {
        total += u64::from(v);
    }
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let total_f = total as f64;
    let v_mix = if total == 0 {
        x.v_root
    } else {
        let mut prior_sum = 0.0;
        let mut weighted = 0.0;
        for a in 0..visits.len() {
            if visits[a] > 0 {
                prior_sum += x.priors[a];
                weighted += x.priors[a] * (values[a] / f64::from(visits[a]));
            }
        }
        if prior_sum > 0.0 {
            (x.v_root + (total_f / prior_sum) * weighted) / (1.0 + total_f)
        } else {
            x.v_root / (1.0 + total_f)
        }
    };
    let most = visits.iter().copied().max().unwrap_or(0);
    let mut sigma = Vec::new();
    for a in 0..visits.len() {
        let q = if visits[a] > 0 {
            values[a] / f64::from(visits[a])
        } else {
            v_mix
        };
        let rescaled = (q + 1.0) / 2.0;
        sigma.push((x.c_visit + f64::from(most)) * x.c_scale * rescaled);
    }
    sigma
}

/// Sorts `set` by `key` descending, ties to the lowest action.
fn rank(set: &[u32], key: &dyn Fn(u32) -> f64) -> Vec<u32> {
    let mut rest = set.to_vec();
    let mut out = Vec::new();
    while !rest.is_empty() {
        let mut best = 0;
        for i in 1..rest.len() {
            let (ki, kb) = (key(rest[i]), key(rest[best]));
            if ki > kb || (ki == kb && rest[i] < rest[best]) {
                best = i;
            }
        }
        out.push(rest.remove(best));
    }
    out
}

pub fn run(x: &Input<'_>) -> Output {
    let actions = x.priors.len();
    let mut logits = Vec::new();
    for a in 0..actions {
        logits.push(if x.legal[a] && x.priors[a] > 0.0 {
            math::ln(x.priors[a])
        } else {
            f64::NEG_INFINITY
        });
    }
    let candidates: Vec<u32> = (0..actions)
        .filter(|&a| logits[a] > f64::NEG_INFINITY)
        .map(|a| u32::try_from(a).unwrap())
        .collect();
    let gl = |a: u32| x.g[a as usize] + logits[a as usize];
    let mut considered = rank(&candidates, &gl);
    considered.truncate(usize::try_from(x.m).unwrap().min(considered.len()));
    let m = u32::try_from(considered.len()).unwrap();
    let mut phases = 0;
    while (1u32 << phases) < m {
        phases += 1;
    }
    let phases = phases.max(1);
    let mut visits = x.visits.clone();
    let mut values = x.values.clone();
    let mut survivors = considered.clone();
    let mut used = 0u32;
    let mut phase = 0;
    while used < x.n {
        let s = u32::try_from(survivors.len()).unwrap();
        let len = if phase < phases {
            ((x.n / (phases * s)).max(1) * s).min(x.n - used)
        } else {
            x.n - used
        };
        for i in 0..len {
            let a = survivors[(i % s) as usize] as usize;
            visits[a] += 1;
            values[a] += x.h[a];
        }
        used += len;
        if used >= x.n {
            break;
        }
        if phase < phases {
            let sigma = completed_q(x, &visits, &values);
            let score = |a: u32| gl(a) + sigma[a as usize];
            let ranked = rank(&survivors, &score);
            let kept = &ranked[..survivors.len().div_ceil(2)];
            survivors.retain(|a| kept.contains(a));
        }
        phase += 1;
    }
    let sigma = completed_q(x, &visits, &values);
    let score = |a: u32| gl(a) + sigma[a as usize];
    let winner = rank(&survivors, &score)[0];
    let mut z = Vec::new();
    for a in 0..actions {
        z.push(if logits[a] > f64::NEG_INFINITY {
            logits[a] + sigma[a]
        } else {
            f64::NEG_INFINITY
        });
    }
    let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut e = Vec::new();
    let mut sum = 0.0;
    for &za in &z {
        let v = if za > f64::NEG_INFINITY {
            math::exp(za - top)
        } else {
            0.0
        };
        e.push(v);
        sum += v;
    }
    Output {
        visits,
        values: values.iter().map(|v| v.to_bits()).collect(),
        considered,
        winner,
        policy: e.iter().map(|v| (v / sum).to_bits()).collect(),
    }
}
