// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Gumbel root (design note for #69, steps 1 to 9; Danihelka et al.
//! 2022): Gumbel-top-k sampling of `m` root actions, sequential halving
//! over the call's simulation budget, completed Q with `v_mix`,
//! `sigma` with the fixed `(q + 1) / 2` rescale, and the improved policy
//! `pi'`. Interior nodes stay PUCT.
//!
//! Pinned arithmetic (all `f64`, canonical controls, ADR-0008):
//! - `l(a) = math::ln(pi(a))` over the legal root actions with `pi(a) > 0`,
//!   `-inf` otherwise; `g` from [`crate::noise::gumbel`], the first `A`
//!   draws of the call's search stream.
//! - `S_0`: the `m' = min(m, |{l > -inf}|)` actions of largest `g + l`,
//!   ties to the lowest action index, kept in that order.
//! - `P = max(1, ceil_log2(m'))` phases; phase `j` gives each survivor
//!   `max(1, floor(n / (P * |S_j|)))` visits, the phase capped by the
//!   remaining budget; the schedule repeats the survivors (in `g + l`
//!   order) round robin. A last phase gives what is left to the single
//!   final survivor.
//! - At a phase end with budget left, the `ceil(|S_j| / 2)` best by
//!   `(g + l) + sigma(q')` survive, ties to the lowest action index, still
//!   in `g + l` order. When the budget ends at that phase end, the final
//!   choice takes the best survivor by the same score instead.
//! - `q(a) = W(a) / N(a)` for a visited root edge, from the root mover's
//!   view (the backup's sign convention); otherwise `v_mix =
//!   (v_root + (sum N / sum_vis pi) * sum_vis pi q) / (1 + sum N)`, sums
//!   in action order over every root edge; `v_mix = v_root` when
//!   `sum N = 0`, and the weighted term is 0 when the visited edges carry
//!   no prior mass (as `mctx` does).
//! - `q' = (q + 1) / 2`; `sigma(q') = ((c_visit + max N) * c_scale) * q'`
//!   with `max N` over every root edge at the decision (earlier calls
//!   included).
//! - `pi'(a) = exp(z(a) - max z) / sum`, `z = l + sigma(q')` over the
//!   actions with `l > -inf`, `0` elsewhere; `math::exp`, sum in action
//!   order.
//!
//! Gumbel and Dirichlet root noise are refused together by `Mcts::new`, so
//! they never share a call's `StreamId::SEARCH` draws: that is why no
//! draw order between them is defined.

use tessera_rl_num::math;
use tessera_rl_num::rng::StepRng;

use crate::error::SearchError;
use crate::noise;

/// Bytes per action counted for one tree's Gumbel buffers in the group
/// memory bound (Gumbel slots only). The buffers take 80 bytes per action
/// (`g`, logits and priors, 24; the root edge copies, 24; four `u32`
/// action lists, 16; two policies, 16), so 128 leaves 1.6 times that.
pub const GUMBEL_BYTES_PER_ACTION: u64 = 128;

/// The Gumbel root's options.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct GumbelConfig {
    /// Actions sampled at the root, `>= 1` (default 16).
    pub m: u32,
    /// `c_visit` of `sigma`, finite and `>= 0` (default 50).
    pub c_visit: f64,
    /// `c_scale` of `sigma`, finite and `>= 0` (default 1).
    pub c_scale: f64,
}

impl Default for GumbelConfig {
    fn default() -> Self {
        Self {
            m: 16,
            c_visit: 50.0,
            c_scale: 1.0,
        }
    }
}

/// How the root action of each simulation is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum RootSelection {
    /// PUCT at the root, as everywhere (the default).
    #[default]
    Puct,
    /// Gumbel-top-k with sequential halving.
    Gumbel(GumbelConfig),
}

/// One root edge's statistics at a decision.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RootEdge {
    pub(crate) visits: u32,
    pub(crate) value: f64,
    pub(crate) virtual_loss: u32,
}

fn gumbel_error(reason: &'static str) -> SearchError {
    SearchError::GumbelInvariant { reason }
}

fn at<T: Copy + Default>(v: &[T], i: usize) -> T {
    v.get(i).copied().unwrap_or_default()
}

fn idx(v: u32) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}

/// `ceil(log2(x))` for `x >= 1`, by bit operations.
#[must_use]
pub(crate) const fn ceil_log2(x: u32) -> u32 {
    if x <= 1 {
        0
    } else {
        u32::BITS - (x - 1).leading_zeros()
    }
}

/// The quantities of one decision over the root edges.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Decision {
    v_mix: f64,
    /// `(c_visit + max N) * c_scale`.
    scale: f64,
}

impl Decision {
    /// `sigma(q')` of an edge.
    fn sigma(self, e: RootEdge) -> f64 {
        let q = if e.visits == 0 {
            self.v_mix
        } else {
            e.value / f64::from(e.visits)
        };
        #[expect(
            clippy::manual_midpoint,
            reason = "the pinned rescale (q + 1) / 2 of the design note"
        )]
        let rescaled = (q + 1.0) / 2.0;
        self.scale * rescaled
    }
}

/// `v_mix` and `sigma`'s scale over `edges`, with tree priors `priors`.
/// Total: `v_mix = v_root` when no edge has a visit.
pub(crate) fn decision(
    cfg: &GumbelConfig,
    edges: &[RootEdge],
    priors: &[f64],
    v_root: f64,
) -> Decision {
    let mut sum_n = 0u64;
    let mut max_n = 0u32;
    let mut pi_vis = 0.0;
    let mut pq = 0.0;
    for (a, e) in edges.iter().enumerate() {
        sum_n += u64::from(e.visits);
        max_n = max_n.max(e.visits);
        if e.visits > 0 {
            let p = at(priors, a);
            pi_vis += p;
            pq += p * (e.value / f64::from(e.visits));
        }
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "visit counts are far below 2^53"
    )]
    let n = sum_n as f64;
    let v_mix = if sum_n == 0 {
        v_root
    } else if pi_vis > 0.0 {
        (v_root + (n / pi_vis) * pq) / (1.0 + n)
    } else {
        v_root / (1.0 + n)
    };
    Decision {
        v_mix,
        scale: (cfg.c_visit + f64::from(max_n)) * cfg.c_scale,
    }
}

/// The Gumbel state of one tree: this call's working state, and the
/// results of the last committed call.
pub(crate) struct Gumbel {
    pub(crate) cfg: GumbelConfig,
    g: Vec<f64>,
    logits: Vec<f64>,
    priors: Vec<f64>,
    edges: Vec<RootEdge>,
    /// `S_0`, in `g + l` order.
    considered: Vec<u32>,
    /// The current survivors, in `g + l` order.
    survivors: Vec<u32>,
    keep: Vec<u32>,
    budget: u32,
    phases: u32,
    phase: u32,
    /// Visits scheduled in the current phase.
    phase_len: u32,
    /// Visits of the current phase backed up.
    backed: u32,
    /// Visits of completed phases.
    used: u32,
    finished: bool,
    policy: Vec<f64>,
    winner: Option<u32>,
    /// The last committed call's results, with the key of its root.
    pub_policy: Vec<f64>,
    pub_considered: Vec<u32>,
    pub_winner: Option<(u64, u32)>,
}

impl Gumbel {
    pub(crate) fn new(cfg: GumbelConfig, actions: usize, budget: u32) -> Self {
        Self {
            cfg,
            g: vec![0.0; actions],
            logits: vec![0.0; actions],
            priors: vec![0.0; actions],
            edges: vec![
                RootEdge {
                    visits: 0,
                    value: 0.0,
                    virtual_loss: 0,
                };
                actions
            ],
            considered: Vec::with_capacity(actions),
            survivors: Vec::with_capacity(actions),
            keep: Vec::with_capacity(actions),
            budget,
            phases: 1,
            phase: 0,
            phase_len: 0,
            backed: 0,
            used: 0,
            finished: true,
            policy: vec![0.0; actions],
            winner: None,
            pub_policy: vec![0.0; actions],
            pub_considered: Vec::with_capacity(actions),
            pub_winner: None,
        }
    }

    /// The call's setup: `g` from `rng` (its first `A` draws), logits from
    /// the tree priors `pi` over `legal`, `S_0` and the first phase.
    pub(crate) fn begin(&mut self, rng: &mut StepRng, pi: impl Fn(usize) -> (f64, bool)) {
        noise::gumbel(rng, &mut self.g);
        self.setup(pi);
    }

    /// [`Self::begin`] after the draws of `g`.
    fn setup(&mut self, pi: impl Fn(usize) -> (f64, bool)) {
        for (a, (l, p)) in self
            .logits
            .iter_mut()
            .zip(self.priors.iter_mut())
            .enumerate()
        {
            let (prior, legal) = pi(a);
            *p = prior;
            *l = if legal && prior > 0.0 {
                math::ln(prior)
            } else {
                f64::NEG_INFINITY
            };
        }
        self.considered.clear();
        for (a, &l) in (0u32..).zip(&self.logits) {
            if l > f64::NEG_INFINITY {
                self.considered.push(a);
            }
        }
        let (g, logits) = (&self.g, &self.logits);
        let score = |a: u32| at(g, idx(a)) + at(logits, idx(a));
        self.considered.sort_unstable_by(|&x, &y| {
            score(y)
                .partial_cmp(&score(x))
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(x.cmp(&y))
        });
        let m = self
            .considered
            .len()
            .min(usize::try_from(self.cfg.m).unwrap_or(usize::MAX));
        self.considered.truncate(m);
        self.survivors.clear();
        self.survivors.extend_from_slice(&self.considered);
        let m = u32::try_from(m).unwrap_or(u32::MAX);
        self.phases = ceil_log2(m).max(1);
        self.phase = 0;
        self.used = 0;
        self.backed = 0;
        self.finished = false;
        self.winner = None;
        self.start_phase();
    }

    /// Sets the current phase's length from the survivors and the budget
    /// left.
    fn start_phase(&mut self) {
        let left = self.budget.saturating_sub(self.used);
        let s = u64::try_from(self.survivors.len())
            .unwrap_or(u64::MAX)
            .max(1);
        let len = if self.phase < self.phases {
            let per = (u64::from(self.budget) / (u64::from(self.phases) * s)).max(1);
            (per * s).min(u64::from(left))
        } else {
            u64::from(left)
        };
        self.phase_len = u32::try_from(len).unwrap_or(left);
        self.backed = 0;
    }

    /// Visits still to schedule in the current phase.
    #[must_use]
    pub(crate) fn phase_left(&self) -> u32 {
        if self.finished {
            0
        } else {
            self.phase_len.saturating_sub(self.backed)
        }
    }

    /// The root action of the `k`-th leaf of the next batch.
    #[must_use]
    pub(crate) fn action(&self, k: usize) -> Option<u32> {
        let n = self.survivors.len();
        if n == 0 {
            return None;
        }
        let i = idx(self.backed).saturating_add(k) % n;
        self.survivors.get(i).copied()
    }

    /// Records `n` backed-up visits of the current phase.
    pub(crate) fn backed_up(&mut self, n: usize) {
        self.backed = self
            .backed
            .saturating_add(u32::try_from(n).unwrap_or(u32::MAX));
    }

    /// Whether the current phase's last visit is backed up.
    #[must_use]
    pub(crate) fn phase_complete(&self) -> bool {
        !self.finished && self.backed >= self.phase_len
    }

    /// Loads the root edges' statistics for a decision; checks no virtual
    /// loss is pending and that some edge has a visit.
    fn load(&mut self, edge: impl Fn(usize) -> RootEdge) -> Result<(), SearchError> {
        let mut any = false;
        for (a, e) in self.edges.iter_mut().enumerate() {
            *e = edge(a);
            any |= e.visits > 0;
        }
        let pending = self.edges.iter().any(|e| e.virtual_loss != 0);
        debug_assert!(!pending, "a Gumbel decision with pending virtual loss");
        if pending {
            return Err(gumbel_error("a decision with pending virtual loss"));
        }
        debug_assert!(any, "a Gumbel decision with no root visit");
        if !any {
            return Err(gumbel_error("a decision with no root visit"));
        }
        Ok(())
    }

    /// `(g + l) + sigma(q')` of action `a`.
    fn score(&self, d: Decision, a: u32) -> f64 {
        let i = idx(a);
        let e = self.edges.get(i).copied().unwrap_or(RootEdge {
            visits: 0,
            value: 0.0,
            virtual_loss: 0,
        });
        (at(&self.g, i) + at(&self.logits, i)) + d.sigma(e)
    }

    /// The phase end, once its last visit is backed up (a no-op at any
    /// other time): halves the survivors if budget is left and starts the
    /// next phase.
    pub(crate) fn advance(
        &mut self,
        edge: impl Fn(usize) -> RootEdge,
        v_root: Option<f64>,
    ) -> Result<(), SearchError> {
        if !self.phase_complete() {
            return Ok(());
        }
        self.load(edge)?;
        self.used = self.used.saturating_add(self.phase_len);
        if self.used >= self.budget {
            self.finished = true;
            self.phase_len = 0;
            self.backed = 0;
            return Ok(());
        }
        if self.phase < self.phases {
            let v_root = v_root.ok_or(gumbel_error("no root value"))?;
            let d = decision(&self.cfg, &self.edges, &self.priors, v_root);
            let keep = self.survivors.len().div_ceil(2);
            self.keep.clear();
            for &s in &self.survivors {
                let ss = self.score(d, s);
                #[expect(clippy::float_cmp, reason = "an exact tie goes to the lowest action")]
                let beaten = self
                    .survivors
                    .iter()
                    .filter(|&&t| {
                        let st = self.score(d, t);
                        st > ss || (st == ss && t < s)
                    })
                    .count();
                if beaten < keep {
                    self.keep.push(s);
                }
            }
            core::mem::swap(&mut self.survivors, &mut self.keep);
        }
        self.phase = self.phase.saturating_add(1);
        self.start_phase();
        Ok(())
    }

    /// The call's end: the final action and `pi'`.
    pub(crate) fn finish(
        &mut self,
        edge: impl Fn(usize) -> RootEdge,
        v_root: Option<f64>,
    ) -> Result<(), SearchError> {
        self.load(edge)?;
        let v_root = v_root.ok_or(gumbel_error("no root value"))?;
        let d = decision(&self.cfg, &self.edges, &self.priors, v_root);
        let mut best: Option<(f64, u32)> = None;
        for &s in &self.survivors {
            let sc = self.score(d, s);
            #[expect(clippy::float_cmp, reason = "an exact tie goes to the lowest action")]
            let better = best.is_none_or(|(b, a)| sc > b || (sc == b && s < a));
            if better {
                best = Some((sc, s));
            }
        }
        self.winner = best.map(|(_, a)| a);
        let mut top = f64::NEG_INFINITY;
        for (a, p) in self.policy.iter_mut().enumerate() {
            let l = at(&self.logits, a);
            *p = if l > f64::NEG_INFINITY {
                let e = self.edges.get(a).copied().unwrap_or(RootEdge {
                    visits: 0,
                    value: 0.0,
                    virtual_loss: 0,
                });
                l + d.sigma(e)
            } else {
                f64::NEG_INFINITY
            };
            top = top.max(*p);
        }
        let mut sum = 0.0;
        for p in &mut self.policy {
            *p = if *p > f64::NEG_INFINITY {
                math::exp(*p - top)
            } else {
                0.0
            };
            sum += *p;
        }
        for p in &mut self.policy {
            *p /= sum;
        }
        Ok(())
    }

    /// The phase, its length, visits backed in it, visits of completed
    /// phases, whether the schedule is finished, and the survivors.
    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> (u32, u32, u32, u32, bool, Vec<u32>) {
        (
            self.phase,
            self.phase_len,
            self.backed,
            self.used,
            self.finished,
            self.survivors.clone(),
        )
    }

    /// Publishes the call's results for the root `key`.
    pub(crate) fn publish(&mut self, key: u64) {
        core::mem::swap(&mut self.policy, &mut self.pub_policy);
        self.pub_considered.clear();
        self.pub_considered.extend_from_slice(&self.considered);
        self.pub_winner = self.winner.map(|a| (key, a));
    }

    /// The last committed call's `pi'`, if its root is `key`.
    pub(crate) fn improved_policy(&self, key: u64) -> Option<&[f64]> {
        self.pub_winner
            .filter(|&(k, _)| k == key)
            .map(|_| self.pub_policy.as_slice())
    }

    /// The last committed call's `S_0`, if its root is `key`.
    pub(crate) fn considered(&self, key: u64) -> Option<&[u32]> {
        self.pub_winner
            .filter(|&(k, _)| k == key)
            .map(|_| self.pub_considered.as_slice())
    }

    /// The last committed call's action, if its root is `key`.
    pub(crate) fn winner(&self, key: u64) -> Option<u32> {
        self.pub_winner.filter(|&(k, _)| k == key).map(|(_, a)| a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLAT: RootEdge = RootEdge {
        visits: 2,
        value: 0.5,
        virtual_loss: 0,
    };

    /// Equal `g`, equal priors and equal edges: every tie goes to the
    /// lowest action, in the top-k, at the halving and in the final
    /// choice.
    #[test]
    fn test_gumbel_exact_ties_go_to_the_lowest_action() {
        let mut g = Gumbel::new(
            GumbelConfig {
                m: 4,
                ..GumbelConfig::default()
            },
            6,
            12,
        );
        g.setup(|a| (if a == 1 { 0.0 } else { 0.2 }, true));
        assert_eq!(g.considered, [0, 2, 3, 4]);
        // Phase 0: 4 survivors, P = 2, one visit each.
        assert_eq!(g.phase_len, 4);
        g.backed_up(4);
        g.advance(|_| FLAT, Some(0.0)).unwrap();
        assert_eq!(g.survivors, [0, 2]);
        // Phase 1: three visits each.
        assert_eq!(g.phase_len, 6);
        g.backed_up(6);
        g.advance(|_| FLAT, Some(0.0)).unwrap();
        assert_eq!(g.survivors, [0]);
        // Final choice among tied survivors.
        g.survivors.clear();
        g.survivors.extend_from_slice(&[3, 2, 4]);
        g.finish(|_| FLAT, Some(0.0)).unwrap();
        assert_eq!(g.winner, Some(2));
    }

    /// `v_mix` is `v_root` when no edge has a visit (the function is
    /// total; decisions refuse that case).
    #[test]
    fn test_gumbel_v_mix_without_visits_is_v_root() {
        let none = [RootEdge {
            visits: 0,
            value: 0.0,
            virtual_loss: 0,
        }; 3];
        let d = decision(&GumbelConfig::default(), &none, &[0.5, 0.25, 0.25], 0.375);
        assert_eq!(d.v_mix.to_bits(), 0.375_f64.to_bits());
        assert_eq!(d.scale.to_bits(), 50.0_f64.to_bits());
        // Visited edges with no prior mass: the weighted term is 0.
        let some = [FLAT, none[0], none[0]];
        let d = decision(&GumbelConfig::default(), &some, &[0.0, 0.5, 0.5], 0.375);
        assert_eq!(d.v_mix.to_bits(), (0.375_f64 / 3.0).to_bits());
    }

    /// Drives the phases with every visit backed up and returns
    /// `(survivors, phase_len)` per phase.
    fn phases(m: u32, actions: usize, n: u32) -> Vec<(usize, u32)> {
        let mut g = Gumbel::new(
            GumbelConfig {
                m,
                ..GumbelConfig::default()
            },
            actions,
            n,
        );
        g.setup(|_| (0.1, true));
        let mut out = Vec::new();
        while g.phase_left() > 0 {
            out.push((g.survivors.len(), g.phase_len));
            g.backed_up(idx(g.phase_len));
            g.advance(|_| FLAT, Some(0.0)).unwrap();
        }
        out
    }

    /// The budget left after the last halving phase goes to the final
    /// survivor; the sequences agree with an independent script.
    #[test]
    fn test_gumbel_phase_schedule_pinned_when_budget_is_left_after_halving() {
        assert_eq!(phases(16, 7, 64), [(7, 21), (4, 20), (2, 20), (1, 3)]);
        assert_eq!(
            phases(16, 16, 100),
            [(16, 16), (8, 24), (4, 24), (2, 24), (1, 12)]
        );
    }

    /// The note's operation order `(v_root + (n / pi_vis) * pq) / (1 + n)`
    /// differs in the last bit from `n * (pq / pi_vis)` here.
    #[test]
    fn test_gumbel_v_mix_operation_order_bits_pinned() {
        let edges = [0.5, -0.25, 0.1].map(|value| RootEdge {
            visits: 1,
            value,
            virtual_loss: 0,
        });
        let d = decision(&GumbelConfig::default(), &edges, &[0.3, 0.25, 0.2], 0.0625);
        // The other order gives 0x3fbf_851e_b851_eb86.
        assert_eq!(d.v_mix.to_bits(), 0x3fbf_851e_b851_eb85);
    }

    /// An illegal action with a positive prior is never considered and
    /// gets `pi' = 0`.
    #[test]
    fn test_gumbel_illegal_action_with_prior_is_excluded() {
        let mut g = Gumbel::new(GumbelConfig::default(), 4, 8);
        g.setup(|a| (0.25, a != 2));
        assert_eq!(g.considered, [0, 1, 3]);
        g.finish(|_| FLAT, Some(0.0)).unwrap();
        assert_eq!(g.policy[2].to_bits(), 0.0_f64.to_bits());
        assert!(g.policy.iter().enumerate().all(|(a, &p)| a == 2 || p > 0.0));
    }

    #[test]
    fn test_gumbel_ceil_log2_small_values_exact() {
        let want = [0, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4];
        for (x, w) in (0u32..).zip(want) {
            assert_eq!(ceil_log2(x), w, "x {x}");
        }
        assert_eq!(ceil_log2(16), 4);
        assert_eq!(ceil_log2(17), 5);
        assert_eq!(ceil_log2(u32::MAX), 32);
    }
}
