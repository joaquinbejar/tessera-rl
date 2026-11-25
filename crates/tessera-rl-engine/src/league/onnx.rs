// SPDX-License-Identifier: MIT OR Apache-2.0

//! ONNX opponents (ADR-0088 item 2.3, erratum 2; #384 head C): one batch
//! per ONNX policy per step, on the calling thread, through a sealed
//! [`OpponentPolicies`] trait object (the engine cannot depend on
//! `tessera-rl-onnx`). The batch reads the observation rows the engine
//! wrote at the previous reset or step, in ascending policy id, then
//! instance, then slot order; the engine writes the actions into its own
//! action copy after the boundary (a value that is not an `i32` poisons at
//! once), and the workers validate them against the heads and the masks
//! (ADR-0079 item 3.2). A pool without an
//! ONNX policy never reaches this module.

use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout};
use thiserror::Error;

use super::{OpponentKind, sealed};
use crate::error::EngineError;
use crate::layout::OPPONENT_ONNX;
use crate::masks::MaskProvider;
use crate::pool::EnvPool;

/// Why an ONNX opponent pass failed. The pool is not poisoned (no state
/// was written); the step returns the error.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("ONNX opponent policy: {reason}")]
pub struct OpponentPolicyError {
    /// What failed.
    pub reason: String,
}

/// The ONNX policies of a league, implemented by `tessera-rl-onnx`
/// (ADR-0088 erratum 2). The engine calls [`Self::act`] once per ONNX
/// policy per step, on the calling thread.
pub trait OpponentPolicies: sealed::Sealed + Send {
    /// Whether this object runs the policy whose model has `content_hash`.
    fn serves(&self, content_hash: &[u8; 16]) -> bool;

    /// Runs the policy of `content_hash` on `n` observation rows (`rows`,
    /// one agent's exported leaves each, concatenated) and writes `n x
    /// heads` actions into `actions` (the model's `int64` output).
    ///
    /// # Errors
    ///
    /// [`OpponentPolicyError`] for a failed run.
    fn act(
        &mut self,
        content_hash: &[u8; 16],
        rows: &[u8],
        n: usize,
        actions: &mut [i64],
    ) -> Result<(), OpponentPolicyError>;
}

/// The per-step ONNX pass of one pool: the policies and buffers
/// preallocated for every `(instance, opponent slot)` pair.
pub(crate) struct OnnxPass {
    pub(crate) policies: Box<dyn OpponentPolicies>,
    /// Bytes of one agent's exported observation row.
    pub(crate) row_bytes: usize,
    /// The gathered rows of one policy's batch.
    pub(crate) rows: Vec<u8>,
    /// The batch's actions.
    pub(crate) out: Vec<i64>,
    /// The batch's `(instance, slot)` pairs, in order.
    pub(crate) pairs: Vec<(u32, u16)>,
}

impl core::fmt::Debug for OnnxPass {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OnnxPass")
            .field("row_bytes", &self.row_bytes)
            .finish_non_exhaustive()
    }
}

impl<E, M> EnvPool<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    /// Sets the ONNX policies of the pool's league (ADR-0088 item 2.3,
    /// erratum 2), before the first reset, once: every `Onnx` policy of the
    /// league's view must be one `policies` serves. Allocates the batch
    /// buffers for every `(instance, opponent slot)` pair.
    ///
    /// # Errors
    ///
    /// [`EngineError::League`] without a league, after the first reset, a
    /// second time, with continuous action heads, or for an ONNX policy of
    /// the view that `policies` does not serve.
    pub fn with_onnx_opponents(
        mut self,
        policies: Box<dyn OpponentPolicies>,
    ) -> Result<Self, EngineError> {
        let bad = |reason| Err(EngineError::League { reason });
        let Some(league) = self.league.as_ref() else {
            return bad("ONNX opponents need a league (with_league first)");
        };
        if self.onnx.is_some() {
            return bad("ONNX opponents are set once");
        }
        if self.headers.iter().any(|h| !h.engine().fresh.get()) {
            return bad("ONNX opponents are set before the first reset");
        }
        if !self.actions.continuous.is_empty() {
            return bad("ONNX opponents take discrete actions only");
        }
        for kind in league.view.kinds() {
            if let OpponentKind::Onnx { content_hash } = kind
                && !policies.serves(content_hash)
            {
                return bad("an ONNX policy of the league that the policies do not serve");
            }
        }
        let n = self.headers.len();
        let a = E::MAX_AGENTS;
        let rows = n.checked_mul(a).ok_or_else(crate::pool::too_large)?;
        let row_bytes = (0..crate::MAX_LEAVES)
            .map_while(|l| self.outputs.observation(l).map(<[u8]>::len))
            .map(|len| len.checked_div(rows).unwrap_or(0))
            .sum::<usize>();
        let dh = self.actions.discrete.len();
        self.onnx = Some(Box::new(OnnxPass {
            policies,
            row_bytes,
            rows: vec![
                0;
                rows.checked_mul(row_bytes)
                    .ok_or_else(crate::pool::too_large)?
            ],
            out: vec![0; rows.checked_mul(dh).ok_or_else(crate::pool::too_large)?],
            pairs: Vec::with_capacity(rows),
        }));
        Ok(self)
    }

    /// Whether the pool's ONNX opponents serve `content_hash` (a replay's
    /// check before any event, ADR-0088 section 8).
    pub(crate) fn serves_onnx(&self, content_hash: &[u8; 16]) -> bool {
        self.onnx
            .as_ref()
            .is_some_and(|p| p.policies.serves(content_hash))
    }

    /// The step's ONNX opponent pass, on the calling thread, after the
    /// boundary copied the caller's actions: per ONNX policy in id order,
    /// the acting opponent slots of the instances assigned to it, in
    /// instance then slot order, run in one batch; the actions are written
    /// into the engine's action copy (the workers then validate them
    /// against the heads and the masks). No allocation.
    pub(crate) fn run_onnx_opponents(&mut self) -> Result<(), EngineError> {
        let Self {
            league,
            onnx,
            headers,
            outputs,
            actions,
            discrete,
            poisoned,
            ..
        } = self;
        let Some(league) = league.as_deref() else {
            return Ok(());
        };
        let any_onnx = actions.opponent_mode.contains(&OPPONENT_ONNX);
        let Some(pass) = onnx.as_deref_mut() else {
            if any_onnx {
                return Err(EngineError::League {
                    reason: "an ONNX opponent without with_onnx_opponents",
                });
            }
            return Ok(());
        };
        if !any_onnx {
            return Ok(());
        }
        let instances = headers.len();
        let agents = E::MAX_AGENTS;
        let heads = actions.discrete.len();
        let mut leaves: [&[u8]; crate::MAX_LEAVES] = [&[]; crate::MAX_LEAVES];
        for (leaf_index, slot) in leaves.iter_mut().enumerate() {
            if let Some(leaf) = outputs.observation(leaf_index) {
                *slot = leaf;
            }
        }
        for (policy, kind) in league.view.kinds().iter().enumerate() {
            let OpponentKind::Onnx { content_hash } = kind else {
                continue;
            };
            pass.pairs.clear();
            let mut used = 0usize;
            for (i, header) in headers.iter().enumerate() {
                let engine = header.engine();
                let assigned = league
                    .assignment
                    .get(i)
                    .copied()
                    .and_then(|x| usize::try_from(x).ok());
                if assigned != Some(policy)
                    || actions.opponent_mode.get(i) != Some(&OPPONENT_ONNX)
                    || engine.pending_reset.get()
                {
                    continue;
                }
                for (slot, row) in header.agents().iter().enumerate() {
                    let opponent = slot < 64 && (league.slots >> slot) & 1 == 1;
                    if !(opponent && row.alive.get() && row.acting.get()) {
                        continue;
                    }
                    let row_index = i * agents + slot;
                    for leaf in leaves.iter().filter(|leaf| !leaf.is_empty()) {
                        let leaf_bytes = leaf.len() / (instances * agents).max(1);
                        if let (Some(src), Some(dst)) = (
                            leaf.get(row_index * leaf_bytes..(row_index + 1) * leaf_bytes),
                            pass.rows.get_mut(used..used + leaf_bytes),
                        ) {
                            dst.copy_from_slice(src);
                        }
                        used += leaf_bytes;
                    }
                    pass.pairs.push((
                        u32::try_from(i).unwrap_or(u32::MAX),
                        u16::try_from(slot).unwrap_or(u16::MAX),
                    ));
                }
            }
            let batch = pass.pairs.len();
            if batch == 0 {
                continue;
            }
            let (Some(rows), Some(out)) =
                (pass.rows.get(..used), pass.out.get_mut(..batch * heads))
            else {
                continue;
            };
            pass.policies.act(content_hash, rows, batch, out)?;
            write_actions(&pass.pairs, out, heads, agents, discrete)
                .inspect_err(|_| *poisoned = true)?;
        }
        Ok(())
    }
}

/// Writes a batch's actions into the engine's action copy at their
/// `(instance, slot)` rows. An action that is not a valid `i32` row value
/// is an invalid opponent action (ADR-0088 item 2.3) here, and the caller
/// poisons the pool; the workers check every other value against the
/// head's choices and the masks, as for heuristic opponents.
fn write_actions(
    pairs: &[(u32, u16)],
    out: &[i64],
    heads: usize,
    agents: usize,
    discrete: &mut [i32],
) -> Result<(), EngineError> {
    for (pair, &(instance, slot)) in pairs.iter().enumerate() {
        let row_index = usize::try_from(instance).unwrap_or(0) * agents + usize::from(slot);
        for head in 0..heads {
            let raw = out.get(pair * heads + head).copied().unwrap_or(-1);
            let value = u32::try_from(raw)
                .ok()
                .and_then(|raw| i32::try_from(raw).ok())
                .ok_or(EngineError::OpponentAction { instance, slot })?;
            if let Some(d) = discrete.get_mut(row_index * heads + head) {
                *d = value;
            }
        }
    }
    Ok(())
}
