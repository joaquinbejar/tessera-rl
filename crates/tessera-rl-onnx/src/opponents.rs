// SPDX-License-Identifier: MIT OR Apache-2.0

//! ONNX league opponents (ADR-0088 item 2.3, erratum 2; #384 head C): the
//! engine's [`OpponentPolicies`] backed by one ONNX Runtime session per
//! model, at the evaluator's pinned settings (ADR-0057 section 2).
//!
//! A model is identified in the league by its content hash: the first 16
//! bytes of the sha256 of the model file. Its inputs are the evaluator's
//! (one per observation leaf, ADR-0057 section 5); its one output
//! `actions` is `int64` `[batch, 1]` or `[batch]`: the action of each row,
//! not logits, so the engine samples nothing (no RNG domain for
//! opponents). The engine's workers validate the actions; a run's own
//! allocations are outside the engine's zero-allocation guarantee, as for
//! the evaluator. Results are in ADR-0057's determinism class (one
//! runtime, never cross-platform).

use std::path::Path;

use ort::session::Session;
use ort::value::{TensorElementType, ValueType};
use sha2::{Digest, Sha256};
use tessera_rl_core::Environment;
use tessera_rl_engine::league::{OpponentPolicies, OpponentPolicyError, sealed};

use crate::error::OnnxError;
use crate::evaluator::{
    Buffer, Built, OnnxOptions, build_session, check_inputs, dims_match, empty_buffer, fill,
    read_model, shape_is, tensor,
};
use crate::layout::{LeafInput, action_count, leaf_inputs};
use crate::loader::load_runtime;
use crate::provenance::Provenance;

/// One opponent model.
#[derive(Debug)]
struct Model {
    content_hash: [u8; 16],
    session: Session,
    buffers: Vec<Buffer>,
    provenance: Provenance,
}

/// The ONNX policies of a league for environment `E`'s observations, to
/// hand to `EnvPool::with_onnx_opponents`.
#[derive(Debug)]
pub struct OnnxOpponents {
    models: Vec<Model>,
    leaves: Vec<LeafInput>,
    row_bytes: usize,
}

/// The league content hash of a model's bytes: the first 16 bytes of
/// their sha256.
#[must_use]
pub fn content_hash(model: &[u8]) -> [u8; 16] {
    let digest = Sha256::digest(model);
    let mut out = [0u8; 16];
    out.copy_from_slice(digest.get(..16).unwrap_or(&[0; 16]));
    out
}

impl OnnxOpponents {
    /// Loads the opponent models at `models` for `cfg`'s environment.
    ///
    /// # Errors
    ///
    /// Any loader error; [`OnnxError::Model`] if a model is too large or
    /// cannot be read or parsed; [`OnnxError::Provider`] as for the
    /// evaluator; [`OnnxError::Contract`] if the action space is not
    /// `Discrete`, a model's inputs or its `actions` output do not match,
    /// or two models have the same content hash.
    pub fn new<E: Environment>(
        cfg: &E::Config,
        models: &[&Path],
        options: &OnnxOptions,
    ) -> Result<Self, OnnxError> {
        let runtime = load_runtime(&options.library, options.telemetry)?;
        let contract = |what: String| OnnxError::Contract { what };
        let leaves = leaf_inputs(&E::observation_space(cfg).map_err(|e| contract(e.to_string()))?)?;
        action_count(&E::action_space(cfg).map_err(|e| contract(e.to_string()))?)?;
        let mut loaded: Vec<Model> = Vec::with_capacity(models.len());
        for &path in models {
            let bytes = read_model(path, options.max_model_bytes)?;
            let hash = content_hash(&bytes);
            if loaded.iter().any(|m| m.content_hash == hash) {
                return Err(contract(format!(
                    "{} has the content hash of an earlier model",
                    path.display()
                )));
            }
            let session =
                build_session(runtime.environment(), &bytes, options).map_err(|e| match e {
                    Built::Provider(reason) => OnnxError::Provider { reason },
                    Built::Model(reason) => OnnxError::Model {
                        path: path.to_path_buf(),
                        reason,
                    },
                })?;
            check_inputs(&session, &leaves)?;
            check_actions(&session)?;
            let buffers = leaves
                .iter()
                .map(|l| empty_buffer(l.dtype))
                .collect::<Result<_, _>>()?;
            loaded.push(Model {
                content_hash: hash,
                session,
                buffers,
                provenance: Provenance::new(runtime, crate::hash::sha256(&bytes), options),
            });
        }
        let row_bytes = leaves.iter().map(LeafInput::bytes).sum();
        Ok(Self {
            models: loaded,
            leaves,
            row_bytes,
        })
    }

    /// Each model's content hash and provenance, in load order.
    pub fn models(&self) -> impl Iterator<Item = (&[u8; 16], &Provenance)> {
        self.models.iter().map(|m| (&m.content_hash, &m.provenance))
    }
}

/// `actions`: `int64`, `[batch, 1]` or `[batch]`.
fn check_actions(session: &Session) -> Result<(), OnnxError> {
    let contract = |what: String| OnnxError::Contract { what };
    let output = session
        .outputs()
        .iter()
        .find(|o| o.name() == "actions")
        .ok_or_else(|| contract("the model has no output named \"actions\"".to_owned()))?;
    let ValueType::Tensor { ty, shape, .. } = output.dtype() else {
        return Err(contract("output \"actions\" is not a tensor".to_owned()));
    };
    if *ty != TensorElementType::Int64 {
        return Err(contract(format!("output \"actions\" is {ty:?}, not int64")));
    }
    if !(shape.len() == 1 || dims_match(shape, &[1])) {
        return Err(contract(format!(
            "output \"actions\" has shape {:?}, expected [batch] or [batch, 1]",
            &shape[..]
        )));
    }
    Ok(())
}

impl sealed::Sealed for OnnxOpponents {}

impl OpponentPolicies for OnnxOpponents {
    fn serves(&self, content_hash: &[u8; 16]) -> bool {
        self.models.iter().any(|m| m.content_hash == *content_hash)
    }

    fn act(
        &mut self,
        content_hash: &[u8; 16],
        rows: &[u8],
        n: usize,
        actions: &mut [i64],
    ) -> Result<(), OpponentPolicyError> {
        let failed = |reason: String| OpponentPolicyError { reason };
        let model = self
            .models
            .iter_mut()
            .find(|m| m.content_hash == *content_hash)
            .ok_or_else(|| failed("no model with this content hash".to_owned()))?;
        if rows.len() != n.saturating_mul(self.row_bytes) || actions.len() != n {
            return Err(failed(format!(
                "{} bytes and {} actions for {n} rows of {} bytes, one head",
                rows.len(),
                actions.len(),
                self.row_bytes
            )));
        }
        fill(&self.leaves, &mut model.buffers, rows, n, self.row_bytes);
        let mut inputs: Vec<(String, ort::session::SessionInputValue<'_>)> =
            Vec::with_capacity(self.leaves.len());
        for (leaf, buffer) in self.leaves.iter().zip(&model.buffers) {
            let mut dims = Vec::with_capacity(leaf.shape.len() + 1);
            dims.push(n);
            dims.extend(
                leaf.shape
                    .iter()
                    .map(|&d| usize::try_from(d).unwrap_or(usize::MAX)),
            );
            inputs.push((
                leaf.name.clone(),
                tensor(buffer, dims).map_err(|e| failed(e.to_string()))?,
            ));
        }
        let outputs = model.session.run(inputs).map_err(|e| failed(e.to_string()));
        // As the evaluator: restore the canonical floating-point controls
        // before the engine steps (ADR-0037, ADR-0057).
        let _ = tessera_rl_num::fp::ensure_canonical();
        let outputs = outputs?;
        let out = outputs
            .get("actions")
            .ok_or_else(|| failed("no actions output".to_owned()))?;
        let (shape, data) = out
            .try_extract_tensor::<i64>()
            .map_err(|e| failed(e.to_string()))?;
        if !(shape_is(shape, &[n]) || shape_is(shape, &[n, 1])) || data.len() != n {
            return Err(failed(format!(
                "output \"actions\" has shape {shape:?}, expected [{n}] or [{n}, 1]"
            )));
        }
        actions.copy_from_slice(data);
        Ok(())
    }
}
