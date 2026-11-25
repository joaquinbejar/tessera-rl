// SPDX-License-Identifier: MIT OR Apache-2.0

//! Curriculum targets from Python (ADR-0081 items 3 and 5.5, #613): the
//! adaptive fields an environment exposes by name, and the validation of a
//! batch of changes before any write.

use tessera_rl_core::{Adaptive, EnvConfig, Environment};

use crate::env::BindError;

/// The adaptive fields of `E`, by name, as integers: how to read one from a
/// slot's target, and how to write one into a target and into a config copy
/// (whose own validation then checks the value).
pub(crate) struct AdaptiveFields<E: Environment> {
    /// The field names, in a fixed order; a field is its index here.
    pub(crate) names: &'static [&'static str],
    /// Field `i` of a target.
    pub(crate) get: fn(&Adaptive<E>, usize) -> Option<i64>,
    /// Writes `value` into field `i` of both; `false` when the value does
    /// not fit the field's type or `i` is not a field.
    pub(crate) set: fn(&mut E::Config, &mut Adaptive<E>, usize, i64) -> bool,
}

impl<E: Environment> Clone for AdaptiveFields<E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Environment> Copy for AdaptiveFields<E> {}

/// The refusal of the target calls on a pool without targets.
pub(crate) fn no_targets() -> BindError {
    BindError::Argument(
        "targets: the pool has no targets (make it with targets=True, sync mode)".to_owned(),
    )
}

/// Gather's adaptive field, `items` (version 2, #613).
pub(crate) fn gather() -> AdaptiveFields<tessera_rl_classic::Gather> {
    AdaptiveFields {
        names: &["items"],
        get: |a, i| (i == 0).then(|| i64::from(a.items)),
        set: |cfg, a, i, v| match (i, u8::try_from(v)) {
            (0, Ok(v)) => {
                cfg.items = v;
                a.items = v;
                true
            }
            _ => false,
        },
    }
}

/// Every slot's targets, field by field (`num_envs * names.len()` values).
pub(crate) fn flat<E: Environment>(
    fields: &AdaptiveFields<E>,
    targets: &[Adaptive<E>],
) -> Vec<i64> {
    targets
        .iter()
        .flat_map(|t| (0..fields.names.len()).map(move |i| (fields.get)(t, i).unwrap_or(0)))
        .collect()
}

/// The new target of every slot `changes` names, each from that slot's
/// current target with the listed fields replaced, all checked before the
/// caller writes any: a known slot, a known field, a value of the field's
/// type, and the config's own validation of a config holding the slot's
/// whole current target with the changes applied (ranges and cross-field
/// checks). A slot listed twice gets its changes in order.
///
/// # Errors
///
/// `ArgumentError` for an unknown slot or field or a length mismatch,
/// `ConfigError` (through the engine) for a value the config refuses.
pub(crate) fn merged<E: Environment>(
    fields: &AdaptiveFields<E>,
    cfg: &E::Config,
    targets: &[Adaptive<E>],
    (ids, which, values): (&[u32], &[u32], &[i64]),
) -> Result<Vec<(u32, Adaptive<E>)>, BindError>
where
    E::Config: Clone,
{
    if ids.len() != which.len() || ids.len() != values.len() {
        return Err(BindError::Argument(
            "set_targets: ids, fields and values differ in length".to_owned(),
        ));
    }
    let mut out: Vec<(u32, Adaptive<E>, E::Config)> = Vec::new();
    for ((&id, &field), &value) in ids.iter().zip(which).zip(values) {
        let Some(current) = usize::try_from(id).ok().and_then(|i| targets.get(i)) else {
            return Err(BindError::Argument(format!(
                "set_targets: no instance {id} (num_envs {})",
                targets.len()
            )));
        };
        let field = usize::try_from(field).unwrap_or(usize::MAX);
        let Some(name) = fields.names.get(field) else {
            return Err(BindError::Argument(format!(
                "set_targets: no adaptive field {field} (fields {:?})",
                fields.names
            )));
        };
        let at = match out.iter().position(|(i, _, _)| *i == id) {
            Some(at) => at,
            None => {
                // The config copy holds the slot's whole current target
                // first, so the config's cross-field checks see the merged
                // values, whatever the number of adaptive fields.
                let mut copy = cfg.clone();
                let mut target = *current;
                for i in 0..fields.names.len() {
                    if let Some(v) = (fields.get)(current, i) {
                        (fields.set)(&mut copy, &mut target, i, v);
                    }
                }
                out.push((id, target, copy));
                out.len() - 1
            }
        };
        let Some((_, target, copy)) = out.get_mut(at) else {
            continue;
        };
        if !(fields.set)(copy, target, field, value) {
            return Err(BindError::Argument(format!(
                "set_targets: {value} does not fit the field {name:?}"
            )));
        }
    }
    for (_, _, copy) in &out {
        EnvConfig::validate(copy).map_err(|e| BindError::Engine(e.into()))?;
    }
    Ok(out.into_iter().map(|(id, t, _)| (id, t)).collect())
}
