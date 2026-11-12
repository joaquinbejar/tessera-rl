// SPDX-License-Identifier: MIT OR Apache-2.0

//! The telemetry gate (ADR-0057 section 4, owner decision of 2026-11-12):
//! the runtime is never loaded unless `ORT_DISABLE_TELEMETRY=1` is already
//! in the process environment, or the caller accepts telemetry explicitly.
//! The environment is only read, never changed.

use std::ffi::OsStr;

use crate::error::OnnxError;

/// The variable ONNX Runtime reads to disable its telemetry.
pub const TELEMETRY_VARIABLE: &str = "ORT_DISABLE_TELEMETRY";

/// Whether the runtime may be loaded with its telemetry enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TelemetryPolicy {
    /// Load only when `ORT_DISABLE_TELEMETRY=1` is already set.
    #[default]
    RequireDisabled,
    /// Load even when it is not set: ONNX Runtime may then queue telemetry
    /// events (spike S1). The documented opt-out.
    Accept,
}

/// Applies the gate to the process environment. Returns whether telemetry
/// may be enabled in the runtime about to be loaded.
pub(crate) fn gate(policy: TelemetryPolicy) -> Result<bool, OnnxError> {
    gate_with(policy, std::env::var_os(TELEMETRY_VARIABLE).as_deref())
}

/// The gate for a given value of the variable.
pub(crate) fn gate_with(policy: TelemetryPolicy, value: Option<&OsStr>) -> Result<bool, OnnxError> {
    let disabled = value == Some(OsStr::new("1"));
    match (disabled, policy) {
        (true, _) => Ok(false),
        (false, TelemetryPolicy::Accept) => Ok(true),
        (false, TelemetryPolicy::RequireDisabled) => Err(OnnxError::TelemetryNotDisabled {
            variable: TELEMETRY_VARIABLE,
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use super::{TELEMETRY_VARIABLE, TelemetryPolicy, gate_with};
    use crate::error::OnnxError;

    #[test]
    fn test_telemetry_gate_unset_require_is_refused() {
        assert_eq!(
            gate_with(TelemetryPolicy::RequireDisabled, None),
            Err(OnnxError::TelemetryNotDisabled {
                variable: TELEMETRY_VARIABLE
            })
        );
    }

    #[test]
    fn test_telemetry_gate_other_values_are_refused() {
        for v in ["0", "true", "", " 1", "1 "] {
            assert!(
                gate_with(TelemetryPolicy::RequireDisabled, Some(OsStr::new(v))).is_err(),
                "{v:?}"
            );
        }
    }

    #[test]
    fn test_telemetry_gate_set_is_off_under_both_policies() {
        let one = Some(OsStr::new("1"));
        assert_eq!(gate_with(TelemetryPolicy::RequireDisabled, one), Ok(false));
        assert_eq!(gate_with(TelemetryPolicy::Accept, one), Ok(false));
    }

    #[test]
    fn test_telemetry_gate_unset_accept_is_on() {
        assert_eq!(gate_with(TelemetryPolicy::Accept, None), Ok(true));
    }

    #[test]
    fn test_telemetry_default_policy_requires_disabled() {
        assert_eq!(TelemetryPolicy::default(), TelemetryPolicy::RequireDisabled);
    }
}
