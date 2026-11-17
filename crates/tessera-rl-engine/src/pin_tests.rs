// SPDX-License-Identifier: MIT OR Apache-2.0

//! The pin decision and the active-engine registration (ADR-0005, ADR-0062
//! item 2), as pure functions, and the registration a started runtime
//! holds.

use core::num::NonZeroUsize;

use crate::error::EngineError;
use crate::options::SyncOptions;
use crate::pool::{Pinning, SyncRuntime, pin_plan, registers};

const CORES: &[usize] = &[0, 1, 2, 3];

#[test]
fn test_pin_plan_explicit_cpu_set_pins_for_both_kinds() {
    for kind in [Pinning::Auto, Pinning::ExplicitOnly] {
        for others in [0, 3] {
            assert_eq!(
                pin_plan(Some(vec![2, 3]), Some(CORES), 2, others, kind),
                Ok(Some(vec![2, 3]))
            );
        }
    }
}

#[test]
fn test_pin_plan_explicit_cpu_set_refused_when_short_or_unreported() {
    for kind in [Pinning::Auto, Pinning::ExplicitOnly] {
        assert!(matches!(
            pin_plan(Some(vec![0]), Some(CORES), 2, 0, kind),
            Err(EngineError::Options {
                option: "cpu_set",
                ..
            })
        ));
        assert!(matches!(
            pin_plan(Some(vec![0, 9]), Some(CORES), 2, 0, kind),
            Err(EngineError::Options {
                option: "cpu_set",
                ..
            })
        ));
        // No core ids reported (macOS): no explicit set can be honoured.
        assert!(matches!(
            pin_plan(Some(vec![0, 1]), None, 2, 0, kind),
            Err(EngineError::Options {
                option: "cpu_set",
                ..
            })
        ));
    }
}

#[test]
fn test_pin_plan_env_pool_auto_pins_only_alone_with_enough_cores() {
    assert_eq!(
        pin_plan(None, Some(CORES), 2, 0, Pinning::Auto),
        Ok(Some(vec![0, 1]))
    );
    assert_eq!(
        pin_plan(None, Some(CORES), 2, 1, Pinning::Auto),
        Ok(None),
        "another engine"
    );
    assert_eq!(
        pin_plan(None, Some(CORES), 5, 0, Pinning::Auto),
        Ok(None),
        "too few cores"
    );
    assert_eq!(
        pin_plan(None, None, 2, 0, Pinning::Auto),
        Ok(None),
        "no cores reported"
    );
}

#[test]
fn test_pin_plan_host_pool_without_cpu_set_never_pins() {
    for others in [0, 2] {
        for available in [Some(CORES), None] {
            assert_eq!(
                pin_plan(None, available, 2, others, Pinning::ExplicitOnly),
                Ok(None)
            );
        }
    }
}

#[test]
fn test_registers_every_env_pool_and_only_explicit_host_pools() {
    assert!(registers(Pinning::Auto, false));
    assert!(registers(Pinning::Auto, true));
    assert!(!registers(Pinning::ExplicitOnly, false));
    assert!(registers(Pinning::ExplicitOnly, true));
}

#[test]
fn test_sync_runtime_host_without_cpu_set_holds_no_registration() {
    let two = SyncOptions::new(NonZeroUsize::new(2).unwrap());
    let host = SyncRuntime::start(
        two.clone(),
        Pinning::ExplicitOnly,
        crate::sys::Faults::default(),
    )
    .unwrap();
    assert!(host.registration.is_none() && !host.pinned);
    let env = SyncRuntime::start(two, Pinning::Auto, crate::sys::Faults::default()).unwrap();
    assert!(env.registration.is_some());
}
