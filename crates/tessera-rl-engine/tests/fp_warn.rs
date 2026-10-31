// SPDX-License-Identifier: MIT OR Apache-2.0

//! ADR-0037 (c): with `debug-checks`, the warning about a non-canonical
//! caller is emitted inside the canonical environment, so subscriber code
//! never runs under the caller's controls (codex, R-PR14).

#![cfg(feature = "debug-checks")]

mod common;

use core::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use common::{CountConfig, Counter};
use tessera_rl_engine::{EnvPool, PoolOptions, Reseed};
use tessera_rl_num::fp;
use tessera_rl_num::identity::Seed;
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

/// Counts events, and events seen outside the canonical environment.
struct Probe {
    events: Arc<AtomicU32>,
    non_canonical: Arc<AtomicU32>,
}

impl Subscriber for Probe {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, _: &Event<'_>) {
        self.events.fetch_add(1, Ordering::Relaxed);
        if !fp::is_canonical() {
            self.non_canonical.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn enter(&self, _: &Id) {}

    fn exit(&self, _: &Id) {}
}

#[test]
fn test_non_canonical_entry_warning_runs_inside_the_canonical_environment() {
    let events = Arc::new(AtomicU32::new(0));
    let non_canonical = Arc::new(AtomicU32::new(0));
    let probe = Probe {
        events: Arc::clone(&events),
        non_canonical: Arc::clone(&non_canonical),
    };
    tracing::subscriber::with_default(probe, || {
        fp::with_hostile_flush_to_zero(|| {
            let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
            let mut p = EnvPool::<Counter>::new(CountConfig::default(), options).unwrap();
            p.reset_all(Reseed::Keep).unwrap();
        });
    });
    assert!(
        events.load(Ordering::Relaxed) >= 1,
        "no warning was emitted"
    );
    assert_eq!(
        non_canonical.load(Ordering::Relaxed),
        0,
        "a callback ran non-canonical"
    );
}
