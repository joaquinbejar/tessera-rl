// SPDX-License-Identifier: MIT OR Apache-2.0

//! The order book's `head_active` declaration (ADR-0086): the step
//! ignores inactive heads (`check_head_activity` on sampled records), and
//! the canonical legal joints map one-to-one onto `decide`'s distinct
//! decisions, so search has exactly one edge per game action.

#![expect(
    clippy::unwrap_used,
    reason = "test fixtures: a failure is the assertion"
)]

mod support;

use core::num::NonZeroU32;
use std::collections::{BTreeMap, BTreeSet};

use support::{Rng, config, input, legal_actions, pool};
use tessera_rl_core::space::JointHeads;
use tessera_rl_engine::Reseed;
use tessera_rl_markets::actions::{self, CANCELS, Context, HEADS, KINDS, MASK_LEN, OFFSETS, SIZES};
use tessera_rl_markets::limits::MAX_AGENTS;
use tessera_rl_markets::{LimitOrderBook, LobConfig};
use tessera_rl_search::check_head_activity;

fn joint() -> JointHeads {
    let parents: Vec<_> = (0..HEADS).map(actions::head_parents).collect();
    JointHeads::new(
        &[KINDS, OFFSETS, SIZES, CANCELS],
        &parents,
        u32::MAX,
        |h, p| actions::head_active(h, p[0]),
    )
    .unwrap()
}

/// Records along a seeded legal trajectory: every `every` steps.
fn roots(
    cfg: &LobConfig,
    seed: u64,
    steps: usize,
    every: usize,
) -> Vec<tessera_rl_engine::Record<LimitOrderBook>> {
    let mut p = pool(cfg.clone(), 1, seed, None);
    let mut masks = p.reset_all(Reseed::Keep).unwrap().masks().to_vec();
    let mut rng = Rng(seed ^ 0x9e37);
    let mut out = vec![p.snapshot(0).unwrap()];
    for step in 1..=steps {
        let a = legal_actions(&mut rng, &masks, 1);
        masks = p.step(input(&a)).unwrap().masks().to_vec();
        if step % every == 0 {
            out.push(p.snapshot(0).unwrap());
        }
    }
    out
}

#[test]
fn test_lob_head_activity_step_ignores_inactive_heads() {
    let cfg = config(4, 2);
    let roots = roots(&cfg, 31, 24, 8);
    let report =
        check_head_activity::<LimitOrderBook>(&cfg, &roots, NonZeroU32::new(3).unwrap(), 0x1ead)
            .unwrap();
    assert_eq!(report.roots, roots.len());
    assert!(report.cases > 100, "{report:?}");
    assert!(report.with_inactive > 0, "{report:?}");
}

#[test]
fn test_lob_canonical_legal_joints_map_one_to_one_onto_decisions() {
    let joint = joint();
    assert_eq!(joint.joint(), 1728);
    assert_eq!(joint.mask_len(), MASK_LEN);
    let mut checked = 0usize;
    for (seed, latency) in [(5u64, 0u32), (17, 2)] {
        let cfg = config(4, latency);
        for root in roots(&cfg, seed, 40, 5) {
            let mut p = pool(cfg.clone(), 1, seed, None);
            p.reset_all(Reseed::Keep).unwrap();
            p.restore(0, &root).unwrap();
            let masks = p.outputs().unwrap().masks().to_vec();
            for agent in 0..usize::try_from(cfg.agents).unwrap().min(MAX_AGENTS) {
                let row = &masks[agent * MASK_LEN..(agent + 1) * MASK_LEN];
                let ctx = Context::new(&cfg, root.state(), agent);
                // Every legal decision, from any joint (canonical or not).
                let mut all = BTreeSet::new();
                let mut v = [0i32; HEADS];
                for j in 0..joint.joint() {
                    joint.decode(j, &mut v);
                    if let Some(d) = actions::decide(&cfg, &ctx, v) {
                        all.insert(format!("{d:?}"));
                    }
                }
                // The canonical legal joints: one per decision.
                let mut canonical = BTreeMap::new();
                for j in 0..joint.joint() {
                    if !joint.is_legal(j, |at| row[at].get()) {
                        continue;
                    }
                    joint.decode(j, &mut v);
                    let d = actions::decide(&cfg, &ctx, v)
                        .unwrap_or_else(|| panic!("legal joint {v:?} has no decision"));
                    assert!(
                        canonical.insert(format!("{d:?}"), j).is_none(),
                        "two canonical joints for {d:?}"
                    );
                }
                assert_eq!(
                    canonical.keys().collect::<Vec<_>>(),
                    all.iter().collect::<Vec<_>>(),
                    "agent {agent}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked >= 40, "{checked}");
}
