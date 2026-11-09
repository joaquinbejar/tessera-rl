//! The lane CartPole step against `physics::step`, bit for bit, on every
//! level this CPU has; and mutants the differential must catch.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::manual_is_multiple_of,
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    reason = "test vectors and helpers"
)]

use tessera_rl_classic::cartpole::lane::{self, Columns, LaneOutcome};
use tessera_rl_classic::cartpole::physics::{self, FORCE_MAG, GRAVITY, LENGTH, MASS_POLE, TAU};
use tessera_rl_num::fp::{with_canonical, with_hostile_flush_to_zero};
use tessera_rl_num::math::lane::{self as mlane, F64Lanes};
use tessera_rl_simd::{Kernel, Resolved};

const POLEMASS_LENGTH: f64 = 0.05;
const TOTAL_MASS: f64 = 1.1;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 ^ self.0 >> 29
    }
    fn range(&mut self, half: f64) -> f64 {
        ((self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * half
    }
}

fn rows_to_columns(rows: &[[f64; 4]; 8]) -> Columns {
    let mut c = [[0.0; 8]; 4];
    for (l, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            c[j][l] = *v;
        }
    }
    c
}

/// Mismatches of `step` against the reference over these batches.
fn mismatches(
    batches: &[([[f64; 4]; 8], [bool; 8])],
    step: impl Fn(&mut Columns, [bool; 8]) -> LaneOutcome,
) -> usize {
    let mut bad = 0;
    for (rows, push) in batches {
        let mut columns = rows_to_columns(rows);
        let out = with_canonical(|| step(&mut columns, *push));
        for l in 0..8 {
            let (want, term) = with_canonical(|| {
                let w = physics::step(rows[l], push[l]);
                (w, physics::is_terminal(w))
            });
            for j in 0..4 {
                let cast = with_canonical(|| want[j] as f32);
                if columns[j][l].to_bits() != want[j].to_bits()
                    || out.observations[j][l].to_bits() != cast.to_bits()
                {
                    bad += 1;
                }
            }
            if (out.terminal >> l & 1 == 1) != term {
                bad += 1;
            }
        }
    }
    bad
}

fn live_batches(n: usize, seed: u64) -> Vec<([[f64; 4]; 8], [bool; 8])> {
    let mut r = Rng(seed);
    (0..n)
        .map(|_| {
            let mut rows = [[0.0; 4]; 8];
            let mut push = [false; 8];
            for l in 0..8 {
                rows[l] = [r.range(2.4), r.range(5.0), r.range(0.21), r.range(5.0)];
                push[l] = r.next() & 1 == 1;
            }
            (rows, push)
        })
        .collect()
}

const EDGES: &[f64] = &[
    0.0,
    -0.0,
    1.0,
    5e-324,
    -5e-324,
    2.2250738585072014e-308,
    2.4,
    -2.4,
    2.4000000000000004,
    -2.4000000000000004,
    physics::THETA_THRESHOLD,
    -physics::THETA_THRESHOLD,
    f64::MAX,
    f64::MIN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    4294967296.0,
    f64::from_bits(0x41f0_0000_0000_0001),
    8589934592.0,
    -4503599627370496.0,
    1e300,
];

const THETA_WORDS: [u64; 6] = [
    0x3fe9_21fb_ffff_ffff,
    0x3fe9_21fc_0000_0000,
    0x3e4f_ffff_ffff_ffff,
    0x3e50_0000_0000_0000,
    0x3e46_a09d_ffff_ffff,
    0x3e46_a09e_0000_0000,
];

/// Finite states whose scalar result is well defined bit for bit. NaN results
/// are compared separately.
fn edge_batches(seed: u64, n: usize) -> Vec<([[f64; 4]; 8], [bool; 8])> {
    let mut r = Rng(seed);
    let pick = |r: &mut Rng| {
        if r.next() % 4 == 0 {
            let w = THETA_WORDS[(r.next() % 6) as usize];
            let v = f64::from_bits(w);
            if r.next() & 1 == 1 { -v } else { v }
        } else {
            EDGES[(r.next() % EDGES.len() as u64) as usize]
        }
    };
    (0..n)
        .map(|_| {
            let mut rows = [[0.0; 4]; 8];
            let mut push = [false; 8];
            for l in 0..8 {
                // Mixed lanes: ordinary, extreme and tail together.
                rows[l] = if r.next() & 1 == 0 {
                    [r.range(2.4), r.range(5.0), r.range(0.21), r.range(5.0)]
                } else {
                    [pick(&mut r), pick(&mut r), pick(&mut r), pick(&mut r)]
                };
                push[l] = r.next() & 1 == 1;
            }
            (rows, push)
        })
        .collect()
}

#[test]
fn test_lanes_live_states_match_physics_on_every_level() {
    let batches = live_batches(50_000, 11);
    for level in Resolved::available() {
        assert_eq!(
            mismatches(&batches, |c, p| lane::step(level, c, p)),
            0,
            "{}",
            level.level()
        );
    }
}

#[test]
fn test_lanes_edge_vectors_match_physics_on_every_level() {
    let batches = edge_batches(29, 50_000);
    for level in Resolved::available() {
        assert_eq!(
            mismatches(&batches, |c, p| lane::step(level, c, p)),
            0,
            "{}",
            level.level()
        );
    }
}

#[test]
fn test_lanes_canonical_scope_overrides_hostile_flush_to_zero() {
    // The inner scope is the caller's canonical mode; subnormals must survive.
    let batches = edge_batches(31, 5_000);
    for level in Resolved::available() {
        let bad =
            with_hostile_flush_to_zero(|| mismatches(&batches, |c, p| lane::step(level, c, p)));
        assert_eq!(bad, 0, "{}", level.level());
    }
}

#[test]
fn test_lanes_trajectory_matches_physics() {
    for level in Resolved::available() {
        let mut rows = live_batches(1, 3)[0]
            .0
            .map(|r| [r[0] * 0.01, r[1] * 0.01, r[2] * 0.1, r[3] * 0.01]);
        let mut columns = rows_to_columns(&rows);
        let mut r = Rng(5);
        for _ in 0..500 {
            let push: [bool; 8] = core::array::from_fn(|_| r.next() & 1 == 1);
            let _ = with_canonical(|| lane::step(level, &mut columns, push));
            for l in 0..8 {
                rows[l] = with_canonical(|| physics::step(rows[l], push[l]));
            }
            assert_eq!(
                columns.map(|c| c.map(f64::to_bits)),
                rows_to_columns(&rows).map(|c| c.map(f64::to_bits))
            );
        }
    }
}

// -- mutants: the differential must report each one, on every level ----------

#[derive(Clone, Copy)]
enum Mutation {
    /// `a * (b * c)` where the reference has `(a * b) * c`.
    Reorder,
    /// A fused multiply-add where the reference rounds twice.
    Fused,
}

struct Mutant<'a>(&'a mut Columns, [bool; 8], Mutation);

impl Kernel for Mutant<'_> {
    type Output = LaneOutcome;

    fn run<L: F64Lanes>(self) -> LaneOutcome {
        let k = L::splat;
        let [x, x_dot, theta, theta_dot] = self.0.map(L::from_array);
        let force = L::select(L::mask_from_array(self.1), k(FORCE_MAG), k(-FORCE_MAG));
        let (c, s) = (mlane::cos(theta), mlane::sin(theta));
        let pushed = match self.2 {
            Mutation::Reorder => k(POLEMASS_LENGTH) * ((theta_dot * theta_dot) * s),
            Mutation::Fused => k(POLEMASS_LENGTH) * (theta_dot * theta_dot) * s,
        };
        let temp = (force + pushed) / k(TOTAL_MASS);
        let acc = (k(GRAVITY) * s - c * temp)
            / (k(LENGTH) * (k(4.0 / 3.0) - k(MASS_POLE) * (c * c) / k(TOTAL_MASS)));
        let xacc = temp - k(POLEMASS_LENGTH) * acc * c / k(TOTAL_MASS);
        let fused = |a: L, b: L| -> L {
            let (a, b) = (a.to_array(), b.to_array());
            let mut out = [0.0; 8];
            for i in 0..8 {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "the fused mutant the differential must catch"
                )]
                {
                    out[i] = TAU.mul_add(b[i], a[i]);
                }
            }
            L::from_array(out)
        };
        let next = match self.2 {
            Mutation::Reorder => [
                x + k(TAU) * x_dot,
                x_dot + k(TAU) * xacc,
                theta + k(TAU) * theta_dot,
                theta_dot + k(TAU) * acc,
            ],
            Mutation::Fused => [
                fused(x, x_dot),
                fused(x_dot, xacc),
                fused(theta, theta_dot),
                fused(theta_dot, acc),
            ],
        };
        *self.0 = next.map(L::to_array);
        let observations = next.map(L::to_f32_array);
        LaneOutcome {
            observations,
            terminal: 0,
        }
    }
}

#[test]
fn test_lanes_differential_catches_a_changed_operation_order() {
    let batches = live_batches(2_000, 7);
    for level in Resolved::available() {
        let bad = mismatches(&batches, |c, p| {
            level.dispatch(Mutant(c, p, Mutation::Reorder))
        });
        assert!(bad > 0, "{}", level.level());
    }
}

#[test]
fn test_lanes_differential_catches_a_fused_operation() {
    let batches = live_batches(2_000, 7);
    for level in Resolved::available() {
        let bad = mismatches(&batches, |c, p| {
            level.dispatch(Mutant(c, p, Mutation::Fused))
        });
        assert!(bad > 0, "{}", level.level());
    }
}
