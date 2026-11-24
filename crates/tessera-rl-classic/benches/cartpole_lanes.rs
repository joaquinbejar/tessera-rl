// SPDX-License-Identifier: MIT OR Apache-2.0

//! CartPole lane gain per SIMD level (issue #152 P5, PR 1; design
//! `docs/reviews/152-p5/DESIGN-r2.md` section 2.1).
//!
//! Every bench steps the same 256 groups of 8 live states, in the fast
//! domain, once per iteration. An instance whose next state is terminal is
//! reset to its initial state, as a pool's auto-reset would, so the states
//! stay in the domain across iterations; both arms do the same resets.
//!
//! - `scalar_group_x8`: per group, 8 times `physics::step`, `is_terminal`
//!   and the observation's f32 cast, in place over `[CartPoleState; 8]`.
//! - `lane_group/<level>`: per group, the gather of the 8 states and actions
//!   into columns, one `lane::step`, the scatter back into the array of structs, then
//!   per instance the `is_terminal` and f32 cast recompute the pool keeps.
//!   The honest group gain `g` is `scalar_group_x8 / lane_group/<level>`.
//! - `scalar_kernel_x8` and `lane_kernel/<level>`: the bare kernels over
//!   columns, no gather or scatter; `g_kernel` is reported, not ruled on.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_classic::CartPoleState;
use tessera_rl_classic::cartpole::lane::{self, Columns};
use tessera_rl_classic::cartpole::physics;
use tessera_rl_num::fp::with_canonical;
use tessera_rl_simd::Resolved;

const GROUPS: usize = 256;

/// A small deterministic generator for the inputs.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[-0.05, 0.05)`: CartPole's reset range, in the fast domain.
    fn small(&mut self) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "53 random bits, exact")]
        let unit = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        unit * 0.1 - 0.05
    }
}

struct Inputs {
    initial: Vec<[CartPoleState; 8]>,
    actions: Vec<[i32; 8]>,
}

fn inputs() -> Inputs {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let initial = (0..GROUPS)
        .map(|_| {
            core::array::from_fn(|_| CartPoleState {
                s: core::array::from_fn(|_| rng.small()),
            })
        })
        .collect();
    let actions = (0..GROUPS)
        .map(|_| core::array::from_fn(|_| i32::from(rng.next() & 1 == 1)))
        .collect();
    Inputs { initial, actions }
}

/// The per-instance work the pool keeps after the kernel: the end check and
/// the observation cast; a terminal instance is reset.
#[inline]
fn after_kernel(state: &mut CartPoleState, initial: &CartPoleState) -> [f32; 4] {
    #[expect(clippy::cast_possible_truncation, reason = "the observation cast")]
    let obs = state.s.map(|v| v as f32);
    if physics::is_terminal(state.s) {
        *state = *initial;
    }
    obs
}

fn scalar_group(states: &mut [CartPoleState; 8], actions: &[i32; 8], initial: &[CartPoleState; 8]) {
    for ((state, &a), init) in states.iter_mut().zip(actions).zip(initial) {
        state.s = physics::step(state.s, a == 1);
        black_box(after_kernel(state, init));
    }
}

fn lane_group(
    level: Resolved,
    states: &mut [CartPoleState; 8],
    actions: &[i32; 8],
    initial: &[CartPoleState; 8],
) {
    // Gather: AoS rows into columns, actions into the lane mask.
    let mut columns: Columns = [[0.0; 8]; 4];
    for (lane, state) in states.iter().enumerate() {
        for (column, &value) in columns.iter_mut().zip(&state.s) {
            if let Some(c) = column.get_mut(lane) {
                *c = value;
            }
        }
    }
    let push_right = actions.map(|a| a == 1);
    black_box(lane::step(level, &mut columns, push_right));
    // Scatter, then the per-instance recompute the pool keeps.
    for ((lane, state), init) in states.iter_mut().enumerate().zip(initial) {
        for (value, column) in state.s.iter_mut().zip(&columns) {
            if let Some(&c) = column.get(lane) {
                *value = c;
            }
        }
        black_box(after_kernel(state, init));
    }
}

fn bench(c: &mut Criterion) {
    let Inputs { initial, actions } = inputs();
    let mut group = c.benchmark_group("cartpole_lanes");

    let mut states = initial.clone();
    group.bench_function("scalar_group_x8", |b| {
        b.iter(|| {
            with_canonical(|| {
                for ((s, a), i) in states.iter_mut().zip(&actions).zip(&initial) {
                    scalar_group(s, a, i);
                }
            });
        });
    });

    for level in Resolved::available() {
        let mut states = initial.clone();
        group.bench_function(format!("lane_group/{}", level.level()), |b| {
            b.iter(|| {
                with_canonical(|| {
                    for ((s, a), i) in states.iter_mut().zip(&actions).zip(&initial) {
                        lane_group(level, s, a, i);
                    }
                });
            });
        });
    }

    // Bare kernels over columns, from the same initial states each iteration.
    let columns: Vec<Columns> = initial
        .iter()
        .map(|g| {
            core::array::from_fn(|f| {
                core::array::from_fn(|l| {
                    g.get(l).and_then(|st| st.s.get(f)).copied().unwrap_or(0.0)
                })
            })
        })
        .collect();
    let pushes: Vec<[bool; 8]> = actions.iter().map(|a| a.map(|x| x == 1)).collect();
    let mut work = columns.clone();
    group.bench_function("scalar_kernel_x8", |b| {
        b.iter(|| {
            work.clone_from(&columns);
            with_canonical(|| {
                for (cols, push) in work.iter_mut().zip(&pushes) {
                    for (lane, &p) in push.iter().enumerate() {
                        let row = cols.map(|c| c.get(lane).copied().unwrap_or(0.0));
                        let next = physics::step(row, p);
                        for (column, value) in cols.iter_mut().zip(next) {
                            if let Some(c) = column.get_mut(lane) {
                                *c = value;
                            }
                        }
                        black_box(physics::is_terminal(next));
                    }
                }
            });
        });
    });
    for level in Resolved::available() {
        let mut work = columns.clone();
        group.bench_function(format!("lane_kernel/{}", level.level()), |b| {
            b.iter(|| {
                work.clone_from(&columns);
                with_canonical(|| {
                    for (cols, push) in work.iter_mut().zip(&pushes) {
                        black_box(lane::step(level, cols, *push));
                    }
                });
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
