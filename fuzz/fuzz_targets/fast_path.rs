// SPDX-License-Identifier: MIT OR Apache-2.0

//! The validation fast path (env-step plan r3 section 3.3) against the slow
//! path: arbitrary sequences of steps (arbitrary actions), resets, restores,
//! forks and imports on a CartPole pool (one discrete head) and a Pendulum
//! pool (one continuous head), each run twice, once forced onto the slow path
//! before every operation. Both must return the same results and hold the
//! same records after every operation; nothing may panic.

#![no_main]

use core::num::NonZeroU32;

use libfuzzer_sys::fuzz_target;
use tessera_rl_classic::{CartPole, CartPoleConfig, Pendulum, PendulumConfig};
use tessera_rl_core::Environment;
use tessera_rl_engine::{ActionInput, AutoReset, DecodedRecord, EnvPool, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

const N: u32 = 4;

/// Reads bytes off the front of the fuzz input.
struct Bytes<'a>(&'a [u8]);

impl Bytes<'_> {
    fn byte(&mut self) -> Option<u8> {
        let (&b, rest) = self.0.split_first()?;
        self.0 = rest;
        Some(b)
    }

    fn word(&mut self) -> Option<[u8; 4]> {
        let w = self.0.get(..4)?;
        let out = [w[0], w[1], w[2], w[3]];
        self.0 = &self.0[4..];
        Some(out)
    }
}

/// One module per environment: the engine's bounds on `EnvPool` methods
/// are concrete per environment, so each pool type gets its own copy.
macro_rules! env_runner {
    ($m:ident, $env:ty) => {
        mod $m {
            use super::*;
            type E = $env;

            fn records(p: &EnvPool<E>) -> Vec<Vec<u8>> {
                (0..N)
                    .map(|i| p.snapshot(i).map(|r| r.to_bytes()).unwrap_or_default())
                    .collect()
            }

            /// Runs the operations of `data` on a fast-path pool and a slow-path pool of
            /// `E`; `continuous` selects which action buffer the fuzz words fill.
            pub(crate) fn run(
                cfg: &<E as Environment>::Config,
                data: &mut Bytes<'_>,
                continuous: bool,
            ) {
                let Some(mode) = data.byte() else { return };
                let options = PoolOptions {
                    auto_reset: if mode & 1 == 0 {
                        AutoReset::SameStep
                    } else {
                        AutoReset::NextStep
                    },
                    max_episode_steps: NonZeroU32::new(1 + u32::from(mode >> 4)),
                    ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(u64::from(mode)))
                };
                let (Ok(mut fast), Ok(mut slow)) = (
                    EnvPool::<E>::new(cfg.clone(), options),
                    EnvPool::<E>::new(cfg.clone(), options),
                ) else {
                    return;
                };
                let rows = usize::try_from(N).unwrap() * E::MAX_AGENTS;
                for _ in 0..64 {
                    let Some(op) = data.byte() else { return };
                    slow.forget_validation_fast_path();
                    let (rf, rs) = match op % 8 {
                        0 => (
                            format!("{:?}", fast.reset_all(Reseed::Keep).map(|_| ())),
                            format!("{:?}", slow.reset_all(Reseed::Keep).map(|_| ())),
                        ),
                        1 => {
                            let (src, dst) = (u32::from(op >> 3) % N, u32::from(op >> 5) % N);
                            let Ok(r) = fast.snapshot(src) else { continue };
                            (
                                format!("{:?}", fast.restore(dst, &r)),
                                format!("{:?}", slow.restore(dst, &r)),
                            )
                        }
                        2 => {
                            let (src, dst) = (u32::from(op >> 3) % N, u32::from(op >> 5) % N);
                            (
                                format!("{:?}", fast.fork(src, &[dst])),
                                format!("{:?}", slow.fork(src, &[dst])),
                            )
                        }
                        3 => {
                            let (src, dst) = (u32::from(op >> 3) % N, u32::from(op >> 5) % N);
                            let Ok(r) = fast.snapshot(src) else { continue };
                            let rec = DecodedRecord::<E>::from(r);
                            (
                                format!("{:?}", fast.import(&[dst], &[rec])),
                                format!("{:?}", slow.import(&[dst], &[rec])),
                            )
                        }
                        _ => {
                            // Mostly in-range actions, some arbitrary: each row takes a
                            // fuzz word; a byte decides whether it is used raw.
                            let mut d = Vec::new();
                            let mut c = Vec::new();
                            for _ in 0..rows {
                                let Some(w) = data.word() else { return };
                                let raw = w[3] & 0x80 != 0;
                                if continuous {
                                    c.push(if raw {
                                        f32::from_le_bytes(w)
                                    } else {
                                        f32::from(w[0] % 5) - 2.0
                                    });
                                } else {
                                    d.push(if raw {
                                        i32::from_le_bytes(w)
                                    } else {
                                        i32::from(w[0] % 2)
                                    });
                                }
                            }
                            let input = ActionInput {
                                discrete: &d,
                                continuous: &c,
                            };
                            (
                                format!("{:?}", fast.step(input).map(|_| ())),
                                format!("{:?}", slow.step(input).map(|_| ())),
                            )
                        }
                    };
                    assert_eq!(rf, rs, "results differ");
                    assert_eq!(records(&fast), records(&slow), "records differ");
                }
            }
        }
    };
}

env_runner!(cartpole, CartPole);
env_runner!(pendulum, Pendulum);

fuzz_target!(|data: &[u8]| {
    let mut bytes = Bytes(data);
    match bytes.byte() {
        Some(b) if b & 1 == 0 => cartpole::run(&CartPoleConfig::default(), &mut bytes, false),
        Some(_) => pendulum::run(&PendulumConfig::default(), &mut bytes, true),
        None => {}
    }
});
