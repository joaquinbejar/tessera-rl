//! The study's rule-conformance check (#639, design r2 section 5 step 3).
//! `grade.sh` copies this file into the graded crate as
//! `tests/study_conform.rs`, with `__LIB__` replaced by the crate's library
//! name, and runs it with:
//! - `STUDY_SHEET`: `a`, `b` or `c`, which picks the compared fields
//!   (design r2 section 3.1);
//! - `STUDY_GOLDEN`: the reference's trajectory, study/grading/golden/<sheet>.txt;
//! - `STUDY_BLESS=1` to record that file from a reference instead.
//!
//! Fixed seeds, a fixed action script (a hashed choice per row, replaced by
//! the row's first legal action when the mask forbids it) and fixed pool
//! sizes make the trajectory a function of the environment alone.

use core::num::NonZeroU32;
use std::fmt::Write as _;

use __LIB__::StudyEnv;
use tessera_rl_core::Environment;
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

/// The graded pool.
type Pool = EnvPool<StudyEnv, Masked>;

/// Pool seeds.
const SEEDS: [u64; 3] = [1, 2, 639];
/// Instances per pool.
const INSTANCES: u32 = 8;
/// Steps after the first reset.
const CALLS: u32 = 120;

/// splitmix64, as in the scaffold's checks.
const fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The bits of `flags` as `0` and `1`.
fn bits(flags: &[tessera_rl_core::Bool8]) -> String {
    flags
        .iter()
        .map(|f| if f.get() { '1' } else { '0' })
        .collect()
}

/// One line per call: the fields of `sheet`.
fn line(p: &Pool, sheet: &str, seed: u64, call: u32) -> String {
    let out = p.outputs().unwrap();
    let mut s = format!("seed {seed} call {call}");
    if sheet != "c" {
        let obs: String = out
            .observation(0)
            .unwrap()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        write!(s, " obs {obs}").unwrap();
    }
    let rewards: Vec<String> = out
        .rewards()
        .iter()
        .map(|r| format!("{:08x}", r.to_bits()))
        .collect();
    write!(s, " rew {}", rewards.join(",")).unwrap();
    write!(s, " term {}", bits(out.episode_terminated())).unwrap();
    write!(s, " trunc {}", bits(out.episode_truncated())).unwrap();
    if sheet != "a" {
        write!(s, " mask {}", bits(out.masks())).unwrap();
    }
    s
}

/// The action script of call `t`.
fn actions(p: &Pool, seed: u64, t: u32) -> Vec<i32> {
    let masks = p.outputs().unwrap().masks();
    let rows = usize::try_from(INSTANCES).unwrap() * <StudyEnv as Environment>::MAX_AGENTS;
    let width = masks.len() / rows;
    (0..rows)
        .map(|row| {
            let legal = |a: usize| masks.get(row * width + a).is_some_and(|m| m.get());
            let key = u64::try_from(row).unwrap() ^ (u64::from(t) << 32) ^ seed;
            let pick = usize::try_from(mix(key) % u64::try_from(width).unwrap()).unwrap();
            let a = if legal(pick) {
                pick
            } else {
                (0..width).find(|&a| legal(a)).unwrap_or(0)
            };
            i32::try_from(a).unwrap()
        })
        .collect()
}

/// The trajectory text of `sheet`.
fn trajectory(sheet: &str) -> String {
    let mut text = String::new();
    for seed in SEEDS {
        let options = PoolOptions::new(NonZeroU32::new(INSTANCES).unwrap(), Seed::new(seed));
        let mut p = Pool::with_masks(Default::default(), options).unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        writeln!(text, "{}", line(&p, sheet, seed, 0)).unwrap();
        for t in 1..=CALLS {
            let acts = actions(&p, seed, t);
            p.step(ActionInput {
                discrete: &acts,
                continuous: &[],
            })
            .unwrap();
            writeln!(text, "{}", line(&p, sheet, seed, t)).unwrap();
        }
    }
    text
}

#[test]
fn test_study_conformance_matches_the_reference_trajectory() {
    let sheet = std::env::var("STUDY_SHEET").expect("STUDY_SHEET: a, b or c");
    assert!(
        ["a", "b", "c"].contains(&sheet.as_str()),
        "STUDY_SHEET: a, b or c"
    );
    let golden = std::env::var("STUDY_GOLDEN").expect("STUDY_GOLDEN: the reference trajectory");
    let got = trajectory(&sheet);
    if std::env::var_os("STUDY_BLESS").is_some_and(|v| v == "1") {
        std::fs::write(&golden, &got).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&golden).expect("the reference trajectory");
    if let Some((k, (g, w))) = got
        .lines()
        .zip(want.lines())
        .enumerate()
        .find(|(_, (g, w))| g != w)
    {
        let field = ["obs", "rew", "term", "trunc", "mask"]
            .into_iter()
            .find(|f| {
                let pick = |l: &str| {
                    l.split(&format!(" {f} "))
                        .nth(1)
                        .map(|r| r.split(' ').next().unwrap_or("").to_owned())
                };
                pick(g) != pick(w)
            })
            .unwrap_or("line");
        panic!(
            "conformance: line {} differs in `{field}`\n got: {g}\nwant: {w}",
            k + 1
        );
    }
    assert_eq!(
        got.lines().count(),
        want.lines().count(),
        "conformance: trajectory length"
    );
}
