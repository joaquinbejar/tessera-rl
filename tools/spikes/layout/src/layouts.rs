// SPDX-License-Identifier: MIT OR Apache-2.0

//! The storage layouts under comparison. Every layout runs the same
//! per-instance kernel; they differ only in where headers and states live.
//!
//! - [`Parallel`] (L0, ADR-0036): `[Header; N]` and `[State; N]`.
//! - [`Combined`] (L1): `[Record { header, state }; N]`.
//! - [`Columns`] (L2, SoA): headers as in L0, the state split into one
//!   column per field, gathered and scattered around the kernel.
//! - [`Blocks`] (L3, AoSoA): blocks of [`LANES`] states stored field by
//!   field, gathered and scattered around the kernel.
//!
//! A snapshot of one instance is two `memcpy`s in L0 (header, state), one
//! in L1, and a gather of every column in L2 and L3.

use tessera_rl_core::{Environment, RecordLayout, StateInit};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed};
use tessera_rl_num::rng::RngAuthority;

use crate::columns::Columnar;
use crate::kernel::{SpikeEnv, Work, reset, step};

/// Lanes per AoSoA block.
pub const LANES: usize = 8;

/// A storage layout for `n` instances of `E`.
pub trait Layout<E: SpikeEnv>: Sized {
    /// Short name for reports.
    const NAME: &'static str;
    /// `n` instances seeded `seed + i`, each reset.
    fn new(cfg: &E::Config, n: usize, seed: u64) -> Self;
    /// One step of every instance.
    fn step_all(&mut self, cfg: &E::Config, actions: &[i32], work: &mut Work, limit: u32);
    /// A copy of instance `i` (a snapshot).
    fn get(&self, i: usize) -> Option<(E::Header, E::State)>;
    /// Overwrites instance `i` (a restore or a fork destination).
    fn set(&mut self, i: usize, header: &E::Header, state: &E::State);
    /// Copies every instance into `dst` (same size), in this layout's own
    /// storage: whole-buffer copies, as a pool snapshot would do.
    fn copy_all(&self, dst: &mut Self);
}

fn fresh<E: Environment>(cfg: &E::Config, i: usize, seed: u64) -> (E::Header, E::State) {
    let mut header = E::Header::state_default();
    let logical = LogicalId::new(u32::try_from(i).unwrap_or(u32::MAX));
    header.engine_mut().identity = RngIdentity::new(
        Seed::new(seed.wrapping_add(u64::from(logical.get()))),
        logical,
    );
    let mut state = E::State::state_default();
    reset::<E>(
        cfg,
        &RngAuthority::engine_only(),
        &mut header,
        &mut state,
        0,
    );
    (header, state)
}

/// L0: parallel header and state arrays (ADR-0036).
#[derive(Debug)]
pub struct Parallel<E: Environment> {
    headers: Vec<E::Header>,
    states: Vec<E::State>,
}

impl<E: SpikeEnv> Layout<E> for Parallel<E> {
    const NAME: &'static str = "L0-parallel";

    fn new(cfg: &E::Config, n: usize, seed: u64) -> Self {
        let (headers, states) = (0..n).map(|i| fresh::<E>(cfg, i, seed)).unzip();
        Self { headers, states }
    }

    fn step_all(&mut self, cfg: &E::Config, actions: &[i32], work: &mut Work, limit: u32) {
        let authority = RngAuthority::engine_only();
        let a = E::MAX_AGENTS;
        for (i, (h, s)) in self.headers.iter_mut().zip(&mut self.states).enumerate() {
            let row = actions.get(i * a..(i + 1) * a).unwrap_or(&[]);
            step::<E>(cfg, &authority, h, s, row, work, i, limit);
        }
    }

    fn get(&self, i: usize) -> Option<(E::Header, E::State)> {
        Some((*self.headers.get(i)?, *self.states.get(i)?))
    }

    fn set(&mut self, i: usize, header: &E::Header, state: &E::State) {
        if let (Some(h), Some(s)) = (self.headers.get_mut(i), self.states.get_mut(i)) {
            *h = *header;
            *s = *state;
        }
    }

    fn copy_all(&self, dst: &mut Self) {
        // Two memcpys: headers, then states.
        dst.headers.copy_from_slice(&self.headers);
        dst.states.copy_from_slice(&self.states);
    }
}

/// One combined record.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Record<H, S> {
    header: H,
    state: S,
}

/// L1: combined records.
#[derive(Debug)]
pub struct Combined<E: Environment> {
    records: Vec<Record<E::Header, E::State>>,
}

impl<E: SpikeEnv> Layout<E> for Combined<E> {
    const NAME: &'static str = "L1-combined";

    fn new(cfg: &E::Config, n: usize, seed: u64) -> Self {
        let records = (0..n)
            .map(|i| {
                let (header, state) = fresh::<E>(cfg, i, seed);
                Record { header, state }
            })
            .collect();
        Self { records }
    }

    fn step_all(&mut self, cfg: &E::Config, actions: &[i32], work: &mut Work, limit: u32) {
        let authority = RngAuthority::engine_only();
        let a = E::MAX_AGENTS;
        for (i, r) in self.records.iter_mut().enumerate() {
            let row = actions.get(i * a..(i + 1) * a).unwrap_or(&[]);
            step::<E>(
                cfg,
                &authority,
                &mut r.header,
                &mut r.state,
                row,
                work,
                i,
                limit,
            );
        }
    }

    fn get(&self, i: usize) -> Option<(E::Header, E::State)> {
        let r = self.records.get(i)?;
        Some((r.header, r.state))
    }

    fn set(&mut self, i: usize, header: &E::Header, state: &E::State) {
        // Field by field, in place (no temporary record).
        if let Some(r) = self.records.get_mut(i) {
            r.header = *header;
            r.state = *state;
        }
    }

    fn copy_all(&self, dst: &mut Self) {
        dst.records.copy_from_slice(&self.records);
    }
}

/// L2: SoA state columns (headers as in L0).
#[derive(Debug)]
pub struct Columns<E: Environment>
where
    E::State: Columnar,
{
    headers: Vec<E::Header>,
    columns: <E::State as Columnar>::Columns,
}

impl<E: SpikeEnv> Layout<E> for Columns<E>
where
    E::State: Columnar,
{
    const NAME: &'static str = "L2-soa";

    fn new(cfg: &E::Config, n: usize, seed: u64) -> Self {
        let mut columns = <E::State as Columnar>::columns(n);
        let headers = (0..n)
            .map(|i| {
                let (h, s) = fresh::<E>(cfg, i, seed);
                s.scatter(&mut columns, i);
                h
            })
            .collect();
        Self { headers, columns }
    }

    fn step_all(&mut self, cfg: &E::Config, actions: &[i32], work: &mut Work, limit: u32) {
        let authority = RngAuthority::engine_only();
        let a = E::MAX_AGENTS;
        for (i, h) in self.headers.iter_mut().enumerate() {
            let row = actions.get(i * a..(i + 1) * a).unwrap_or(&[]);
            let mut s = <E::State as Columnar>::gather(&self.columns, i);
            step::<E>(cfg, &authority, h, &mut s, row, work, i, limit);
            s.scatter(&mut self.columns, i);
        }
    }

    fn get(&self, i: usize) -> Option<(E::Header, E::State)> {
        Some((
            *self.headers.get(i)?,
            <E::State as Columnar>::gather(&self.columns, i),
        ))
    }

    fn set(&mut self, i: usize, header: &E::Header, state: &E::State) {
        if let Some(h) = self.headers.get_mut(i) {
            *h = *header;
            state.scatter(&mut self.columns, i);
        }
    }

    fn copy_all(&self, dst: &mut Self) {
        dst.headers.copy_from_slice(&self.headers);
        <E::State as Columnar>::copy_columns(&self.columns, &mut dst.columns);
    }
}

/// L3: AoSoA, blocks of [`LANES`] states (headers as in L0).
#[derive(Debug)]
pub struct Blocks<E: Environment>
where
    E::State: Columnar,
{
    headers: Vec<E::Header>,
    blocks: Vec<<E::State as Columnar>::Columns>,
}

impl<E: SpikeEnv> Layout<E> for Blocks<E>
where
    E::State: Columnar,
{
    const NAME: &'static str = "L3-aosoa";

    fn new(cfg: &E::Config, n: usize, seed: u64) -> Self {
        let mut blocks: Vec<_> = (0..n.div_ceil(LANES))
            .map(|_| <E::State as Columnar>::columns(LANES))
            .collect();
        let headers = (0..n)
            .map(|i| {
                let (h, s) = fresh::<E>(cfg, i, seed);
                if let Some(b) = blocks.get_mut(i / LANES) {
                    s.scatter(b, i % LANES);
                }
                h
            })
            .collect();
        Self { headers, blocks }
    }

    fn step_all(&mut self, cfg: &E::Config, actions: &[i32], work: &mut Work, limit: u32) {
        let authority = RngAuthority::engine_only();
        let a = E::MAX_AGENTS;
        for (i, h) in self.headers.iter_mut().enumerate() {
            let Some(b) = self.blocks.get_mut(i / LANES) else {
                continue;
            };
            let row = actions.get(i * a..(i + 1) * a).unwrap_or(&[]);
            let mut s = <E::State as Columnar>::gather(b, i % LANES);
            step::<E>(cfg, &authority, h, &mut s, row, work, i, limit);
            s.scatter(b, i % LANES);
        }
    }

    fn get(&self, i: usize) -> Option<(E::Header, E::State)> {
        let b = self.blocks.get(i / LANES)?;
        Some((
            *self.headers.get(i)?,
            <E::State as Columnar>::gather(b, i % LANES),
        ))
    }

    fn set(&mut self, i: usize, header: &E::Header, state: &E::State) {
        if let (Some(h), Some(b)) = (self.headers.get_mut(i), self.blocks.get_mut(i / LANES)) {
            *h = *header;
            state.scatter(b, i % LANES);
        }
    }

    fn copy_all(&self, dst: &mut Self) {
        dst.headers.copy_from_slice(&self.headers);
        for (d, s) in dst.blocks.iter_mut().zip(&self.blocks) {
            <E::State as Columnar>::copy_columns(s, d);
        }
    }
}
